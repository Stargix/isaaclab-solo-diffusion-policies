# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect expert demonstrations from trained RSL-RL policies for Diffusion Policy training.

The active schema is ``single`` expert collection. ``phase_a_closed_loop``
records a fixed procedural route plus the action of a route-aware command
teacher wrapped around an existing locomotion expert.  Route generation is
defined by a command capability envelope supplied on the command line, never by
the name of a particular skill/checkpoint.

HDF5 schema (per demo, under ``data/demo_<k>``)::

    data/demo_<k>/
        obs/
            joint_pos          (T, 12)   float32   -- measured joint positions
            joint_vel          (T, 12)   float32   -- measured joint velocities
            base_ang_vel       (T, 3)    float32   -- base angular velocity (body frame)
            projected_gravity  (T, 3)    float32   -- gravity in body frame (IMU orientation)
            last_action        (T, 12)   float32   -- action executed at t-1 (zeros at t0)
            root_pos_w         (T, 3)    float32   -- base position in world (viz only)
            root_quat_w        (T, 4)    float32   -- base orientation (WXYZ, world; viz only)
            command_speed      (T, 3)    float32   -- commanded (vx, vy, wz)
            desired_base_height(T, 1)    float32   -- commanded base height, not measured height
            reference_pos_w    (T, 3)    float32   -- command-integrated desired base route
            reference_yaw_w    (T, 1)    float32   -- desired world yaw of that route
            reference_command  (T, 3)    float32   -- command used to integrate the route
        actions                (T, 12)   float32   -- expert action executed at step t
        dones                  (T,)      bool      -- True only on the last step of the demo
        skill_idx              (T,)      int8      -- index into data.attrs["skill_names"]
        attrs:
            num_samples        = T
            skills_sequence    = [skill_a, skill_b, ...]  (order of first appearance)

    data.attrs:
        skill_names = [name_0, name_1, ...]  (maps skill_idx -> skill name)
        convention  = "aligned: obs[t] is the state BEFORE executing actions[t]; last_action[t] = actions[t-1]"

Alignment convention (important for the DataLoader):
    Each recorded step ``t`` stores the proprioceptive state of the robot *before*
    the action is executed together with the action that the expert takes from
    that state. Therefore ``obs[t]`` and ``actions[t]`` are aligned in time and
    ``last_action[t] == actions[t-1]`` (with ``last_action[0] == 0``).

Example usage::

    # Phase A: closed-loop route teacher around a single expert
    python scripts/diffusion_policy/data/collect_data.py --mode single --task="solo12-crouch-v0" \
        --checkpoint checkpoints/crouch_exponential.pt --skill_name crouch --desired_base_height 0.1705 \
        --route_profile phase_a_closed_loop --include_warmup_frames --num_envs 128 --num_steps 1000000 \
        --output_name crouch_phase_a_closed_loop_v1.hdf5 --headless
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch

from capability_routes import (
    CapabilityLimits,
    ReferenceRoute,
    ROUTE_FAMILIES,
    TrackerGains,
    generate_capability_route,
    tracking_command,
)

ROUTE_FAMILIES_TO_IDX = {name: index for index, name in enumerate(ROUTE_FAMILIES)}

# --------------------------------------------------------------------------- #
# Path setup: expose the upstream rsl_rl scripts so Hydra task resolution works.
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

# Launch Omniverse / Isaac Sim app BEFORE importing sim-only modules.
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Collect expert trajectory demonstrations in parallel.")
parser.add_argument("--mode", type=str, choices=["single", "chained"], default="single",
                    help="Collection mode: single policy (Dataset A) or skill chaining (Dataset B).")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g. solo12-v0).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to expert checkpoint (required for single mode).")
parser.add_argument("--skill_name", type=str, default=None,
                    help="Explicit expert name (for example sprint or crouch); avoids filename-based inference.")
parser.add_argument("--checkpoints", type=str, nargs="+", default=[],
                    help="Paths to expert checkpoints for chained mode (walk crouch jump ...).")
parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=50000,
                    help="Total number of timesteps to SAVE across all kept episodes.")
parser.add_argument("--output_name", type=str, default="raw_dataset.hdf5", help="Output file name.")
parser.add_argument("--seed", type=int, default=42, help="Seed for command/skill sampling and collection metadata.")
parser.add_argument("--command_resample_time_s", type=float, default=2.0,
                    help="Time interval (s) to resample speed commands within an episode.")
parser.add_argument("--command_profile", choices=["native", "shared_height"], default="native",
                    help="native: each expert's envelope; shared_height: common walk/crouch envelope for a fair posture-height ablation.")
parser.add_argument(
    "--route_profile",
    choices=["random_velocity", "phase_a", "phase_a_closed_loop"],
    default="random_velocity",
    help=(
        "random_velocity preserves the legacy velocity dataset. phase_a preserves the old "
        "walk-only command-integrated route collector. phase_a_closed_loop generates a fixed "
        "capability-bounded route and labels it with a route-aware command teacher."
    ),
)
parser.add_argument(
    "--startup_hold_steps",
    type=int,
    default=25,
    help="Phase-A route modes: zero-command steps after reset before motion (50 Hz).",
)
parser.add_argument(
    "--include_warmup_frames",
    action="store_true",
    help=(
        "Keep reset/startup frames instead of discarding them. Required for Phase-A routes so the "
        "offline policy contains the padded-zero deployment history."
    ),
)
parser.add_argument("--route_vx_min", type=float, default=0.10,
                    help="Closed-loop Phase A: minimum nominal forward speed (m/s).")
parser.add_argument("--route_vx_max", type=float, default=0.45,
                    help="Closed-loop Phase A: maximum command magnitude in x (m/s).")
parser.add_argument("--route_vy_abs_max", type=float, default=0.30,
                    help="Closed-loop Phase A: maximum corrective lateral command magnitude (m/s).")
parser.add_argument("--route_wz_abs_max", type=float, default=0.50,
                    help="Closed-loop Phase A: maximum yaw-rate command magnitude (rad/s).")
parser.add_argument("--route_curvature_abs_max", type=float, default=0.90,
                    help="Closed-loop Phase A: maximum nominal path curvature (1/m).")
parser.add_argument("--route_accel_abs_max", type=float, default=0.45,
                    help="Closed-loop Phase A: nominal longitudinal acceleration limit (m/s^2).")
parser.add_argument("--route_initial_lateral_offset_m", type=float, default=0.06,
                    help="Closed-loop Phase A: uniform initial reference offset magnitude (m).")
parser.add_argument("--route_initial_yaw_offset_rad", type=float, default=0.12,
                    help="Closed-loop Phase A: uniform initial reference yaw offset magnitude (rad).")
parser.add_argument("--tracker_lookahead_s", type=float, default=0.45,
                    help="Closed-loop Phase A: teacher preview horizon in seconds.")
parser.add_argument("--tracker_longitudinal_gain", type=float, default=0.55)
parser.add_argument("--tracker_lateral_gain", type=float, default=1.00)
parser.add_argument("--tracker_heading_gain", type=float, default=1.15)
parser.add_argument("--tracker_lateral_yaw_gain", type=float, default=0.65)
parser.add_argument("--desired_base_height", type=float, required=True,
                    help="Expert's commanded base height in metres; recorded for the velocity-height baseline.")
parser.add_argument("--min_demo_len", type=int, default=100,
                    help="Minimum recorded step length to keep an episode.")
parser.add_argument("--warmup_steps", type=int, default=25,
                    help="Steps to skip at the start of every episode (settling transient, ~0.5s at 50Hz).")
parser.add_argument("--verbose_filters", action="store_true",
                    help="Print one line per discarded episode. By default only periodic summaries are printed.")
parser.add_argument("--physics_dr_mode", type=str, choices=["off", "light", "full"], default="light",
                    help=(
                        "Physics domain randomization profile. 'light' keeps DR useful for diffusion data "
                        "without the very aggressive training ranges that can kill most expert rollouts."
                    ))
parser.add_argument("--disable_physics_dr", action="store_true",
                    help="Diagnostic mode: disable physics domain randomization (reproduces the old cleaner collection).")
parser.add_argument("--min_steps_per_skill", type=int, default=150,
                    help="Chained: minimum steps to run a skill before a transition.")
parser.add_argument("--max_steps_per_skill", type=int, default=300,
                    help="Chained: maximum steps to run a skill before a transition.")
parser.add_argument("--survival_check_steps", type=int, default=100,
                    help="Chained: steps to survive after a transition for the episode to be kept.")
parser.add_argument("--transition_blend_steps", type=int, default=0,
                    help="Steps to blend actions during policy transitions. 0 = abrupt (physical, recommended).")
parser.add_argument("--transition_blend_type", type=str, choices=["linear", "cosine"], default="cosine",
                    help="Interpolation curve for blending actions.")
parser.add_argument("--fall_gravity_z", type=float, default=-0.6,
                    help="Active-fall guardrail: gravity Z threshold (base tilt > ~53deg).")
parser.add_argument("--fall_height", type=float, default=0.11,
                    help="Active-fall guardrail: base height threshold in meters.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Strip parsed args so Hydra does not choke on them.
sys.argv = [sys.argv[0]] + hydra_args
# Data collection is normally run headless.
args_cli.headless = True if args_cli.headless is None else args_cli.headless
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports requiring an active simulation.
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Command ranges per skill. Kept conservative vs. the RL training envelope so the
# expert stays inside its stable operating region (DiffuseLoco recommends sampling
# goals within the stability envelope of the source policy).
# --------------------------------------------------------------------------- #
SKILL_COMMAND_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "walk":   {"vx": (-1.5, 1.5), "vy": (-1, 1), "wz": (-1, 1)},
    # crouch policy's stable envelope (trained up to +-1.0 / +-0.5 / +-1.0).
    "crouch": {"vx": (-0.75, 0.75), "vy": (-0.5, 0.5), "wz": (-0.5, 0.5)},
    "jump":   {"vx": (-1.2, 1.2),  "vy": (-0.6, 0.6),  "wz": (-0.6, 0.6)},
    "sprint": {"vx": (0.3, 2.0), "vy": (-0.2, 0.2), "wz": (-0.2, 0.2)},
}
SHARED_HEIGHT_COMMAND_RANGE = {"vx": (-0.75, 0.75), "vy": (-0.5, 0.5), "wz": (-0.5, 0.5)}

# HDF5 convention string stored as an attribute for downstream consumers.
HDF5_CONVENTION = (
    "aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; "
    "last_action[t] = actions[t-1] (last_action[0] = 0)."
)


# --------------------------------------------------------------------------- #
# Environment configuration
# --------------------------------------------------------------------------- #
def configure_light_physics_dr(events_cfg: Any) -> None:
    """Use a deployment-oriented DR profile for expert data collection.

    The full training DR in ``Solo12EnvCfg.EventCfg`` is useful when training RL
    experts from scratch, but it is too aggressive for collecting demonstrations
    from fixed checkpoints: in particular ``joint_friction=(0.01, 1.0)`` can make
    most rollouts unrecoverable for the low-torque Solo12. Light DR keeps the
    context's intent (mass/friction/inertia/CoM variability) while staying inside
    the expert policy's demonstrated stability envelope.
    """
    if hasattr(events_cfg, "physics_material") and events_cfg.physics_material is not None:
        events_cfg.physics_material.params["static_friction_range"] = (0.85, 1.35)
        events_cfg.physics_material.params["dynamic_friction_range"] = (0.80, 1.30)
    if hasattr(events_cfg, "add_base_mass") and events_cfg.add_base_mass is not None:
        events_cfg.add_base_mass.params["mass_distribution_params"] = (0.95, 1.10)
    if hasattr(events_cfg, "joint_friction") and events_cfg.joint_friction is not None:
        events_cfg.joint_friction.params["friction_distribution_params"] = (0.01, 0.20)
    if hasattr(events_cfg, "inertia_scale") and events_cfg.inertia_scale is not None:
        events_cfg.inertia_scale.params["inertia_distribution_params"] = (0.90, 1.10)
    if hasattr(events_cfg, "base_com") and events_cfg.base_com is not None:
        events_cfg.base_com.params["com_range"] = {
            "x": (-0.008, 0.008),
            "y": (-0.006, 0.006),
            "z": (-0.010, 0.010),
        }


def configure_env_for_collection(env_cfg: Any, args: argparse.Namespace) -> None:
    """Override env config for clean, DR-on data collection.

    Keeps physics domain randomization (masses, frictions, inertias, CoM) via the
    ``events`` manager, but disables everything that would either corrupt the
    recorded observations or perturb the robot with forces the policy cannot
    anticipate (pushes, actuation delays, observation noise, reset-velocity noise).
    """
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    # 6s episodes = 300 steps at 50Hz to prevent cumulative tracking drift in Phase A.
    env_cfg.episode_length_s = 6.0
    env_cfg.command_resampling_time_s = 1.0e9
    env_cfg.standing_env_prob = 0.0

    # --- Domain Randomization: KEEP events (physics randomization) active. ---
    # EventCfg only randomizes physics (rigid-body material, base mass, joint
    # friction, inertia, CoM) in "startup" mode. There are no push events there,
    # so we must NOT null the whole events manager.
    # (Previously this did `env_cfg.events = None`, which removed all physics DR.)
    if args.disable_physics_dr:
        args.physics_dr_mode = "off"
    if args.physics_dr_mode == "off" and getattr(env_cfg, "events", None) is not None:
        env_cfg.events = None
    elif args.physics_dr_mode == "light" and getattr(env_cfg, "events", None) is not None:
        configure_light_physics_dr(env_cfg.events)

    # --- Disable observation corruption: the saved obs are read from raw robot
    # data; disabling corruption keeps the policy input consistent with the saved obs.
    if hasattr(env_cfg, "enable_observation_corruption"):
        env_cfg.enable_observation_corruption = False

    # --- Disable artificial actuation delay so saved actions map 1:1 to applied targets.
    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)

    # --- Disable external base pushes (the DP cannot anticipate them).
    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)
    if hasattr(env_cfg, "base_push_force_xy_range"):
        env_cfg.base_push_force_xy_range = (0.0, 0.0)
    if hasattr(env_cfg, "base_push_force_z_range"):
        env_cfg.base_push_force_z_range = (0.0, 0.0)
    if hasattr(env_cfg, "forces_applied_to_base_curriculum"):
        env_cfg.forces_applied_to_base_curriculum = []

    # --- Clean, stationary episode starts (no random reset velocities).
    if hasattr(env_cfg, "reset_base_lin_vel_range"):
        env_cfg.reset_base_lin_vel_range = (0.0, 0.0)
    if hasattr(env_cfg, "reset_base_ang_vel_range"):
        env_cfg.reset_base_ang_vel_range = (0.0, 0.0)
    if hasattr(env_cfg, "flexed_initial_joint_pos_noise_range"):
        env_cfg.flexed_initial_joint_pos_noise_range = (0.0, 0.0)

    if hasattr(env_cfg, "kp") and hasattr(env_cfg, "kd"):
        print(f"[INFO] Environment PD gains: Kp={env_cfg.kp}, Kd={env_cfg.kd}")


# --------------------------------------------------------------------------- #
# Command sampling
# --------------------------------------------------------------------------- #
def _sample_phase_a_command(rng: random.Random) -> tuple[float, float, float]:
    """Sample one stable walk command phase for reference-path tracking.

    The discrete mixture deliberately contains stops, starts, forward arcs,
    lateral motion and a small reverse component.  This is not a global
    planner: it supplies locally executable command/reference pairs for the
    first path-following experiment.
    """

    mode = rng.choices(
        ("stop", "straight", "arc", "lateral", "reverse"),
        weights=(0.20, 0.30, 0.30, 0.12, 0.08),
        k=1,
    )[0]
    if mode == "stop":
        return 0.0, 0.0, 0.0
    if mode == "straight":
        return rng.uniform(0.20, 0.65), rng.uniform(-0.10, 0.10), rng.uniform(-0.12, 0.12)
    if mode == "arc":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        return rng.uniform(0.20, 0.55), rng.uniform(-0.08, 0.08), direction * rng.uniform(0.25, 0.70)
    if mode == "lateral":
        direction = -1.0 if rng.random() < 0.5 else 1.0
        return rng.uniform(0.15, 0.40), direction * rng.uniform(0.15, 0.35), rng.uniform(-0.25, 0.25)
    return rng.uniform(-0.35, -0.12), rng.uniform(-0.12, 0.12), rng.uniform(-0.25, 0.25)


def resample_command(
    env_idx: int,
    skill: str,
    commands_tensor: torch.Tensor,
    device: torch.device,
    rng: random.Random,
) -> None:
    """Sample a reproducible (vx, vy, wz) command for one environment in-place."""

    del device  # The command tensor already owns the correct device.
    if args_cli.route_profile == "phase_a":
        vx, vy, wz = _sample_phase_a_command(rng)
        commands_tensor[env_idx, 0] = vx
        commands_tensor[env_idx, 1] = vy
        commands_tensor[env_idx, 2] = wz
        return
    ranges = (
        SHARED_HEIGHT_COMMAND_RANGE
        if args_cli.command_profile == "shared_height" and skill in {"walk", "crouch"}
        else SKILL_COMMAND_RANGES.get(skill)
    )
    if ranges is None:
        vx = vy = wz = 0.0
    else:
        vx = rng.uniform(*ranges["vx"])
        vy = rng.uniform(*ranges["vy"])
        wz = rng.uniform(*ranges["wz"])
    commands_tensor[env_idx, 0] = vx
    commands_tensor[env_idx, 1] = vy
    commands_tensor[env_idx, 2] = wz


def infer_skill_name(checkpoint_path: str) -> str:
    """Map a checkpoint filename to one of the known skill names."""
    name = Path(checkpoint_path).stem.lower()
    for skill in ("walk", "crouch", "jump", "crab", "sprint"):
        if skill in name:
            return skill
    return name


# --------------------------------------------------------------------------- #
# Raw proprioception query
# --------------------------------------------------------------------------- #
def query_raw_state(raw_env: Any, joint_ids: slice,
                    env_mask: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """Read raw proprioceptive + world state straight from the articulation.

    Returns a dict of numpy arrays with leading dim = number of envs (or masked).
    Using raw ``robot.data`` (not the obs manager) guarantees the saved obs are
    clean and free of any observation-corruption noise.
    """
    robot = raw_env._robot
    if env_mask is None:
        sel = slice(None)
    else:
        sel = torch.as_tensor(env_mask, dtype=torch.long, device=robot.device)
    return {
        "joint_pos":         robot.data.joint_pos[sel, joint_ids].cpu().numpy(),
        "joint_vel":         robot.data.joint_vel[sel, joint_ids].cpu().numpy(),
        "base_ang_vel":      robot.data.root_ang_vel_b[sel].cpu().numpy(),
        "projected_gravity": robot.data.projected_gravity_b[sel].cpu().numpy(),
        "root_pos_w":        robot.data.root_pos_w[sel].cpu().numpy(),
        "root_quat_w":       robot.data.root_quat_w[sel].cpu().numpy(),
        "command":           raw_env._commands[sel, :3].cpu().numpy(),
    }


def state_is_fall(state: Dict[str, np.ndarray], env_idx: int,
                  gravity_z_thresh: float, height_thresh: float) -> bool:
    """Active-fall guardrail: base tilt too large or base too close to the ground."""
    g_z = float(state["projected_gravity"][env_idx, 2])
    h = float(state["root_pos_w"][env_idx, 2])
    return (g_z > gravity_z_thresh) or (h < height_thresh)


def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """Extract world yaw from WXYZ quaternions without importing simulator helpers."""

    quat = np.asarray(quat_wxyz, dtype=np.float32)
    w, x, y, z = (quat[..., index] for index in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)).astype(np.float32)


def integrate_reference_pose(
    position_w: np.ndarray,
    yaw_w: float,
    command_b: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, float]:
    """Advance the command-generated planar reference one control step.

    ``command_b`` is expressed in the reference body frame, exactly like the
    RL expert command.  The reference is deliberately independent of the
    achieved robot state after its initial pose: it remains a desired route
    when tracking errors occur.
    """

    vx, vy, wz = (float(value) for value in command_b[:3])
    cos_yaw, sin_yaw = math.cos(float(yaw_w)), math.sin(float(yaw_w))
    next_position = np.asarray(position_w, dtype=np.float32).copy()
    next_position[0] += float(dt) * (cos_yaw * vx - sin_yaw * vy)
    next_position[1] += float(dt) * (sin_yaw * vx + cos_yaw * vy)
    return next_position, float(yaw_w + float(dt) * wz)


class CollectionStats:
    """Simple counters for collection progress and dataset quality."""

    def __init__(self) -> None:
        self.saved_demos = 0
        self.saved_steps = 0
        self.discard_guardrail = 0
        self.discard_sim = 0
        self.discard_short = 0

    @property
    def discarded(self) -> int:
        return self.discard_guardrail + self.discard_sim + self.discard_short

    @property
    def finalized(self) -> int:
        return self.saved_demos + self.discarded

    @property
    def survival_rate(self) -> float:
        if self.finalized == 0:
            return 0.0
        return self.saved_demos / self.finalized

    def summary(self) -> str:
        return (
            f"saved_demos={self.saved_demos}, saved_steps={self.saved_steps}, "
            f"discarded={self.discarded} "
            f"(guardrail={self.discard_guardrail}, sim={self.discard_sim}, short={self.discard_short}), "
            f"survival={100.0 * self.survival_rate:.3f}%"
        )


# --------------------------------------------------------------------------- #
# Policy management
# --------------------------------------------------------------------------- #
class PolicyManager:
    """Loads expert policies and produces per-env actions (with optional blending)."""

    def __init__(self, vec_env, agent_cfg, device, mode: str,
                 checkpoint: str | None, checkpoints: Sequence[str], skill_name: str | None = None):
        self.vec_env = vec_env
        self.device = device
        self.mode = mode

        # Each entry: (skill_name, inference_callable, nn_module_for_reset)
        self.policies: List[Tuple[str, Any, Any]] = []
        self.skill_names: List[str] = []
        self.skill_to_idx: Dict[str, int] = {}

        if mode == "single":
            if not checkpoint:
                raise ValueError("Single mode requires the --checkpoint argument.")
            name = skill_name or infer_skill_name(checkpoint)
            if name not in SKILL_COMMAND_RANGES:
                raise ValueError(
                    f"Unknown skill {name!r}. Pass --skill_name using one of {sorted(SKILL_COMMAND_RANGES)}."
                )
            self._load(name, os.path.abspath(checkpoint), agent_cfg, vec_env)
        else:
            if len(checkpoints) < 2:
                raise ValueError("Chained mode requires at least two paths in --checkpoints.")
            for cp_path in checkpoints:
                name = infer_skill_name(cp_path)
                # Guarantee unique skill keys (avoid overwriting duplicate skills).
                key = name
                suffix = 1
                while key in self.skill_to_idx:
                    suffix += 1
                    key = f"{name}_{suffix}"
                self._load(key, os.path.abspath(cp_path), agent_cfg, vec_env)

        self.skill_names = [name for name, _, _ in self.policies]
        self.skill_to_idx = {name: i for i, name in enumerate(self.skill_names)}

    def _load(self, name: str, path: str, agent_cfg, vec_env) -> None:
        runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        print(f"[INFO] Loading policy '{name}' from: {path}")
        # Inference only: skip optimizer (older checkpoints often have mismatched param groups).
        runner.load(path, load_optimizer=False)
        policy_fn = runner.get_inference_policy(device=self.device)
        try:
            policy_nn = runner.alg.policy          # rsl_rl >= 2.3
        except AttributeError:
            policy_nn = runner.alg.actor_critic    # rsl_rl <= 2.2
        self.policies.append((name, policy_fn, policy_nn))

    # -- action inference --------------------------------------------------- #
    def get_actions(self, policy_obs: torch.Tensor,
                    current_skill_idx: np.ndarray, prev_skill_idx: np.ndarray,
                    blend_steps_left: np.ndarray, blend_steps: int, blend_type: str
                    ) -> torch.Tensor:
        num_envs = current_skill_idx.shape[0]
        num_actions = self.vec_env.num_actions
        actions = torch.zeros((num_envs, num_actions), device=self.device)

        if self.mode == "single":
            # Single policy: the TensorDict obs is what play.py passes too.
            with torch.inference_mode():
                actions = self.policies[0][1](policy_obs)
            return actions

        # Chained: query the active policy of each env via masking.
        with torch.inference_mode():
            for s_idx, (_, policy_fn, _) in enumerate(self.policies):
                mask_np = current_skill_idx == s_idx
                if not mask_np.any():
                    continue
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                actions[mask] = policy_fn(policy_obs[mask])

        # Optional action blending across the previous -> current policy transition.
        if blend_steps > 0 and np.any(blend_steps_left > 0):
            actions = self._blend_actions(
                policy_obs, actions, current_skill_idx, prev_skill_idx,
                blend_steps_left, blend_steps, blend_type,
            )
        return actions

    def _blend_actions(self, policy_obs: torch.Tensor, actions_new: torch.Tensor,
                       current_skill_idx: np.ndarray, prev_skill_idx: np.ndarray,
                       blend_steps_left: np.ndarray, blend_steps: int, blend_type: str
                       ) -> torch.Tensor:
        in_blend_np = blend_steps_left > 0
        actions_old = torch.zeros_like(actions_new)
        with torch.inference_mode():
            for s_idx, (_, policy_fn, _) in enumerate(self.policies):
                mask_np = in_blend_np & (prev_skill_idx == s_idx)
                if not mask_np.any():
                    continue
                mask = torch.as_tensor(mask_np, dtype=torch.bool, device=self.device)
                actions_old[mask] = policy_fn(policy_obs[mask])

        steps_in = (blend_steps - blend_steps_left[in_blend_np]).astype(np.float32)
        alpha_lin = steps_in / float(blend_steps)
        if blend_type == "cosine":
            alpha_np = 0.5 * (1.0 - np.cos(np.pi * alpha_lin))
        else:
            alpha_np = alpha_lin
        alphas = torch.as_tensor(alpha_np, dtype=torch.float32, device=self.device).unsqueeze(-1)

        actions = actions_new.clone()
        blend_mask = torch.as_tensor(in_blend_np, dtype=torch.bool, device=self.device)
        actions[blend_mask] = (1.0 - alphas) * actions_old[blend_mask] + alphas * actions_new[blend_mask]
        return actions

    def reset_recurrent_states(self, dones: torch.Tensor) -> None:
        """Reset recurrent hidden states of done envs (no-op for MLP policies)."""
        for _, _, policy_nn in self.policies:
            reset_fn = getattr(policy_nn, "reset", None)
            if callable(reset_fn):
                reset_fn(dones)


# --------------------------------------------------------------------------- #
# HDF5 writer
# --------------------------------------------------------------------------- #
class HDF5Writer:
    """Writes kept episodes to an HDF5 file in the diffusion_policy schema."""

    def __init__(
        self,
        path: Path,
        skill_names: List[str],
        seed: int,
        desired_base_height: float,
        *,
        route_profile: str,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = h5py.File(path, "w")
        self._data = self._f.create_group("data")
        self._data.attrs["skill_names"] = np.array(skill_names, dtype="S32")
        self._data.attrs["convention"] = HDF5_CONVENTION
        self._data.attrs["control_rate_hz"] = 50.0
        self._data.attrs["collection_seed"] = int(seed)
        self._data.attrs["condition_schema"] = "velocity_xyyaw_plus_desired_base_height_v1"
        self._data.attrs["desired_base_height_m"] = float(desired_base_height)
        self._data.attrs["reference_schema"] = (
            "fixed_capability_route_with_closed_loop_teacher_v2"
            if route_profile == "phase_a_closed_loop"
            else "command_integrated_world_route_v1"
        )
        self._data.attrs["route_profile"] = str(route_profile)
        if route_profile == "phase_a_closed_loop":
            self._data.attrs["route_teacher"] = "timed_preview_feedback_to_velocity_expert_v1"
            self._data.attrs["route_family_names"] = np.array(ROUTE_FAMILIES, dtype="S32")
        self._demo_counter = 0
        self._total_steps = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def write_demo(self, frames: List[Dict[str, np.ndarray]],
                   skills_sequence: List[str], target_steps: int) -> bool:
        """Save one episode. Returns True if saved, False if skipped (e.g. target reached)."""
        if not frames:
            return False
        if self._total_steps >= target_steps:
            return False

        keys = (
            "joint_pos", "joint_vel", "base_ang_vel", "projected_gravity",
            "last_action", "root_pos_w", "root_quat_w", "command", "desired_base_height",
            "reference_pos_w", "reference_yaw_w", "reference_command",
            "reference_progress", "tracking_error_frenet", "route_family_idx",
            "actions", "skill_idx",
        )
        stacked = {k: np.stack([fr[k] for fr in frames], axis=0) for k in keys}

        demo_name = f"demo_{self._demo_counter}"
        grp = self._data.create_group(demo_name)
        obs_grp = grp.create_group("obs")
        obs_grp.create_dataset("joint_pos", data=stacked["joint_pos"].astype(np.float32))
        obs_grp.create_dataset("joint_vel", data=stacked["joint_vel"].astype(np.float32))
        obs_grp.create_dataset("base_ang_vel", data=stacked["base_ang_vel"].astype(np.float32))
        obs_grp.create_dataset("projected_gravity", data=stacked["projected_gravity"].astype(np.float32))
        obs_grp.create_dataset("last_action", data=stacked["last_action"].astype(np.float32))
        obs_grp.create_dataset("root_pos_w", data=stacked["root_pos_w"].astype(np.float32))
        obs_grp.create_dataset("root_quat_w", data=stacked["root_quat_w"].astype(np.float32))
        obs_grp.create_dataset("command_speed", data=stacked["command"].astype(np.float32))
        obs_grp.create_dataset("desired_base_height", data=stacked["desired_base_height"].astype(np.float32))
        obs_grp.create_dataset("reference_pos_w", data=stacked["reference_pos_w"].astype(np.float32))
        obs_grp.create_dataset("reference_yaw_w", data=stacked["reference_yaw_w"].astype(np.float32))
        obs_grp.create_dataset("reference_command", data=stacked["reference_command"].astype(np.float32))
        obs_grp.create_dataset("reference_progress", data=stacked["reference_progress"].astype(np.int32))
        obs_grp.create_dataset("tracking_error_frenet", data=stacked["tracking_error_frenet"].astype(np.float32))
        obs_grp.create_dataset("route_family_idx", data=stacked["route_family_idx"].astype(np.int8))

        grp.create_dataset("actions", data=stacked["actions"].astype(np.float32))
        grp.create_dataset("dones", data=self._make_dones(len(frames)))
        grp.create_dataset("skill_idx", data=stacked["skill_idx"].astype(np.int8))

        grp.attrs["num_samples"] = len(frames)
        grp.attrs["skills_sequence"] = np.array(skills_sequence, dtype="S32")
        if "route_family" in frames[0]:
            grp.attrs["route_family"] = str(frames[0]["route_family"])

        self._demo_counter += 1
        self._total_steps += len(frames)
        print(f"[SAVE] {demo_name} (len={len(frames)}, skills={'->'.join(skills_sequence)}). "
              f"Total saved: {self._total_steps}/{target_steps}")
        return True

    @staticmethod
    def _make_dones(n: int) -> np.ndarray:
        dones = np.zeros(n, dtype=bool)
        dones[-1] = True
        return dones

    def close(self) -> None:
        self._f.close()


# --------------------------------------------------------------------------- #
# Per-env tracking state
# --------------------------------------------------------------------------- #
class CollectionState:
    """Parallel-array tracker for all environments."""

    def __init__(self, num_envs: int, policy_names: List[str], args: argparse.Namespace, rng: random.Random):
        self.num_envs = num_envs
        self.policy_names = policy_names
        self.skill_to_idx = {n: i for i, n in enumerate(policy_names)}

        # Skill assignment.
        self.current_skill_idx = np.array(
            [self.skill_to_idx[rng.choice(policy_names)] for _ in range(num_envs)], dtype=np.int16,
        )
        self.prev_skill_idx = self.current_skill_idx.copy()

        # Timing counters.
        self.steps_since_resample = np.zeros(num_envs, dtype=np.int32)
        self.steps_since_start = np.zeros(num_envs, dtype=np.int32)        # for warmup
        self.steps_left_in_skill = np.random.randint(
            args.min_steps_per_skill, args.max_steps_per_skill, size=num_envs,
        ).astype(np.int32)

        # Blending.
        self.blend_steps_left = np.zeros(num_envs, dtype=np.int32)

        # Transition bookkeeping (survival analysis).
        self.last_transition_step = np.zeros(num_envs, dtype=np.int32)
        self.has_transitioned = np.zeros(num_envs, dtype=bool)

        # Phase-A reference route.  It is initialized from the physical pose at
        # every reset and subsequently integrated from commands, not achieved
        # base motion. ``reference_ready`` handles simulator resets that happen
        # after this tracker has been reset.
        self.reference_pos_w = np.zeros((num_envs, 3), dtype=np.float32)
        self.reference_yaw_w = np.zeros(num_envs, dtype=np.float32)
        self.reference_ready = np.zeros(num_envs, dtype=bool)
        self.route_started = np.zeros(num_envs, dtype=bool)
        self.closed_loop_routes: List[ReferenceRoute | None] = [None] * num_envs
        self.latest_tracking_error = np.zeros((num_envs, 3), dtype=np.float32)
        self.latest_teacher_command = np.zeros((num_envs, 3), dtype=np.float32)

        # Per-env skill sequence (order of first appearance) for metadata.
        self.skill_sequences: List[List[str]] = [
            [policy_names[self.current_skill_idx[i]]] for i in range(num_envs)
        ]

    def reset_env(self, env_idx: int, rng: random.Random, args: argparse.Namespace) -> None:
        new_skill = rng.choice(self.policy_names)
        self.current_skill_idx[env_idx] = self.skill_to_idx[new_skill]
        self.prev_skill_idx[env_idx] = self.current_skill_idx[env_idx]
        self.steps_since_resample[env_idx] = 0
        self.steps_since_start[env_idx] = 0
        self.steps_left_in_skill[env_idx] = rng.randint(args.min_steps_per_skill, args.max_steps_per_skill)
        self.blend_steps_left[env_idx] = 0
        self.last_transition_step[env_idx] = 0
        self.has_transitioned[env_idx] = False
        self.skill_sequences[env_idx] = [new_skill]
        self.reference_ready[env_idx] = False
        self.route_started[env_idx] = False
        self.closed_loop_routes[env_idx] = None
        self.latest_tracking_error[env_idx] = 0.0
        self.latest_teacher_command[env_idx] = 0.0

    def initialize_references(self, state: Dict[str, np.ndarray], desired_base_height: float) -> None:
        """Anchor pending references at the reset pose before recording a frame."""

        pending = ~self.reference_ready
        if not np.any(pending):
            return
        self.reference_pos_w[pending] = state["root_pos_w"][pending]
        self.reference_pos_w[pending, 2] = float(desired_base_height)
        self.reference_yaw_w[pending] = yaw_from_quat_wxyz(state["root_quat_w"][pending])
        self.reference_ready[pending] = True

    def initialize_closed_loop_routes(
        self,
        state: Dict[str, np.ndarray],
        desired_base_height: float,
        args: argparse.Namespace,
        rng: random.Random,
        dt: float,
        route_steps: int,
    ) -> None:
        """Plan pending fixed routes from the reset pose, independent of later motion."""

        limits = CapabilityLimits(
            args.route_vx_min,
            args.route_vx_max,
            args.route_vy_abs_max,
            args.route_wz_abs_max,
            args.route_curvature_abs_max,
            args.route_accel_abs_max,
        )
        for env_idx, route in enumerate(self.closed_loop_routes):
            if route is not None:
                continue
            self.closed_loop_routes[env_idx] = generate_capability_route(
                start_pos_w=state["root_pos_w"][env_idx],
                start_yaw_w=float(yaw_from_quat_wxyz(state["root_quat_w"][env_idx])),
                desired_height=desired_base_height,
                steps=route_steps,
                dt=dt,
                limits=limits,
                rng=rng,
                startup_hold_steps=args.startup_hold_steps,
                initial_lateral_offset_m=args.route_initial_lateral_offset_m,
                initial_yaw_offset_rad=args.route_initial_yaw_offset_rad,
            )

    def closed_loop_reference(self, env_idx: int) -> tuple[np.ndarray, float, np.ndarray, str]:
        route = self.closed_loop_routes[env_idx]
        if route is None:
            raise RuntimeError("Closed-loop route requested before it was initialized.")
        index = int(np.clip(self.steps_since_start[env_idx], 0, len(route.pos_w) - 1))
        return route.pos_w[index], float(route.yaw_w[index]), route.nominal_command_b[index], route.family

    def update_closed_loop_command(self, env_idx: int, state: Dict[str, np.ndarray], args: argparse.Namespace) -> None:
        route = self.closed_loop_routes[env_idx]
        if route is None:
            raise RuntimeError("Closed-loop command requested before route initialization.")
        index = int(np.clip(self.steps_since_start[env_idx], 0, len(route.pos_w) - 1))
        # Keep an explicit settling/start token in the dataset.  Once movement
        # begins, the command comes from feedback around the immutable reference.
        if index < args.startup_hold_steps:
            command = np.zeros(3, dtype=np.float32)
            error = np.zeros(3, dtype=np.float32)
        else:
            limits = CapabilityLimits(
                args.route_vx_min, args.route_vx_max, args.route_vy_abs_max,
                args.route_wz_abs_max, args.route_curvature_abs_max, args.route_accel_abs_max,
            )
            gains = TrackerGains(
                args.tracker_lookahead_s, args.tracker_longitudinal_gain,
                args.tracker_lateral_gain, args.tracker_heading_gain, args.tracker_lateral_yaw_gain,
            )
            command, error = tracking_command(
                route, index, state["root_pos_w"][env_idx], state["root_quat_w"][env_idx], limits, gains
            )
        self.latest_teacher_command[env_idx] = command
        self.latest_tracking_error[env_idx] = error

    def advance_reference(self, env_idx: int, command_b: np.ndarray, dt: float) -> None:
        if not self.reference_ready[env_idx]:
            return
        position, yaw = integrate_reference_pose(
            self.reference_pos_w[env_idx], self.reference_yaw_w[env_idx], command_b, dt
        )
        self.reference_pos_w[env_idx] = position
        self.reference_yaw_w[env_idx] = yaw

    def trigger_transition(self, env_idx: int, rng: random.Random, args: argparse.Namespace) -> None:
        old_skill = self.policy_names[self.current_skill_idx[env_idx]]
        new_skill = rng.choice([p for p in self.policy_names if p != old_skill])
        self.prev_skill_idx[env_idx] = self.current_skill_idx[env_idx]
        self.current_skill_idx[env_idx] = self.skill_to_idx[new_skill]
        self.blend_steps_left[env_idx] = args.transition_blend_steps
        self.steps_left_in_skill[env_idx] = rng.randint(args.min_steps_per_skill, args.max_steps_per_skill)
        self.last_transition_step[env_idx] = self.steps_since_start[env_idx]
        self.has_transitioned[env_idx] = True
        if new_skill not in self.skill_sequences[env_idx]:
            self.skill_sequences[env_idx].append(new_skill)


# --------------------------------------------------------------------------- #
# Main collection loop
# --------------------------------------------------------------------------- #
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    output_path = _THIS_DIR / "datasets" / args_cli.output_name

    if args_cli.mode != "single":
        raise ValueError(
            "The velocity-height schema is intentionally single-expert only. "
            "Collect one file per expert and merge them; chained transitions need an explicit per-skill height schedule."
        )
    if args_cli.route_profile == "phase_a" and args_cli.skill_name not in (None, "walk"):
        raise ValueError("phase_a is intentionally walk-only; pass --skill_name walk.")
    if args_cli.route_profile in {"phase_a", "phase_a_closed_loop"} and not args_cli.include_warmup_frames:
        raise ValueError(
            "Phase-A route collection requires --include_warmup_frames so reset/start histories are represented."
        )
    if args_cli.startup_hold_steps < 0:
        raise ValueError("--startup_hold_steps must be non-negative.")

    configure_env_for_collection(env_cfg, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    raw_env = env.unwrapped
    device = torch.device(vec_env.unwrapped.device)
    joint_ids = raw_env._joint_ids

    rng = random.Random(args_cli.seed)
    np.random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    policy_mgr = PolicyManager(vec_env, agent_cfg, device, args_cli.mode,
                               args_cli.checkpoint, args_cli.checkpoints, args_cli.skill_name)
    skill_names = policy_mgr.skill_names
    if args_cli.route_profile == "phase_a" and skill_names != ["walk"]:
        raise ValueError(f"phase_a requires exactly the walk expert, got {skill_names}.")
    num_envs = args_cli.num_envs

    state_tracker = CollectionState(num_envs, skill_names, args_cli, rng)
    resample_interval = max(1, int(round(args_cli.command_resample_time_s / raw_env.step_dt)))

    # Initialize commands for every env. Phase-A routes start with an intentional
    # zero-command hold; later start/stop windows are part of the dataset.
    for i in range(num_envs):
        if args_cli.route_profile in {"phase_a", "phase_a_closed_loop"}:
            raw_env._commands[i, :3] = 0.0
        else:
            resample_command(i, skill_names[state_tracker.current_skill_idx[i]], raw_env._commands, device, rng)

    writer = HDF5Writer(
        output_path,
        skill_names,
        args_cli.seed,
        args_cli.desired_base_height,
        route_profile=args_cli.route_profile,
    )
    stats = CollectionStats()
    print(f"[INFO] Starting collection ({args_cli.mode} mode). Output: {output_path}")
    dr_status = "off" if args_cli.disable_physics_dr else args_cli.physics_dr_mode
    print(f"[INFO] Skills: {skill_names} | DR={dr_status} | blend_steps={args_cli.transition_blend_steps} "
          f"| warmup={args_cli.warmup_steps} | resample={args_cli.command_resample_time_s}s "
          f"| route_profile={args_cli.route_profile}")

    # Per-env frame buffers.
    buffers: List[List[Dict[str, np.ndarray]]] = [[] for _ in range(num_envs)]

    obs = vec_env.get_observations()  # TensorDict; rebuilt after any command update below
    cached_state = query_raw_state(raw_env, joint_ids)  # state at t0 (post-reset)
    route_steps = max(2, int(round(env_cfg.episode_length_s / raw_env.step_dt)))

    steps_collected = 0
    while writer.total_steps < args_cli.num_steps:
        if args_cli.route_profile == "phase_a_closed_loop":
            state_tracker.initialize_closed_loop_routes(
                cached_state, args_cli.desired_base_height, args_cli, rng, raw_env.step_dt, route_steps
            )
            for i in range(num_envs):
                state_tracker.update_closed_loop_command(i, cached_state, args_cli)
                raw_env._commands[i, :3] = torch.as_tensor(
                    state_tracker.latest_teacher_command[i], dtype=raw_env._commands.dtype, device=device
                )
            # ``cached_state`` was queried before the command write.  Update its
            # command field as well: this exact vector is both in the expert
            # observation and stored as command_speed alongside action_t.
            cached_state["command"] = raw_env._commands[:, :3].detach().cpu().numpy()
            # The RSL policy observation contains commands.  It must be rebuilt
            # *after* writing the route-tracker command, otherwise labels are one
            # simulator tick ahead of the observation consumed by the teacher.
            obs = vec_env.get_observations()
        else:
            state_tracker.initialize_references(cached_state, args_cli.desired_base_height)
        # --- 1. Compute actions a_t from the current (time-t) policy obs. --- #
        # Single mode passes the TensorDict (matches rsl_rl play.py); chained mode
        # passes the "policy" obs tensor so it can be masked per active skill.
        if args_cli.mode == "single":
            policy_obs_arg = obs
        else:
            policy_obs_arg = obs["policy"] if isinstance(obs, dict) else obs
        actions = policy_mgr.get_actions(
            policy_obs_arg, state_tracker.current_skill_idx,
            state_tracker.prev_skill_idx, state_tracker.blend_steps_left,
            args_cli.transition_blend_steps, args_cli.transition_blend_type,
        )
        actions_np = actions.detach().cpu().numpy()

        # --- 2. Record aligned (state_t, a_t) frames for envs past warmup. --- #
        # cached_state holds the raw state at time t (post-step of t-1, or post-reset).
        # The action a_t is the expert action taken from state_t -> aligned.
        for i in range(num_envs):
            if not args_cli.include_warmup_frames and state_tracker.steps_since_start[i] < args_cli.warmup_steps:
                continue
            if args_cli.route_profile == "phase_a_closed_loop":
                reference_pos, reference_yaw, reference_command, route_family = state_tracker.closed_loop_reference(i)
                buffers[i].append(_make_frame(
                    cached_state, i, actions_np[i], state_tracker.current_skill_idx[i], args_cli.desired_base_height,
                    reference_pos, reference_yaw, reference_command=reference_command,
                    reference_progress=state_tracker.steps_since_start[i],
                    tracking_error=state_tracker.latest_tracking_error[i], route_family=route_family,
                ))
            else:
                buffers[i].append(_make_frame(
                    cached_state, i, actions_np[i], state_tracker.current_skill_idx[i], args_cli.desired_base_height,
                    state_tracker.reference_pos_w[i], state_tracker.reference_yaw_w[i],
                ))

        # --- 3. Step the simulator (advances to time t+1). --- #
        obs, _, dones, _ = vec_env.step(actions)
        policy_mgr.reset_recurrent_states(dones)
        state_next = query_raw_state(raw_env, joint_ids)  # state at t+1 (post-reset for done envs)
        dones_np = dones.cpu().numpy()
        reset_terminated_np = raw_env.reset_terminated.cpu().numpy()

        # --- 4. Per-env post-step bookkeeping + episode finalization. --- #
        for i in range(num_envs):
            tr = state_tracker
            command_used = cached_state["command"][i].copy()
            if args_cli.route_profile != "phase_a_closed_loop":
                tr.advance_reference(i, command_used, raw_env.step_dt)
            tr.steps_since_start[i] += 1
            tr.steps_since_resample[i] += 1

            # Phase A deliberately starts from a held stop. Afterwards all
            # resampling is recorded both as an expert command and as an
            # independently integrated world-frame reference trajectory.
            if args_cli.route_profile == "phase_a" and not tr.route_started[i]:
                if tr.steps_since_start[i] >= args_cli.startup_hold_steps:
                    resample_command(i, skill_names[tr.current_skill_idx[i]], raw_env._commands, device, rng)
                    tr.steps_since_resample[i] = 0
                    tr.route_started[i] = True
            elif args_cli.route_profile != "phase_a_closed_loop" and tr.steps_since_resample[i] >= resample_interval:
                resample_command(i, skill_names[tr.current_skill_idx[i]], raw_env._commands, device, rng)
                tr.steps_since_resample[i] = 0

            # Chained: schedule skill transitions.
            if args_cli.mode == "chained":
                tr.steps_left_in_skill[i] -= 1
                if tr.steps_left_in_skill[i] <= 0:
                    tr.trigger_transition(i, rng, args_cli)
                    resample_command(i, skill_names[tr.current_skill_idx[i]], raw_env._commands, device, rng)
                    tr.steps_since_resample[i] = 0

            # Episode finalization: sim reset OR active-fall guardrail.
            # The active guardrail intentionally starts AFTER warmup. During the
            # first ~0.5s the robot is settling after reset; killing those short
            # transients prevents every env from ever reaching a clean timeout,
            # especially with physics DR enabled. Simulator-level terminations
            # are still honored during warmup.
            guardrail_active = tr.steps_since_start[i] >= args_cli.warmup_steps
            is_fall = guardrail_active and state_is_fall(
                state_next, i, args_cli.fall_gravity_z, args_cli.fall_height
            )
            if dones_np[i] or is_fall:
                # For active falls the simulator missed, reset the env BEFORE finalizing
                # so that _finalize_episode's skill-specific command resample is applied
                # AFTER the env's own _resample_commands (otherwise it gets overwritten).
                if is_fall and not dones_np[i]:
                    with torch.inference_mode():
                        raw_env._reset_idx(torch.as_tensor([i], device=device))
                _finalize_episode(
                    i, buffers, state_tracker, skill_names, args_cli,
                    is_fall=is_fall, sim_terminated=bool(reset_terminated_np[i]),
                    writer=writer, stats=stats, rng=rng, raw_env=raw_env, device=device,
                )
                if args_cli.route_profile in {"phase_a", "phase_a_closed_loop"}:
                    state_next["command"][i] = 0.0
                # The cached state for a manually-reset env is now stale (pre-reset
                # fallen state), but warmup prevents it from ever being recorded and
                # the next state_next queries refresh it.

        # Decrement blending counters at the very end of the step.
        if args_cli.mode == "chained" and args_cli.transition_blend_steps > 0:
            state_tracker.blend_steps_left = np.maximum(
                0, state_tracker.blend_steps_left - 1,
            ).astype(np.int32)

        cached_state = state_next
        steps_collected += 1
        if steps_collected % 500 == 0:
            print(f"[STATUS] sim_steps={steps_collected}, target_steps={args_cli.num_steps}, {stats.summary()}")

    writer.close()
    print(f"[SUCCESS] Collection complete. Demos: {writer.total_steps} steps saved to {output_path}")
    vec_env.close()


def _make_frame(
    state: Dict[str, np.ndarray],
    env_idx: int,
    action: np.ndarray,
    skill_idx: int,
    desired_base_height: float,
    reference_pos_w: np.ndarray,
    reference_yaw_w: float,
    *,
    reference_command: np.ndarray | None = None,
    reference_progress: int = 0,
    tracking_error: np.ndarray | None = None,
    route_family: str | None = None,
) -> Dict[str, np.ndarray]:
    """Build an aligned recorded frame from the raw state of one env."""
    return {
        "joint_pos":         state["joint_pos"][env_idx],
        "joint_vel":         state["joint_vel"][env_idx],
        "base_ang_vel":      state["base_ang_vel"][env_idx],
        "projected_gravity": state["projected_gravity"][env_idx],
        "last_action":       action,  # overwritten with the previous action in the buffer
        "root_pos_w":        state["root_pos_w"][env_idx],
        "root_quat_w":       state["root_quat_w"][env_idx],
        "command":           state["command"][env_idx],
        "desired_base_height": np.asarray([desired_base_height], dtype=np.float32),
        "reference_pos_w":  np.array(reference_pos_w, dtype=np.float32),
        "reference_yaw_w":  np.asarray([reference_yaw_w], dtype=np.float32),
        "reference_command": state["command"][env_idx] if reference_command is None else reference_command,
        "reference_progress": np.asarray([reference_progress], dtype=np.int32),
        "tracking_error_frenet": (
            np.zeros(3, dtype=np.float32) if tracking_error is None else np.asarray(tracking_error, dtype=np.float32)
        ),
        "route_family_idx": np.asarray([ROUTE_FAMILIES_TO_IDX.get(route_family, -1)], dtype=np.int8),
        "route_family": route_family or "legacy",
        "actions":           action,
        "skill_idx":         np.int8(skill_idx),
    }


def _finalize_episode(env_idx: int, buffers: List[List[Dict[str, np.ndarray]]],
                      tracker: CollectionState, skill_names: List[str], args: argparse.Namespace,
                      is_fall: bool, sim_terminated: bool, writer: HDF5Writer,
                      stats: CollectionStats, rng: random.Random, raw_env: Any, device: torch.device) -> None:
    """Apply survival filtering, save the episode if it survived, and reset tracking."""
    frames = buffers[env_idx]
    buffers[env_idx] = []

    survived = (not is_fall) and (not sim_terminated)  # only a clean timeout counts as survived
    if not survived:
        # Both our guardrail and the simulator's own termination count as a fall/instability.
        if is_fall:
            stats.discard_guardrail += 1
        else:
            stats.discard_sim += 1
        if args.mode == "chained" and tracker.has_transitioned[env_idx]:
            steps_after = tracker.steps_since_start[env_idx] - tracker.last_transition_step[env_idx]
            tag = "transition fall" if steps_after < args_cli.survival_check_steps else "post-transition fall"
            source = "guardrail" if is_fall else "sim-terminated"
            if args.verbose_filters:
                print(f"[FILTER] env {env_idx} discarded ({tag}, {source}, +{steps_after} steps).")
        else:
            source = "guardrail fall" if is_fall else "sim-terminated fall"
            if args.verbose_filters:
                print(f"[FILTER] env {env_idx} discarded ({source} during locomotion).")
        _reset_after_finalize(env_idx, tracker, rng, args, raw_env, device)
        return

    if len(frames) < args.min_demo_len:
        stats.discard_short += 1
        if args.verbose_filters:
            print(f"[FILTER] env {env_idx} discarded (len {len(frames)} < min_demo_len {args.min_demo_len}).")
        _reset_after_finalize(env_idx, tracker, rng, args, raw_env, device)
        return

    # Fix last_action: aligned convention -> last_action[t] = actions[t-1], last_action[0] = 0.
    actions_arr = np.stack([fr["actions"] for fr in frames], axis=0)
    last_act = np.zeros_like(actions_arr)
    if len(frames) > 1:
        last_act[1:] = actions_arr[:-1]
    for t, fr in enumerate(frames):
        fr["last_action"] = last_act[t]

    skills_seq = tracker.skill_sequences[env_idx]
    if writer.write_demo(frames, skills_seq, args.num_steps):
        stats.saved_demos += 1
        stats.saved_steps += len(frames)
    _reset_after_finalize(env_idx, tracker, rng, args, raw_env, device)


def _reset_after_finalize(env_idx: int, tracker: CollectionState, rng: random.Random,
                          args: argparse.Namespace, raw_env: Any, device: torch.device) -> None:
    tracker.reset_env(env_idx, rng, args)
    if args.route_profile in {"phase_a", "phase_a_closed_loop"}:
        raw_env._commands[env_idx, :3] = 0.0
    else:
        resample_command(
            env_idx,
            tracker.policy_names[tracker.current_skill_idx[env_idx]],
            raw_env._commands,
            device,
            rng,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
