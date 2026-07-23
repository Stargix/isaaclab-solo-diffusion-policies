"""Deterministic, paired evaluation for a Phase-B1 residual PPO checkpoint.

The evaluator intentionally measures terminal route outcomes rather than
training return.  A new environment is constructed for each checkpoint with
the same seed and route stage, then the actor mean is executed without PPO
exploration noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate one residual-diffusion PPO checkpoint.")
parser.add_argument("--task", default="solo12-residual-diffusion-rl-v0")
parser.add_argument("--checkpoint", help="RSL-RL residual PPO checkpoint.")
parser.add_argument("--diffusion_checkpoint", required=True, help="Frozen Phase-A diffusion checkpoint.")
parser.add_argument("--output_dir", required=True, help="Directory in which to write summary.json.")
parser.add_argument("--episodes", type=int, default=256, help="Minimum number of completed episodes.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Shared route RNG seed for paired comparison.")
parser.add_argument(
    "--stage", type=int, choices=(0, 1, 2), default=2, help="Fixed RouteBank benchmark stage."
)
parser.add_argument("--max_steps", type=int, default=20_000, help="Safety bound on vectorized control steps.")
parser.add_argument("--trace_envs", type=int, default=6, help="Number of initial episodes to save as qualitative traces.")
parser.add_argument(
    "--prior_only",
    action="store_true",
    help="Execute an exactly zero residual; --checkpoint is then unnecessary.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.prior_only and not args_cli.checkpoint:
    parser.error("--checkpoint is required unless --prior_only is selected")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.terrains import TerrainGeneratorCfg  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _capture_traces(raw_env, traces: list[dict], active: torch.Tensor, time_s: float) -> None:
    """Capture pre-action route state for a small, fixed set of environments."""

    route = raw_env._routes
    robot = raw_env._robot.data
    rows = torch.arange(raw_env.num_envs, device=raw_env.device)
    indices = route.progress_idx
    local_xy = raw_env._local_position()
    height = robot.root_pos_w[:, 2] - raw_env._terrain.env_origins[:, 2]
    target_height = route.height[rows, indices]
    tangent_yaw = route.yaw[rows, indices]
    tangent_speed = (
        robot.root_lin_vel_w[:, 0] * torch.cos(tangent_yaw)
        + robot.root_lin_vel_w[:, 1] * torch.sin(tangent_yaw)
    )

    for env_id, trace in enumerate(traces):
        if not bool(active[env_id]):
            continue
        trace["time_s"].append(time_s)
        trace["actual_xy"].append(local_xy[env_id].detach().cpu().tolist())
        trace["target_height"].append(float(target_height[env_id]))
        trace["actual_height"].append(float(height[env_id]))
        trace["requested_speed"].append(float(route.speed[env_id]))
        trace["tangent_speed"].append(float(tangent_speed[env_id]))
        trace["cross_track"].append(float(route.cross_track[env_id]))
        trace["residual_rms"].append(float(torch.sqrt(raw_env._residual[env_id].square().mean())))


def _write_diagnostics(output_dir: Path, traces: list[dict], summary: dict) -> None:
    """Write compact plots analogous to the Phase-A diffusion evaluator.

    Trace environments are deliberately capped: they are qualitative examples,
    while the JSON outcome rates remain the quantitative evaluation over every
    completed vectorized episode.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ("#0052CC", "#FF5A5F", "#00A86B", "#FFB300", "#7B1FA2", "#6A1B9A")

    rows: list[dict] = []
    for trace_id, trace in enumerate(traces):
        for index, time_s in enumerate(trace["time_s"]):
            rows.append({
                "trace_env": trace_id,
                "route_kind": trace["route_kind"],
                "time_s": time_s,
                "x_m": trace["actual_xy"][index][0],
                "y_m": trace["actual_xy"][index][1],
                "target_height_m": trace["target_height"][index],
                "actual_height_m": trace["actual_height"][index],
                "requested_speed_mps": trace["requested_speed"][index],
                "tangent_speed_mps": trace["tangent_speed"][index],
                "cross_track_m": trace["cross_track"][index],
                "residual_rms": trace["residual_rms"][index],
            })
    if rows:
        with (output_dir / "trace_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    valid_traces = [trace for trace in traces if trace["actual_xy"]]
    if valid_traces:
        columns = min(3, len(valid_traces))
        nrows = int(np.ceil(len(valid_traces) / columns))
        figure, axes = plt.subplots(nrows, columns, figsize=(5.0 * columns, 4.2 * nrows), squeeze=False)
        for index, trace in enumerate(valid_traces):
            axis = axes.flat[index]
            ref = np.asarray(trace["reference_xy"])
            actual = np.asarray(trace["actual_xy"])
            axis.plot(ref[:, 0], ref[:, 1], "--", color="#333333", linewidth=1.8, label="Route reference")
            axis.plot(actual[:, 0], actual[:, 1], color=colors[index % len(colors)], linewidth=2.0, label="Residual policy")
            axis.scatter(ref[0, 0], ref[0, 1], color="#333333", s=20, zorder=3)
            axis.set_title(f"Trace {index}: {trace['route_kind']}")
            axis.set_xlabel("Local x [m]")
            axis.set_ylabel("Local y [m]")
            axis.axis("equal")
            axis.legend(fontsize=8)
        for axis in axes.flat[len(valid_traces):]:
            axis.set_visible(False)
        figure.tight_layout()
        figure.savefig(output_dir / "trajectories.png", dpi=200)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
        for index, trace in enumerate(valid_traces):
            time_s = np.asarray(trace["time_s"])
            color = colors[index % len(colors)]
            axes[0].plot(time_s, trace["target_height"], "--", color=color, alpha=0.85)
            axes[0].plot(time_s, trace["actual_height"], color=color, label=f"trace {index}")
            axes[1].plot(time_s, np.abs(np.asarray(trace["actual_height"]) - np.asarray(trace["target_height"])), color=color, label=f"trace {index}")
        axes[0].set_title("Height tracking (dashed = requested)")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Base height [m]")
        axes[1].set_title("Absolute height error")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Error [m]")
        for axis in axes:
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "height_tracking.png", dpi=200)
        plt.close(figure)

        figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
        for index, trace in enumerate(valid_traces):
            time_s = np.asarray(trace["time_s"])
            color = colors[index % len(colors)]
            axes[0].plot(time_s, trace["requested_speed"], "--", color=color, alpha=0.85)
            axes[0].plot(time_s, trace["tangent_speed"], color=color, label=f"trace {index}")
            axes[1].plot(time_s, np.abs(trace["cross_track"]), color=color, label=f"trace {index}")
        axes[0].set_title("Tangential speed (dashed = requested)")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Speed [m/s]")
        axes[1].set_title("Cross-track error")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Absolute error [m]")
        for axis in axes:
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / "speed_and_tracking.png", dpi=200)
        plt.close(figure)

    outcome_order = (
        "route_success",
        "base_contact",
        "corridor_failure",
        "terminal_overshoot",
        "time_out",
    )
    values = [100.0 * summary["rates"].get(name, 0.0) for name in outcome_order]
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    bars = axis.bar(
        outcome_order,
        values,
        color=("#00A86B", "#FF5A5F", "#FFB300", "#0052CC", "#7B1FA2"),
    )
    axis.set_ylim(0.0, max(100.0, max(values, default=0.0) * 1.15))
    axis.set_ylabel("Completed episodes [%]")
    axis.set_title("Terminal outcomes")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(output_dir / "terminal_outcomes.png", dpi=200)
    plt.close(figure)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0 or args_cli.max_steps <= 0 or args_cli.trace_envs < 0:
        raise ValueError("--episodes, --num_envs and --max_steps must be positive; --trace_envs cannot be negative")

    # A fresh process invocation per checkpoint plus this seed makes the two
    # commands paired: RouteBank consumes the same random stream in each run.
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device
    env_cfg.spatial_diffusion_checkpoint = str(Path(args_cli.diffusion_checkpoint).resolve())

    # The benchmark is defined on a flat plane.  IsaacLab's default plane
    # references the remote Grid USD, which makes evaluation depend on network
    # access and can fail before the policy is even loaded.  Generate the
    # physically equivalent plane locally for a reproducible evaluator; this
    # does not alter the training environment or the learned checkpoint.
    env_cfg.terrain.terrain_type = "generator"
    env_cfg.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=args_cli.seed,
        curriculum=False,
        size=(20.0, 20.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
    )

    # RouteBank selects its stage from this common control-step counter.  Keep
    # the MDP fixed for all evaluated episodes rather than allowing an
    # evaluation run to change difficulty halfway through.
    env_cfg.use_route_curriculum = False
    env_cfg.route_stage = args_cli.stage
    env_cfg.stratified_route_sampling = True

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    observations = env.get_observations()
    raw_env = env.unwrapped
    checkpoint_path = None if args_cli.prior_only else Path(args_cli.checkpoint).resolve()
    if args_cli.prior_only:
        policy_nn = None

        def policy(_observations):
            return torch.zeros(
                raw_env.num_envs,
                raw_env.cfg.action_space,
                device=raw_env.device,
            )

    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint_path), load_optimizer=False)
        policy = runner.get_inference_policy(device=raw_env.device)
        policy_nn = runner.alg.policy

    trace_count = min(args_cli.trace_envs, args_cli.num_envs)
    route_names = ("straight", "s_curve", "right_angle", "random_curve")
    traces: list[dict] = []
    for env_id in range(trace_count):
        route_kind = int(raw_env._routes.route_kind[env_id])
        traces.append({
            "route_kind": route_names[route_kind],
            "reference_xy": raw_env._routes.xy[env_id].detach().cpu().tolist(),
            "time_s": [],
            "actual_xy": [],
            "target_height": [],
            "actual_height": [],
            "requested_speed": [],
            "tangent_speed": [],
            "cross_track": [],
            "residual_rms": [],
        })
    trace_active = torch.ones(trace_count, dtype=torch.bool, device=raw_env.device)
    outcomes = {
        "route_success": 0,
        "base_contact": 0,
        "corridor_failure": 0,
        "time_out": 0,
        "terminal_overshoot": 0,
    }
    outcome_names = tuple(outcomes)
    per_route_outcomes = {
        name: {outcome: 0 for outcome in outcome_names} for name in route_names
    }
    completed = 0
    reward_sum = 0.0
    length_sum_s = 0.0
    steps = 0
    state_sums = {"cross_track_abs_m": 0.0, "height_error_abs_m": 0.0, "speed_error_abs_mps": 0.0, "residual_rms": 0.0}
    state_samples = 0
    terminal_mean_speed_error_sum = 0.0
    terminal_distance_sum = 0.0
    terminal_along_error_sum = 0.0
    terminal_progress_fraction_sum = 0.0
    terminal_height_error_sum = 0.0

    with torch.inference_mode():
        while completed < args_cli.episodes and steps < args_cli.max_steps:
            _capture_traces(raw_env, traces, trace_active, steps * raw_env.step_dt)
            route = raw_env._routes
            robot = raw_env._robot.data
            route_rows = torch.arange(raw_env.num_envs, device=raw_env.device)
            target_height = route.height[route_rows, route.progress_idx]
            base_height = robot.root_pos_w[:, 2] - raw_env._terrain.env_origins[:, 2]
            state_sums["cross_track_abs_m"] += float(route.cross_track.abs().sum())
            state_sums["height_error_abs_m"] += float((base_height - target_height).abs().sum())
            tangent_yaw = route.yaw[route_rows, route.progress_idx]
            tangent_speed = (
                robot.root_lin_vel_w[:, 0] * torch.cos(tangent_yaw)
                + robot.root_lin_vel_w[:, 1] * torch.sin(tangent_yaw)
            )
            state_sums["speed_error_abs_mps"] += float((tangent_speed - route.speed).abs().sum())
            state_sums["residual_rms"] += float(torch.sqrt(raw_env._residual.square().mean(dim=1)).sum())
            state_samples += raw_env.num_envs
            actions = policy(observations)
            observations, _, dones, extras = env.step(actions)
            if policy_nn is not None:
                policy_nn.reset(dones)
            steps += 1

            if trace_count:
                trace_active &= ~dones[:trace_count].bool()

            done_count = int(dones.sum().item())
            if done_count == 0:
                continue
            completed += done_count
            log = extras.get("log", {})
            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            episode_kinds = raw_env._last_episode_route_kind[done_ids].detach().cpu().tolist()
            episode_outcomes = raw_env._last_episode_outcome[done_ids].detach().cpu().tolist()
            for route_kind, outcome_code in zip(episode_kinds, episode_outcomes):
                outcome_name = outcome_names[outcome_code]
                route_name = route_names[route_kind]
                outcomes[outcome_name] += 1
                per_route_outcomes[route_name][outcome_name] += 1
            terminal_mean_speed_error_sum += float(
                raw_env._last_episode_mean_speed_error[done_ids].abs().sum()
            )
            terminal_distance_sum += float(raw_env._last_episode_terminal_distance[done_ids].sum())
            terminal_along_error_sum += float(
                raw_env._last_episode_terminal_along_error[done_ids].sum()
            )
            terminal_progress_fraction_sum += float(
                raw_env._last_episode_progress_fraction[done_ids].sum()
            )
            terminal_height_error_sum += float(
                raw_env._last_episode_height_error[done_ids].abs().sum()
            )
            reward_sum += float(log.get("Episode_Reward/total", 0.0)) * done_count
            length_sum_s += float(log.get("Episode/length_seconds", 0.0)) * done_count

    if completed == 0:
        env.close()
        raise RuntimeError("No evaluation episode completed; increase --max_steps")

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "prior_only": args_cli.prior_only,
        "diffusion_checkpoint": str(Path(args_cli.diffusion_checkpoint).resolve()),
        "seed": args_cli.seed,
        "stage": args_cli.stage,
        "episodes_requested": args_cli.episodes,
        "episodes_completed": completed,
        "vectorized_control_steps": steps,
        "outcomes": outcomes,
        "rates": {name: count / completed for name, count in outcomes.items()},
        "mean_episode_reward": reward_sum / completed,
        "mean_episode_length_s": length_sum_s / completed,
        "mean_terminal_speed_error_abs_mps": terminal_mean_speed_error_sum / completed,
        "mean_terminal_distance_m": terminal_distance_sum / completed,
        "mean_terminal_along_error_m": terminal_along_error_sum / completed,
        "mean_terminal_progress_fraction": terminal_progress_fraction_sum / completed,
        "mean_terminal_height_error_abs_m": terminal_height_error_sum / completed,
        "mean_state_metrics": {name: value / state_samples for name, value in state_sums.items()},
        "per_route": {},
        "qualitative_trace_envs": trace_count,
    }
    for route_name, counts in per_route_outcomes.items():
        route_total = sum(counts.values())
        summary["per_route"][route_name] = {
            "episodes": route_total,
            "outcomes": counts,
            "rates": {
                name: count / route_total if route_total else None for name, count in counts.items()
            },
        }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_diagnostics(output_dir, traces, summary)
    env.close()
    print(json.dumps(summary, indent=2))
    print(f"[PASS] Wrote paired B1 evaluation and diagnostic plots to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
