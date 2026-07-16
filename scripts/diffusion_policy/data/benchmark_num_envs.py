# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark how many parallel Isaac Lab environments fit and run well on this GPU.

The script tries several ``num_envs`` values, creates the requested task, runs a
short warmup, then measures simulation throughput and CUDA memory. If a checkpoint
is provided, the benchmark includes policy inference; otherwise it uses zero
actions and measures mostly environment/simulator capacity.

Examples (PowerShell, from repo root)::

    # Quick capacity benchmark without policy inference
    python scripts/diffusion_policy/data/benchmark_num_envs.py --task="solo12-v0" --headless

    # Benchmark the real data-collection path with the walk policy
    python scripts/diffusion_policy/data/benchmark_num_envs.py --task="solo12-v0" \
        --checkpoint checkpoints/walk_safe.pt --envs 128 256 512 768 1024 --headless

The output CSV is written under ``scripts/diffusion_policy/data/benchmarks/``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_UPSTREAM_RSL_SCRIPT_DIR = _PROJECT_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
if str(_UPSTREAM_RSL_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Benchmark Isaac Lab parallel environment capacity.")
parser.add_argument("--task", type=str, required=True, help="Task name, e.g. solo12-v0.")
parser.add_argument("--checkpoint", type=str, default=None, help="Optional RSL-RL checkpoint to include policy inference.")
parser.add_argument("--envs", type=int, nargs="*", default=None, help="Specific num_envs values to test.")
parser.add_argument("--max_envs", type=int, default=4096, help="Max env count for the default powers-of-two sweep.")
parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup control steps before timing.")
parser.add_argument("--benchmark_steps", type=int, default=200, help="Timed control steps per env count.")
parser.add_argument("--oom_stop", action="store_true", default=True, help="Stop sweep at first out-of-memory failure.")
parser.add_argument("--continue_after_oom", action="store_true", help="Keep testing larger env counts after OOM.")
parser.add_argument("--physics_dr_mode", choices=["off", "light", "full"], default="off",
                    help="DR profile for benchmark. Default off for a stable capacity measurement.")
parser.add_argument("--output_csv", type=str, default=None, help="Optional CSV output path.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.continue_after_oom:
    args_cli.oom_stop = False
args_cli.headless = True if args_cli.headless is None else args_cli.headless
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def default_env_sweep(max_envs: int) -> list[int]:
    values: list[int] = []
    n = 64
    while n <= max_envs:
        values.append(n)
        n *= 2
    return values


def configure_env(env_cfg: Any, num_envs: int, args: argparse.Namespace) -> None:
    """Make the benchmark stable and comparable across env counts."""
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    # Capacity benchmark should not be dominated by external pushes/noisy resets.
    if hasattr(env_cfg, "enable_observation_corruption"):
        env_cfg.enable_observation_corruption = False
    if hasattr(env_cfg, "actuation_delay_range"):
        env_cfg.actuation_delay_range = (0, 0)
    if hasattr(env_cfg, "base_push_interval_range_s"):
        env_cfg.base_push_interval_range_s = (1.0e9, 1.0e9)
    if hasattr(env_cfg, "base_push_force_xy_range"):
        env_cfg.base_push_force_xy_range = (0.0, 0.0)
    if hasattr(env_cfg, "base_push_force_z_range"):
        env_cfg.base_push_force_z_range = (0.0, 0.0)
    if hasattr(env_cfg, "forces_applied_to_base_curriculum"):
        env_cfg.forces_applied_to_base_curriculum = []

    if args.physics_dr_mode == "off" and getattr(env_cfg, "events", None) is not None:
        env_cfg.events = None
    elif args.physics_dr_mode == "light" and getattr(env_cfg, "events", None) is not None:
        configure_light_physics_dr(env_cfg.events)


def configure_light_physics_dr(events_cfg: Any) -> None:
    """Same light DR profile used by collect_data.py."""
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


def cuda_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {"mem_alloc_gb": 0.0, "mem_reserved_gb": 0.0, "mem_peak_gb": 0.0}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "mem_alloc_gb": torch.cuda.memory_allocated(idx) / (1024 ** 3),
        "mem_reserved_gb": torch.cuda.memory_reserved(idx) / (1024 ** 3),
        "mem_peak_gb": torch.cuda.max_memory_allocated(idx) / (1024 ** 3),
    }


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def load_policy(vec_env: RslRlVecEnvWrapper, agent_cfg: Any, checkpoint: str | None, device: torch.device):
    if checkpoint is None:
        return None
    resume_path = os.path.abspath(checkpoint)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    return runner.get_inference_policy(device=device)


def run_one(env_cfg_template: Any, agent_cfg: Any, num_envs: int) -> dict[str, Any]:
    env = None
    vec_env = None
    try:
        env_cfg = copy.deepcopy(env_cfg_template)
        configure_env(env_cfg, num_envs, args_cli)

        env = gym.make(args_cli.task, cfg=env_cfg)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        device = torch.device(vec_env.unwrapped.device)
        policy = load_policy(vec_env, agent_cfg, args_cli.checkpoint, device)

        obs = vec_env.get_observations()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(args_cli.warmup_steps):
            with torch.inference_mode():
                actions = policy(obs) if policy is not None else torch.zeros(
                    (num_envs, vec_env.num_actions), device=device
                )
                obs, _, _, _ = vec_env.step(actions)

        sync_if_cuda(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        for _ in range(args_cli.benchmark_steps):
            with torch.inference_mode():
                actions = policy(obs) if policy is not None else torch.zeros(
                    (num_envs, vec_env.num_actions), device=device
                )
                obs, _, _, _ = vec_env.step(actions)
        sync_if_cuda(device)
        elapsed = time.perf_counter() - start

        stats = cuda_stats(device)
        return {
            "num_envs": num_envs,
            "status": "ok",
            "elapsed_s": elapsed,
            "ctrl_steps_per_s": args_cli.benchmark_steps / elapsed,
            "env_steps_per_s": (num_envs * args_cli.benchmark_steps) / elapsed,
            "ms_per_ctrl_step": 1000.0 * elapsed / args_cli.benchmark_steps,
            "checkpoint": bool(args_cli.checkpoint),
            **stats,
            "error": "",
        }
    except RuntimeError as exc:
        message = str(exc)
        status = "oom" if "out of memory" in message.lower() or "cuda" in message.lower() else "runtime_error"
        return {
            "num_envs": num_envs,
            "status": status,
            "elapsed_s": 0.0,
            "ctrl_steps_per_s": 0.0,
            "env_steps_per_s": 0.0,
            "ms_per_ctrl_step": 0.0,
            "checkpoint": bool(args_cli.checkpoint),
            "mem_alloc_gb": 0.0,
            "mem_reserved_gb": 0.0,
            "mem_peak_gb": 0.0,
            "error": message.splitlines()[0][:240],
        }
    finally:
        if vec_env is not None:
            vec_env.close()
        elif env is not None:
            env.close()
        del vec_env
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def print_result(row: dict[str, Any]) -> None:
    if row["status"] != "ok":
        print(f"[{row['status'].upper()}] envs={row['num_envs']} error={row['error']}")
        return
    print(
        f"[OK] envs={row['num_envs']:5d} | "
        f"env_steps/s={row['env_steps_per_s']:10.0f} | "
        f"ctrl_steps/s={row['ctrl_steps_per_s']:7.1f} | "
        f"ms/ctrl={row['ms_per_ctrl_step']:7.2f} | "
        f"peak_mem={row['mem_peak_gb']:.2f} GB | reserved={row['mem_reserved_gb']:.2f} GB"
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "num_envs", "status", "elapsed_s", "ctrl_steps_per_s", "env_steps_per_s",
        "ms_per_ctrl_step", "checkpoint", "mem_alloc_gb", "mem_reserved_gb",
        "mem_peak_gb", "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved to: {path}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: Any, agent_cfg: Any) -> None:
    env_values = args_cli.envs if args_cli.envs else default_env_sweep(args_cli.max_envs)
    print(
        f"[INFO] Benchmark task={args_cli.task}, envs={env_values}, "
        f"checkpoint={args_cli.checkpoint is not None}, DR={args_cli.physics_dr_mode}, "
        f"warmup={args_cli.warmup_steps}, steps={args_cli.benchmark_steps}"
    )

    rows: list[dict[str, Any]] = []
    for num_envs in env_values:
        row = run_one(env_cfg, agent_cfg, num_envs)
        rows.append(row)
        print_result(row)
        if row["status"] == "oom" and args_cli.oom_stop:
            print("[INFO] Stopping sweep at first OOM. Use --continue_after_oom to keep going.")
            break

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(args_cli.output_csv) if args_cli.output_csv else (
        _THIS_DIR / "benchmarks" / f"num_envs_benchmark_{timestamp}.csv"
    )
    write_csv(rows, out_path)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        fastest = max(ok_rows, key=lambda r: r["env_steps_per_s"])
        largest = max(ok_rows, key=lambda r: r["num_envs"])
        print(
            "[RECOMMENDATION] "
            f"Fastest throughput: {fastest['num_envs']} envs "
            f"({fastest['env_steps_per_s']:.0f} env_steps/s). "
            f"Largest successful: {largest['num_envs']} envs "
            f"(peak_mem={largest['mem_peak_gb']:.2f} GB)."
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
