"""Deterministic, paired evaluation for a Phase-B1 residual PPO checkpoint.

The evaluator intentionally measures terminal route outcomes rather than
training return.  A new environment is constructed for each checkpoint with
the same seed and route stage, then the actor mean is executed without PPO
exploration noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate one residual-diffusion PPO checkpoint.")
parser.add_argument("--task", default="solo12-residual-diffusion-rl-v0")
parser.add_argument("--checkpoint", required=True, help="RSL-RL residual PPO checkpoint.")
parser.add_argument("--diffusion_checkpoint", required=True, help="Frozen Phase-A diffusion checkpoint.")
parser.add_argument("--output_dir", required=True, help="Directory in which to write summary.json.")
parser.add_argument("--episodes", type=int, default=256, help="Minimum number of completed episodes.")
parser.add_argument("--num_envs", type=int, default=64, help="Parallel environments.")
parser.add_argument("--seed", type=int, default=42, help="Shared route RNG seed for paired comparison.")
parser.add_argument("--stage", type=int, choices=(0, 1, 2), default=0, help="RouteBank curriculum stage.")
parser.add_argument("--max_steps", type=int, default=20_000, help="Safety bound on vectorized control steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _count(log: dict, key: str) -> int:
    """Return a reset-event count logged by ResidualDiffusionEnv."""

    value = log.get(key, 0)
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    if args_cli.episodes <= 0 or args_cli.num_envs <= 0 or args_cli.max_steps <= 0:
        raise ValueError("--episodes, --num_envs and --max_steps must be positive")

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

    # RouteBank selects its stage from this common control-step counter.  Keep
    # the MDP fixed for all evaluated episodes rather than allowing an
    # evaluation run to change difficulty halfway through.
    if args_cli.stage == 0:
        env_cfg.curriculum_stage1_steps = args_cli.max_steps + 1
        env_cfg.curriculum_stage2_steps = args_cli.max_steps + 2
    elif args_cli.stage == 1:
        env_cfg.curriculum_stage1_steps = 0
        env_cfg.curriculum_stage2_steps = args_cli.max_steps + 1
    else:
        env_cfg.curriculum_stage1_steps = 0
        env_cfg.curriculum_stage2_steps = 0

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint_path = Path(args_cli.checkpoint).resolve()
    runner.load(str(checkpoint_path), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = runner.alg.policy
    observations = env.get_observations()
    outcomes = {"route_success": 0, "base_contact": 0, "corridor_failure": 0, "time_out": 0}
    completed = 0
    reward_sum = 0.0
    length_sum_s = 0.0
    steps = 0

    with torch.inference_mode():
        while completed < args_cli.episodes and steps < args_cli.max_steps:
            actions = policy(observations)
            observations, _, dones, extras = env.step(actions)
            policy_nn.reset(dones)
            steps += 1

            done_count = int(dones.sum().item())
            if done_count == 0:
                continue
            completed += done_count
            log = extras.get("log", {})
            for name in outcomes:
                outcomes[name] += _count(log, f"Episode_Termination/{name}")
            reward_sum += float(log.get("Episode_Reward/total", 0.0)) * done_count
            length_sum_s += float(log.get("Episode/length_seconds", 0.0)) * done_count

    env.close()
    if completed == 0:
        raise RuntimeError("No evaluation episode completed; increase --max_steps")

    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path),
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
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[PASS] Wrote paired B1 evaluation to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
