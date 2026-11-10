"""Measure frozen experts and posture switches; no policy optimization."""

import argparse
import json
from pathlib import Path
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--walk_checkpoint", required=True)
parser.add_argument("--crouch_checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
output = Path(args.output_dir)
if output.exists() or args.steps < 100:
    raise ValueError("Use a new output directory and at least 100 steps.")
app = AppLauncher(args).app
try:
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from scripts.hierarchical_experts.policy import Experts
    from scripts.dppo_diffusion_rl.checkpointing import sha256_file

    cases = [(skill, skill, speed) for skill in (0, 1)
             for speed in ((0., .25, .5, .75, 1., 1.5) if skill == 0 else (0., .25, .5, .75, 1.))]
    cases += [(a, b, speed) for a, b in ((0, 1), (1, 0)) for speed in (.25, .5, .75)]
    cfg = parse_env_cfg("solo12-v0", device=args.device, num_envs=len(cases), use_fabric=True)
    cfg.seed = args.seed
    cfg.episode_length_s = max(24., args.steps * .02 + 1.)
    cfg.enable_observation_corruption = False
    cfg.actuation_delay_range = (0, 0)
    cfg.events = None
    cfg.base_push_interval_range_s = (1.e9, 1.e9)
    cfg.terrain.terrain_type = "plane"
    env = gym.make("solo12-v0", cfg=cfg)
    try:
        raw = env.unwrapped
        experts = Experts(args.walk_checkpoint, args.crouch_checkpoint).to(raw.device)
        env.reset()
        cases_tensor = torch.tensor(cases, device=raw.device)
        speed = cases_tensor[:, 2]
        command = torch.zeros(len(cases), 3, device=raw.device)
        command[:, 0] = speed
        failures = torch.zeros(len(cases), device=raw.device)
        squared = torch.zeros(len(cases), 2, device=raw.device)
        height = torch.zeros_like(squared)
        count = torch.zeros_like(squared)
        for step in range(args.steps):
            phase = int(step >= args.steps // 2)
            skill = cases_tensor[:, phase].long()
            action = experts.action(raw, skill, command)
            if not torch.isfinite(action).all():
                raise FloatingPointError("Non-finite expert action.")
            _, _, term, trunc, _ = env.step(action)
            failures += (term | trunc).float()
            # Exclude one second after start or the midpoint switch.
            if step % (args.steps // 2) >= 50:
                squared[:, phase] += (raw._robot.data.root_lin_vel_b[:, 0] - speed).square()
                height[:, phase] += raw._robot.data.root_pos_w[:, 2] - raw.scene.env_origins[:, 2]
                count[:, phase] += 1
        rows = [{"initial_skill": int(a), "final_skill": int(b), "vx_request": v,
                 "resets": int(failures[i]), "speed_rmse_by_phase": (squared[i] / count[i].clamp_min(1)).sqrt().tolist(),
                 "height_by_phase": (height[i] / count[i].clamp_min(1)).tolist(),
                 "samples_by_phase": count[i].tolist()} for i, (a, b, v) in enumerate(cases)]
        output.mkdir(parents=True)
        (output / "probe.json").write_text(json.dumps({"seed": args.seed, "steps": args.steps,
            "expert_sha256": {"walk": sha256_file(args.walk_checkpoint), "crouch": sha256_file(args.crouch_checkpoint)},
            "cases": rows}, indent=2), encoding="utf-8")
        print("[PROBE_OK] Results written; inspect tracking, heights and resets before selecting bounds.", flush=True)
    finally:
        env.close()
finally:
    app.close()
