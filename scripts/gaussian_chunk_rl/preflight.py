"""Checkpoint/mean/likelihood gate, runnable without starting Isaac Sim."""

import argparse
import json
import torch
from scripts.dppo_diffusion_rl.checkpointing import sha256_file
from .config import GaussianPPOConfig
from .policy import load_bc_policy


def check(checkpoint, device="cpu"):
    source, actor = load_bc_policy(checkpoint, device, GaussianPPOConfig())
    inputs = tuple(torch.zeros(2, actor.cfg.history, dim, device=device)
                   for dim in (actor.cfg.proprio_dim, actor.cfg.action_hist_dim, actor.cfg.goal_dim))
    with torch.no_grad():
        trajectory, _ = actor.distribution(*inputs)
        torch.testing.assert_close(trajectory.clamp(-1, 1), actor.policy.predict_action(*inputs))
        sample = actor.sample(*inputs)
        recomputed, _ = actor.logprob(*inputs, sample.chain[:, 0])
        torch.testing.assert_close(recomputed, sample.old_logprobs[:, 0])
    return {"checkpoint_sha256": sha256_file(checkpoint), "algorithm": source["algorithm"],
            "mean_parameters": sum(p.numel() for p in actor.policy.parameters()),
            "history": actor.cfg.history, "prediction_horizon": actor.cfg.prediction_horizon,
            "execution_offset": actor.cfg.execution_offset, "exec_horizon": actor.ppo_cfg.exec_horizon,
            "goal_dim": actor.cfg.goal_dim, "mean_and_likelihood_gate": "passed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(check(args.checkpoint, args.device), indent=2))
