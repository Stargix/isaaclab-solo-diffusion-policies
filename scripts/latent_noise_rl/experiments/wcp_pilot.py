"""Canonical WCP latent-noise PPO pilot contract and evaluation commands."""

from __future__ import annotations

import argparse
from pathlib import Path

SOURCE_SHA256 = "40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886"
# Evaluation supplies its own frozen paths. The training task has an internal
# random route bank and task terminations, so playback uses the base physical env.
TASK = "solo12-v0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = PROJECT_ROOT / "scripts" / "diffusion_policy" / "evaluate_policy.py"
ROUTE_BANK = "scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/route_banks/wct_id_routes_v1.npz"
ID_FAMILIES = ("procedural", "coherent_smooth", "rounded_waypoint", "hard_waypoint")
REPEATS = (0, 1)
UPRIGHT_SPEEDS = (0.2, 0.35, 0.5, 0.65, 0.85)
CROUCH_SPEEDS = (0.2, 0.35, 0.45)
TRANSITION_SPEEDS = (0.35, 0.65)
TRANSITION_DIRECTIONS = ("walk_to_crouch", "crouch_to_walk")
WALK_HEIGHT_M = 0.2932
CROUCH_HEIGHT_M = 0.1705
EVAL_SEED = 4242


def evaluation_commands(*, launcher: str, checkpoint: str, output_root: str,
                        route_bank: str = ROUTE_BANK) -> list[str]:
    """Construct the exact three-mode, three-part paired evaluation matrix."""
    route_bank_path = Path(route_bank)
    if not route_bank_path.is_absolute():
        route_bank_path = PROJECT_ROOT / route_bank_path
    parts = (
        ("upright", ("--path_height", str(WALK_HEIGHT_M), "--speeds", "0.2", "0.35", "0.5", "0.65", "0.85")),
        ("crouch", ("--path_height", str(CROUCH_HEIGHT_M), "--speeds", "0.2", "0.35", "0.45")),
        ("transitions", ("--path_height", str(WALK_HEIGHT_M), "--speeds", "0.35", "0.65",
                          "--transition_fractions", "0.5", "--transition_directions", "both")),
    )
    modes = ("checkpoint", "initial", "gaussian")
    commands = []
    for mode in modes:
        for name, condition in parts:
            output = Path(output_root) / f"{mode}_{name}"
            args = [launcher, "-p", str(EVALUATOR),
                    "--task", TASK, "--checkpoint", checkpoint, "--output_dir", str(output),
                    "--path_shapes", *ID_FAMILIES, "--route_bank", str(route_bank_path),
                    "--repeats", "2", "--duration_s", "24", "--route_length_m", "4.0",
                    "--warmup_steps", "25", "--posture_settle_steps", "50",
                    "--seed", str(EVAL_SEED), "--exec_horizon", "4",
                    "--num_inference_steps", "10", "--guidance_scale", "1.0",
                    "--latent_reference_mode", mode, "--deterministic_resets",
                    "--record_staging_failures", "--save_timeseries", "--require_empty_output_dir",
                    *condition, "--headless"]
            commands.append(" ".join(f'"{value}"' if " " in value else value for value in args))
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", default="./isaaclab.sh")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--route_bank", default=ROUTE_BANK)
    args = parser.parse_args()
    for command in evaluation_commands(launcher=args.launcher, checkpoint=args.checkpoint,
                                      output_root=args.output_root, route_bank=args.route_bank):
        print(command)


if __name__ == "__main__":
    main()
