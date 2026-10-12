"""Frozen pilot: pure WCP imitation prior + gait-agnostic nominal posture.

Run preflight with plain Python; run training through isaaclab.sh/isaaclab.bat.
No Isaac imports until every provenance/output check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[4]
PRIOR = ROOT / "scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10/best.pt"
PRIOR_SHA256 = "40caf9ad17846797980c96ef9a621474ca5bb730239185edf9b32cc9cc3d7886"
OUTPUT = ROOT / "scripts/dppo_diffusion_rl/runs/dppo_wcp_nominal_posture_v1_seed42"


def preflight() -> None:
    if not PRIOR.is_file():
        raise FileNotFoundError(f"Missing frozen Phase-A checkpoint: {PRIOR}")
    digest = hashlib.sha256()
    with PRIOR.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != PRIOR_SHA256:
        raise ValueError("Wrong Phase-A prior; no fallback to an old DPPO/WCT actor.")
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite or mix runs: {OUTPUT}")
    print(f"PREFLIGHT_OK prior_sha256={PRIOR_SHA256} output={OUTPUT}", flush=True)


def training_arguments(device: str, wandb: bool) -> list[str]:
    args = [
        "--checkpoint", str(PRIOR), "--require_phase_a_source",
        "--output_dir", str(OUTPUT), "--run_name", "dppo_wcp_nominal_posture_v1_seed42",
        "--posture_tracking_mode", "nominal_window", "--posture_window_s", "0.4",
        "--vertical_velocity_reward_weight", "0.0",
        "--route_distribution", "supported_hybrid_v3",
        "--route_speed_max_mps", "1.0", "--speed_budget_max_mps", "1.5",
        "--transition_boundary_min_m", "0.8", "--transition_boundary_max_m", "3.2",
        "--transition_margin_m", "0.0",
        "--path_reward_weight", "2.0", "--route_cte_rmse_tolerance_m", "0.10",
        "--profile_height_reward_weight", "0.75", "--profile_height_mae_tolerance_m", "0.04",
        "--reference_kl_coef", "0.0", "--symmetry_augmentation", "none",
        "--num_envs", "4096", "--iterations", "150", "--rollout_chunks", "32",
        "--inference_steps", "10", "--finetune_denoising_steps", "5", "--exec_horizon", "4",
        "--critic_warmup_iterations", "10", "--episode_length_s", "24",
        "--select_rollout_actor", "--save_interval", "10", "--seed", "42",
        "--headless", "--device", device,
    ]
    if wandb:
        args.append("--wandb")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    preflight()
    if args.preflight_only:
        return
    sys.path.insert(0, str(ROOT))
    sys.argv = [str(ROOT / "scripts/dppo_diffusion_rl/train.py"), *training_arguments(args.device, args.wandb)]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
