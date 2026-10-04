"""Snapshot the SAME Phase-A prior as the archived WC/WCT-v3 experiments.

If the original best.pt was replaced, recover only the immutable Phase-A
denoiser from a hash-verified historical DPPO file. Never use its trained actor,
critic, reference, optimizer, iteration or best score as initialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

from scripts.dppo_diffusion_rl.checkpointing import load_policy_checkpoint, sha256_file
from scripts.dppo_diffusion_rl.config import DPPOConfig
from scripts.dppo_diffusion_rl.symmetry import LeftRightAugmentation


PHASE_A_SHA = {
    "wc": "b94a6ae7e1da298ff0a33b314adc5e6639a059a61452d423cd926fd9103fedbd",
    "wct": "2f28fd4beb963c74293948507eecd5e7061f34a1436d8014626bd60f2d73da54",
}
ARCHIVE_SHA = {
    "wc": "2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868",
    "wct": "5258fcf239a391bcfe6e4a951944b037b94441113d137b782ad5ade8341fa55f",
}
PHASE_A_PATHS = {
    "wc": (
        "scripts/diffusion_policy/runs/wc_wct_ablation_v1_phase_a_wc_k10/best.pt",
        "checkpoints_iri/checkpoints_dp/wc_wct_phase_a.pt",
    ),
    "wct": (
        "scripts/diffusion_policy/runs/walk_crouch_sprint_hindsight_geom_profile16_a_wcs_v1_k10/best.pt",
        "checkpoints_iri/checkpoints_dp/wct_diffusion_policy_baseline.pt",
    ),
}
ARCHIVE_PATHS = {
    "wc": (
        "scripts/dppo_diffusion_rl/runs/dppo_wc_supported_hybrid_v3/best.pt",
        "checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt",
    ),
    "wct": (
        "scripts/dppo_diffusion_rl/runs/dppo_wct_supported_hybrid_v3/best.pt",
        "checkpoints_iri/checkpoints_dppo/dppo_wct_hybrid_strict.pt",
    ),
}


def recover_phase_a(archive: dict, expected_source_sha: str) -> dict:
    if archive.get("algorithm") != "dppo":
        raise ValueError("Phase-A recovery requires a historical DPPO checkpoint.")
    if archive.get("source_checkpoint_sha256") != expected_source_sha:
        raise ValueError("Historical DPPO checkpoint was not initialized from the required Phase A.")
    base = archive["dppo_base_model_state_dict"]
    state = {"model." + name: value for name, value in base.items()}
    if not state or set(state) != set(archive["ema_model_state_dict"]):
        raise ValueError("Frozen base keys do not exactly cover the original policy model.")
    recovered = {
        name: archive[name]
        for name in ("schema_version", "policy_kind", "config", "normalizer_stats", "split_manifest")
        if name in archive
    }
    recovered.update(
        model_state_dict=state, ema_model_state_dict=state,
        recovered_phase_a_source_sha256=expected_source_sha,
    )
    return recovered


def _matching(paths, expected):
    for value in paths:
        path = Path(value)
        if path.is_file() and sha256_file(path) == expected:
            return path
    return None


def prepare_source(family: str, output_dir: Path) -> Path:
    snapshot = output_dir / "phase_a_source.pt"
    provenance_path = output_dir / "source_provenance.json"
    if snapshot.exists() or provenance_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Phase-A source snapshot.")
    source = _matching(PHASE_A_PATHS[family], PHASE_A_SHA[family])
    method = "original_phase_a_copy"
    if source is None:
        source = _matching(ARCHIVE_PATHS[family], ARCHIVE_SHA[family])
        if source is None:
            raise FileNotFoundError(
                f"No hash-matching {family.upper()} Phase A or historical DPPO archive. "
                f"Expected Phase A {PHASE_A_SHA[family]}; archive {ARCHIVE_SHA[family]}."
            )
        archive = torch.load(source, map_location="cpu", weights_only=False)
        recovered = recover_phase_a(archive, PHASE_A_SHA[family])
        method = "recovered_immutable_phase_a_base"
    output_dir.mkdir(parents=True, exist_ok=True)
    if method == "original_phase_a_copy":
        shutil.copyfile(source, snapshot)
        if sha256_file(snapshot) != PHASE_A_SHA[family]:
            raise RuntimeError("Source changed while snapshotting; refusing to train.")
    else:
        torch.save(recovered, snapshot)
    # Validate the actual snapshot, not just metadata; this imports no simulator.
    checkpoint, policy, _ = load_policy_checkpoint(
        snapshot, torch.device("cpu"),
        DPPOConfig(symmetry_augmentation="mirror", select_rollout_actor=True),
    )
    if checkpoint.get("algorithm") == "dppo" or policy.cfg.goal_dim != 16:
        raise ValueError("Expected pure Phase-A profile16 policy, not a fine-tuned actor.")
    LeftRightAugmentation(policy)
    provenance = {
        "family": family, "method": method, "original_phase_a_sha256": PHASE_A_SHA[family],
        "container_path": str(source.resolve()), "container_sha256": sha256_file(source),
        "snapshot_path": str(snapshot.resolve()), "snapshot_sha256": sha256_file(snapshot),
        "initialization": "Phase-A denoiser only; fresh critic and optimizers",
        "history": policy.cfg.history, "prediction_horizon": policy.cfg.prediction_horizon,
        "execution_offset": policy.cfg.execution_offset,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{family.upper()}: {method}; original prior={PHASE_A_SHA[family]}", file=sys.stderr)
    return snapshot.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("wc", "wct"), required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_source(args.family, args.output_dir))


if __name__ == "__main__":
    main()
