"""Run existing structural/conditioning audits and plot actual expert timing."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_data import ROOT, OUT, PROTOCOL, repair_new_files

sys.path.insert(0, str(ROOT / "scripts/diffusion_policy/data"))
from merge_datasets import merge_hdf5_files
from audit_multiskill_hindsight import audit, make_plots
from audit_expert import audit as audit_expert
from audit_temporal import audit as audit_temporal
from train.data.dataset import _validate_hdf5


def pilot_index():
    source = ROOT / PROTOCOL["sources"]["crouch"]["reuse"]
    view = OUT / "crouch_pilot_view.hdf5"
    if not view.exists():
        with h5py.File(source, "r") as file, h5py.File(view, "x") as output:
            data = output.create_group("data")
            for key, value in file["data"].attrs.items():
                data.attrs[key] = value
            data.attrs["audit_subset_only"] = True
            names = sorted(file["data"], key=lambda name: int(name.split("_")[-1]))
            selected = np.random.default_rng(49).choice(len(names), 100, replace=False)
            for index, number in enumerate(selected):
                data[f"demo_{index}"] = h5py.ExternalLink("../" + source.name, "/data/" + names[number])
    paths = [OUT / "walk_pilot.hdf5", view, OUT / "pronk_pilot.hdf5"]
    index = OUT / "pilot_linked_final.hdf5"
    if not index.exists():
        merge_hdf5_files([str(path) for path in paths], str(index), link_data=True)
    return index


def timing_plot(paths, output):
    colors = {"walk": "#2878B5", "crouch": "#59A14F", "pronk": "#F28E2B"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.8), sharex=True, constrained_layout=True)
    examples = {}
    for column, (skill, path) in enumerate(paths.items()):
        with h5py.File(path, "r") as file:
            names = sorted(file["data"], key=lambda name: int(name.split("_")[-1]))
            # Walk illustrates longer reach (p90); other experts use the median.
            # Selection is by achieved speed, never by a prettier-looking gait.
            sampled = np.random.default_rng(49).choice(len(names), min(100, len(names)), replace=False)
            scores = []
            for number in sampled:
                pos = file["data"][names[number]]["obs/root_pos_w"][:]
                scores.append(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum() / ((len(pos) - 1) * .02))
            percentile = 90 if skill == "walk" else 50
            rank = int(round((len(scores) - 1) * percentile / 100))
            name = names[int(sampled[np.argsort(scores)[rank]])]
            obs = file["data"][name]["obs"]
            pos, quat, cmd = obs["root_pos_w"][:], obs["root_quat_w"][:], obs["command_speed"][:]
            yaw = np.arctan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                             1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
            delta = np.diff(pos[:, :2], axis=0) * 50
            vx = delta[:, 0] * np.cos(yaw[:-1]) + delta[:, 1] * np.sin(yaw[:-1])
            time = np.arange(len(pos)) * .02
            axes[0, column].plot(time, cmd[:, 0], "k--", lw=1.2, label="requested vx")
            axes[0, column].plot(time[:-1], vx, color=colors[skill], lw=1.2, label="achieved vx")
            axes[1, column].plot(time, pos[:, 2], color=colors[skill], lw=1.2, label="actual height")
            axes[1, column].plot(time, obs["desired_base_height"][:, 0], "k--", lw=1.2, label="posture condition")
            for axis in axes[:, column]:
                for tick in np.where(np.any(np.abs(np.diff(cmd, axis=0)) > 1e-6, axis=1))[0] + 1:
                    axis.axvline(time[tick], color="0.65", ls=":", lw=.8)
                axis.grid(alpha=.2)
                axis.spines[["top", "right"]].set_visible(False)
                axis.legend(fontsize=8)
            axes[0, column].set_title(skill)
            axes[1, column].set_xlabel("Recorded episode time [s]")
            axes[1, column].set_ylim(.14, .36)
            examples[skill] = {"source": str(path), "demo": name,
                              "selection": f"p{percentile} episode XY speed among up to 100 random demos, seed49"}
    axes[0, 0].set_ylabel("Body-frame forward speed [m/s]")
    axes[1, 0].set_ylabel("Base height [m]")
    fig.savefig(output / "expert_episode_timing.png", dpi=180)
    plt.close(fig)
    (output / "timing_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "full"), required=True)
    args = parser.parse_args()
    output = ROOT / "scripts/diffusion_policy/data/audits/walk_crouch_pronk_v1" / args.stage
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "pilot":
        repair_new_files("pilot")
    index = pilot_index() if args.stage == "pilot" else OUT / "walk_crouch_pronk_linked.hdf5"
    cfg = SimpleNamespace(dataset=str(index), expected_skills=["walk", "crouch", "pronk"],
                          max_frames_per_skill=30000, max_windows_per_skill=10000,
                          goal_horizon_steps=100, control_rate_hz=50.0, seed=49,
                          goal_representation="hindsight_geom_profile16")
    report, arrays = audit(cfg)
    (output / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["contract_valid"] or report["frame_balance_min_over_max"] != 1:
        raise ValueError(f"Invalid/unbalanced data: {report['contract_errors']}")
    make_plots(report, arrays, output)
    paths = {"walk": OUT / f"walk_{args.stage}.hdf5",
             "crouch": ROOT / PROTOCOL["sources"]["crouch"]["reuse"],
             "pronk": OUT / f"pronk_{args.stage}.hdf5"}
    # Use the real trainer validator on EVERY frame, not just the support sample.
    for path in paths.values():
        _validate_hdf5(str(path), require_reference_path=False, require_desired_height_profile=True)
    diagnostics = {skill: audit_expert(path) for skill, path in paths.items()}
    (output / "expert_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if args.stage == "full":
        temporal = {skill: audit_temporal(path, recorded_startup_ticks=26 if skill == "crouch" else 25)
                    for skill, path in paths.items()}
        (output / "temporal_alignment.json").write_text(json.dumps(temporal, indent=2), encoding="utf-8")
    timing_plot(paths, output)
    print(json.dumps({"contract_valid": report["contract_valid"], "frame_balance": report["frame_balance_min_over_max"],
                      "skills": report["skills"], "plots": str(output)}, indent=2))


if __name__ == "__main__":
    main()
