"""Two outcome-independent physical-transfer illustrations at paper typography."""
import argparse
import json
from pathlib import Path
import numpy as np
from .scenarios import load_manifest, path_for
from .compare import condition_id, align_to_reference


def render(root, manifest_path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    root, output = Path(root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest, banks = load_manifest(manifest_path)
    conditions = {c["id"]: c for c in manifest["conditions"]}
    ids = ["rounded_waypoint_0.5_walk_to_crouch", "ood_corner_0.85_crouch_to_walk"]
    if any(ident not in manifest["smoke_ids"] for ident in ids):
        raise ValueError("Illustrations must have been predeclared")
    mj = root / "sim2sim_paper_mujoco_seed42"
    episodes = json.loads((mj / "episodes.json").read_text(encoding="utf-8"))
    native = root / "sim2sim_paper_isaac_transitions_seed42"
    metadata = json.loads((native / "timeseries_metadata.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.45, 1]})
    colors = {"Isaac / PhysX": "#1F4E79", "MuJoCo": "#D97925"}
    with np.load(mj / "traces.npz", allow_pickle=False) as m, np.load(native / "timeseries.npz", allow_pickle=False) as n:
        for col, ident in enumerate(ids):
            c = conditions[ident]; route = banks[(c["family"], c["repeat"])]; path, _, _ = path_for(c, route, manifest)
            mi = next(i for i, e in enumerate(episodes) if e["id"] == ident)
            scenario = next(s for s in metadata["scenarios"] if condition_id(s) == ident); ni = scenario["index"]
            np_xy = align_to_reference(n["positions_xy"][:, ni], scenario["path_xy"], path[:, :2])
            ax = axes[0, col]; hax = axes[1, col]
            ax.plot(path[:, 0], path[:, 1], "--", color=".3", lw=1.1, label="Reference")
            ax.scatter(*path[0, :2], s=15, facecolors="white", edgecolors=".25", zorder=4)
            ax.scatter(*path[-1, :2], s=22, color=".2", marker="x", zorder=4)
            for engine, xy, height, valid, required in (
                ("Isaac / PhysX", np_xy, n["achieved_height_m"][:, ni], n["valid"][:, ni], n["required_height_m"][:, ni]),
                ("MuJoCo", m[f"episode_{mi}_position"][:, :2], m[f"episode_{mi}_position"][:, 2],
                 m[f"episode_{mi}_valid"], m[f"episode_{mi}_required_height"])):
                times = (np.arange(len(valid)) + 1)[valid] * manifest["dt_s"]
                ax.plot(*xy[valid].T, color=colors[engine], lw=1.0, label=engine)
                hax.plot(times, height[valid], color=colors[engine], lw=.9)
                hax.plot(times, required[valid], color=colors[engine], ls="--", lw=.8, alpha=.65)
            ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("Route-frame x [m]"); ax.set_ylabel("y [m]")
            ax.set_title("Rounded transition, 0.50 m/s" if col == 0 else "Sharp-corner stress, 0.85 m/s")
            hax.set_xlabel("Active time [s]"); hax.set_ylabel("Base height [m]")
            hax.set_ylim(.1, .35)
            for panel in (ax, hax):
                panel.grid(alpha=.15)
        axes[0, 0].legend(loc="best", fontsize=7, frameon=False)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="scratch/wc_deployment")
    parser.add_argument("--manifest", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); render(args.root, args.manifest, args.output)


if __name__ == "__main__":
    main()
