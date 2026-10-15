"""Separate first-entry braking from settled motion; retain the original screen."""
import argparse
import json
from pathlib import Path
import numpy as np
from .paper_analysis import hold_metrics
from .scenarios import load_manifest, path_for
from .reporting import save_json
from .convert import sha


def motion_window(xy, height, valid, start_s, endpoint, target_height, *, dt=.02):
    result = hold_metrics(xy, height, valid, start_s, endpoint, target_height, dt)
    if "speed_p95_mps" not in result:
        return result
    start = round(start_s / dt)
    points = np.asarray(xy)[start:start + round(5 / dt)]
    lag = round(.5 / dt)
    speed = np.linalg.norm(points[lag:] - points[:-lag], axis=1) / (lag * dt)
    result.update(net_displacement_speed_mps=float(np.linalg.norm(points[-1] - points[0]) / ((len(points)-1)*dt)),
                  half_second_speed_p95_mps=float(np.percentile(speed, 95)))
    # The old raw p95 and pass predicate remain unchanged. Averaging diagnostics
    # expose gait oscillations; they are not a replacement safety certificate.
    return result


def load_attempts(root, manifest_path):
    manifest, banks = load_manifest(manifest_path)
    attempts = []
    for seed in manifest["evaluation_seeds"]:
        directory = root / f"sim2sim_paper_mujoco_seed{seed}"
        rows = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
        with np.load(directory / "traces.npz", allow_pickle=False) as traces:
            for i, row in enumerate(rows):
                path, _, _ = path_for(row, banks[(row["family"], row["repeat"])], manifest)
                attempts.append(dict(engine="MuJoCo", seed=seed, id=row["id"], arrival=row.get("task", {}).get("time_s") if row["arrived"] else None,
                    xy=traces[f"episode_{i}_position"][:, :2], height=traces[f"episode_{i}_position"][:, 2],
                    valid=traces[f"episode_{i}_valid"], endpoint=path[-1, :2], target_height=float(path[-1, 2]),
                    conditioning_speed=traces[f"episode_{i}_conditioning_speed"]))
        for part in ("constants", "transitions"):
            directory = root / f"sim2sim_paper_isaac_{part}_seed{seed}"
            rows = json.loads((directory / "episode_metrics.json").read_text(encoding="utf-8"))
            metadata = json.loads((directory / "timeseries_metadata.json").read_text(encoding="utf-8"))
            with np.load(directory / "timeseries.npz", allow_pickle=False) as traces:
                for row in rows:
                    i = int(row["scenario_id"])
                    profile = row["transition_direction"] or ("walk" if np.isclose(row["requested_height"], .2932) else "crouch")
                    attempts.append(dict(engine="Isaac", seed=seed, id=f"{row['path_shape']}_{row['requested_speed']:g}_{profile}",
                        arrival=row["task_success_time_s"] if row["task_arrived"] else None,
                        xy=traces["positions_xy"][:, i], height=traces["achieved_height_m"][:, i], valid=traces["valid"][:, i],
                        endpoint=np.asarray(metadata["scenarios"][i]["path_xy"])[-1],
                        target_height=.2932 if profile in ("walk", "crouch_to_walk") else .1705))
    if len(attempts) != 576 or len({(a["engine"], a["seed"], a["id"]) for a in attempts}) != 576:
        raise ValueError("Missing/duplicated physical attempts")
    return attempts


def analyze(root, manifest_path, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    attempts = load_attempts(root, manifest_path)
    rows = []
    for attempt in attempts:
        arrival = attempt["arrival"]
        for name, delay in (("first_entry", 0.), ("after_1s", 1.), ("after_3s", 3.), ("last_5s", None)):
            start = None if arrival is None else (19. if delay is None else arrival + delay)
            if start is not None and start < arrival:
                metric = {"assessed": False, "pass": False, "reason": "window_precedes_arrival"}
            else:
                metric = motion_window(attempt["xy"], attempt["height"], attempt["valid"], start,
                                       attempt["endpoint"], attempt["target_height"])
            rows.append(dict(engine=attempt["engine"], seed=attempt["seed"], id=attempt["id"], window=name, **metric))
    groups = {}
    for engine in ("Isaac", "MuJoCo"):
        groups[engine] = {}
        for name in ("first_entry", "after_1s", "after_3s", "last_5s"):
            selected = [r for r in rows if r["engine"] == engine and r["window"] == name]
            finite = [r for r in selected if "speed_p95_mps" in r]
            groups[engine][name] = dict(attempts=len(selected), assessed=sum(r["assessed"] for r in selected),
                valid_windows=len(finite), original_strict_passes=sum(r["pass"] for r in selected),
                raw_speed_below_015=sum(r["speed_p95_mps"] <= .15 for r in finite),
                half_second_speed_below_015=sum(r["half_second_speed_p95_mps"] <= .15 for r in finite),
                position_within_020=sum(r["position_p95_m"] <= .2 for r in finite),
                median_net_displacement_speed_mps=float(np.median([r["net_displacement_speed_mps"] for r in finite])),
                median_half_second_speed_p95_mps=float(np.median([r["half_second_speed_p95_mps"] for r in finite])))
    save_json(output / "stopping.json", {"summary": groups, "rows": rows,
        "manifest_sha256": sha(manifest_path), "source_sha256": sha(__file__),
        "scope": "Post-hoc diagnostic of existing traces; unchanged actor and original raw screen.",
        "limitations": ["Averaged base translation is not instantaneous motion or hardware safety.",
                        "Arrival is geometric first entry, not stopped arrival.",
                        "Nonarrivals and invalid/incomplete windows remain explicit.",
                        "Late drift and terminal height error are separate unresolved outcomes."]})
    plot(attempts, output)
    print(json.dumps(groups, indent=2))


def plot(attempts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids = ("straight_0.5_walk", "rounded_waypoint_0.5_walk_to_crouch")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for column, identifier in enumerate(ids):
        for engine, color in (("Isaac", "#1F4E79"), ("MuJoCo", "#D97925")):
            a = next(x for x in attempts if x["seed"] == 42 and x["engine"] == engine and x["id"] == identifier)
            xy, valid = a["xy"], a["valid"]
            times = (np.arange(len(xy)) + 1) * .02 - a["arrival"]
            raw = np.linalg.norm(np.diff(xy, axis=0), axis=1) / .02
            raw[~(valid[1:] & valid[:-1])] = np.nan
            speed = np.linalg.norm(xy[25:] - xy[:-25], axis=1) / .5
            speed[np.convolve(valid.astype(int), np.ones(26, int), "valid") != 26] = np.nan
            axes[0, column].plot((times[1:]+times[:-1])*.5, raw, color=color, alpha=.18, linewidth=.7)
            axes[0, column].plot((times[25:]+times[:-25])*.5, speed, color=color, label=engine, linewidth=1.4)
            distance = np.linalg.norm(xy - a["endpoint"], axis=1)
            distance[~valid] = np.nan
            distance[(times < -1) | (times > 10)] = np.nan
            axes[1, column].plot(times, distance, color=color, label=engine)
        axes[0, column].set_title(identifier.replace("_", " "), fontsize=10)
        axes[0, column].axhline(.15, color=".5", linestyle=":")
        axes[1, column].axhline(.2, color=".5", linestyle=":")
        for ax in axes[:, column]:
            ax.axvline(0, color=".4", linestyle="--", linewidth=.8)
            ax.set_xlim(-1, 10); ax.set_ylim(bottom=0); ax.grid(alpha=.15)
        axes[1, column].set_xlabel("Time since first geometric arrival [s]")
    axes[0, 0].set_ylabel("Planar speed [m/s]")
    axes[1, 0].set_ylabel("Endpoint distance [m]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Braking is not settled holding: thin = 20 ms, thick = 0.5 s displacement", fontsize=10)
    fig.savefig(output / "stopping.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("scratch/wc_deployment"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); analyze(args.root, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
