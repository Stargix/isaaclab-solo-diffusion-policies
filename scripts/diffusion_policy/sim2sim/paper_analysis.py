"""Audit all predeclared transfer attempts; no outcome-filtered example selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .compare import compare, finite_mean
from .scenarios import load_manifest, path_for
from .reporting import save_json
from .convert import sha


def hold_metrics(position, height, valid, arrival_s, target_xy, target_height, dt):
    """Five seconds after first arrival, with failure/insufficient horizon counted."""
    result = {"assessed": False, "pass": False}
    if arrival_s is None or not np.isfinite(arrival_s):
        return result
    start = int(round(arrival_s / dt))
    stop = start + round(5. / dt)
    result["assessed"] = True
    if stop > len(valid) or not np.all(valid[start:stop]):
        result["failure"] = "invalid_or_incomplete_hold_window"
        return result
    xy = np.asarray(position)[start:stop]
    speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    position_p95 = float(np.percentile(np.linalg.norm(xy - target_xy, axis=1), 95))
    speed_p95 = float(np.percentile(speed, 95))
    height_mae = float(np.mean(np.abs(np.asarray(height)[start:stop] - target_height)))
    result.update(position_p95_m=position_p95, position_max_m=float(np.max(np.linalg.norm(xy-target_xy, axis=1))),
                  speed_p95_mps=speed_p95, height_mae_m=height_mae,
                  **{"pass": bool(position_p95 <= .2 and speed_p95 <= .15 and height_mae <= .03)})
    return result


def summarize(rows):
    groups = {"all": rows}
    for speed in (.25, .5, .85):
        groups[f"speed_{speed:g}"] = [r for r in rows if r["speed"] == speed]
    for scope in ("ordinary", "stress"):
        groups[scope] = [r for r in rows if r["scope"] == scope]
    for profile in ("walk", "crouch", "walk_to_crouch", "crouch_to_walk"):
        groups[profile] = [r for r in rows if r["profile"] == profile]
    return {key: {"n": len(g),
                  **{field: sum(bool(r[field]) for r in g) for field in
                     ("arrived", "success", "adherence", "survival", "hold_pass")},
                  "hold_assessed": sum(r["hold"]["assessed"] for r in g),
                  "tracking_n": sum(r["cte"] is not None for r in g),
                  "mean_cte_m": finite_mean([r["cte"] for r in g]),
                  "mean_height_mae_m": finite_mean([r["height_mae"] for r in g])}
            for key, g in groups.items()}


def analyze(root, manifest_path, output):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest, banks = load_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    engines = {"Isaac": [], "MuJoCo": []}; hashes = {}; per_seed = {}
    for seed in manifest["evaluation_seeds"]:
        mjdir = root / f"sim2sim_paper_mujoco_seed{seed}"
        native_dirs = [root / f"sim2sim_paper_isaac_{p}_seed{seed}" for p in ("constants", "transitions")]
        comparison = compare(manifest_path, mjdir, native_dirs, output / f"seed{seed}")
        per_seed[str(seed)] = comparison
        with np.load(mjdir / "traces.npz", allow_pickle=False) as traces:
            episodes = json.loads((mjdir / "episodes.json").read_text(encoding="utf-8"))
            for i, e in enumerate(episodes):
                p = traces[f"episode_{i}_position"]
                path, _, _ = path_for(e, banks[(e["family"], e["repeat"])], manifest)
                hold = hold_metrics(p[:, :2], p[:, 2], traces[f"episode_{i}_valid"],
                                    e.get("task", {}).get("time_s") if e["arrived"] else None,
                                    path[-1, :2], path[-1, 2], manifest["dt_s"])
                tracking = e.get("tracking", {})
                engines["MuJoCo"].append(dict(id=e["id"], seed=seed, profile=e["profile"], speed=e["speed_mps"],
                    scope="stress" if e["family"].startswith("ood_") else "ordinary", arrived=e["arrived"],
                    success=e["task_success"], adherence=e["adherence_success"], survival=e["full_horizon_survival"],
                    cte=tracking.get("cross_track_rmse_m"), height_mae=tracking.get("height_mae_m"),
                    hold=hold, hold_pass=hold["pass"]))
        for directory in native_dirs:
            ep = directory / "episode_metrics.json"
            rows = json.loads(ep.read_text(encoding="utf-8")); hashes[str(ep)] = sha(ep)
            metadata = json.loads((directory / "timeseries_metadata.json").read_text(encoding="utf-8"))
            with np.load(directory / "timeseries.npz", allow_pickle=False) as traces:
                for row in rows:
                    i = int(row["scenario_id"]); scenario = metadata["scenarios"][i]
                    profile = row["transition_direction"] or ("walk" if np.isclose(row["requested_height"], .2932) else "crouch")
                    target_h = .2932 if profile in ("walk", "crouch_to_walk") else .1705
                    hold = hold_metrics(traces["positions_xy"][:, i], traces["achieved_height_m"][:, i],
                                        traces["valid"][:, i], row["task_success_time_s"] if row["task_arrived"] else None,
                                        np.asarray(scenario["path_xy"])[-1], target_h, manifest["dt_s"])
                    cte = row["active_cross_track_rmse_m"]
                    engines["Isaac"].append(dict(id=f"{row['path_shape']}_{row['requested_speed']:g}_{profile}",
                        seed=seed, profile=profile, speed=row["requested_speed"],
                        scope="stress" if row["path_shape"].startswith("ood_") else "ordinary",
                        arrived=row["task_arrived"], success=row["task_success"], survival=row["survived"],
                        adherence=bool(row["task_success"] and cte is not None and cte <= .1),
                        cte=cte, height_mae=row["active_height_mae_m"], hold=hold, hold_pass=hold["pass"]))
        for filename in ("episodes.json", "traces.npz", "run_manifest.json"):
            hashes[str(mjdir / filename)] = sha(mjdir / filename)
        selected = [r for r in engines["Isaac"] if r["seed"] == seed]
        for field, source in (("success", "task_success"), ("arrived", "arrived"), ("survival", "full_horizon_survival")):
            if sum(bool(r[field]) for r in selected) != comparison["Isaac"][source]:
                raise ValueError(f"Per-attempt records disagree with official aggregate: seed{seed}, {field}")
    if any(len(rows) != 288 or len({(r["id"], r["seed"]) for r in rows}) != 288 for rows in engines.values()):
        raise ValueError("Missing or duplicated paired attempts")
    summary = {engine: summarize(rows) for engine, rows in engines.items()}
    save_json(output / "all_attempts.json", engines)
    save_json(output / "results.json", {"summary": summary, "per_seed": per_seed, "source_hashes": hashes,
              "units": "Eight fixed geometries, three inference seeds of one trained actor; no training-seed uncertainty",
              "hold_gate": {"window_s": 5., "position_p95_m": .2, "speed_p95_mps": .15, "height_mae_m": .03,
                            "speed_measurement": "20 ms finite-difference planar position, identical extraction in both engines",
                            "not_hardware_certification": True}})
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="scratch/wc_deployment")
    parser.add_argument("--manifest", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); analyze(args.root, args.manifest, args.output_dir)


if __name__ == "__main__":
    main()
