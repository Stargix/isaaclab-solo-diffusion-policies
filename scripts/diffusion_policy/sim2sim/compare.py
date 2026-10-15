"""Compare paired target banks using official JSON metrics and immutable NPZ traces."""
import argparse
import json
from pathlib import Path
import numpy as np
from .convert import sha
from .reporting import save_json
from .scenarios import load_manifest, path_for


def finite_mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def task_success_count(summary):
    # Historical key counts applicable scenarios, not successful scenarios.
    count = summary["task_success_scenarios"]
    return int(round(summary["overall_task_success_rate"] * count))


def align_to_reference(points, native_path, reference):
    """Remove only the evaluator's global translation/yaw; never fit the rollout."""
    native_path, reference = np.asarray(native_path), np.asarray(reference)
    a, b = native_path[1] - native_path[0], reference[1] - reference[0]
    angle = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = (native_path - native_path[0]) @ rotation.T + reference[0]
    if transformed.shape != reference.shape or not np.allclose(transformed, reference, atol=2e-5):
        raise ValueError("Native route is not the same frozen geometry up to global pose")
    return (np.asarray(points) - native_path[0]) @ rotation.T + reference[0]


def condition_id(scenario):
    profile = scenario["transition_direction"]
    if scenario["transition_fraction"] is None:
        height = scenario["path_height_m"]
        if np.isclose(height, .2932):
            profile = "walk"
        elif np.isclose(height, .1705):
            profile = "crouch"
        else:
            raise ValueError("Unexpected constant height")
    elif scenario["transition_fraction"] != .5:
        raise ValueError("Unexpected transition location")
    return f"{scenario['path_shape']}_{scenario['requested_speed_m_s']:g}_{profile}"


def compare(manifest_path, mujoco_dir, isaac_dirs, output):
    manifest, banks = load_manifest(manifest_path)
    mujoco_dir, output = Path(mujoco_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    run = json.loads((mujoco_dir / "run_manifest.json").read_text(encoding="utf-8"))
    episodes = json.loads((mujoco_dir / "episodes.json").read_text(encoding="utf-8"))
    expected = {c["id"]: c for c in manifest["conditions"]}
    count = len(expected)
    if {e["id"] for e in episodes} != set(expected) or len(episodes) != count:
        raise ValueError("MuJoCo result must contain all declared attempts")
    if run["scenario_manifest_sha256"] != sha(manifest_path):
        raise ValueError("MuJoCo evaluated a different scenario manifest")
    native, summaries, sources = {}, [], {}
    for directory in map(Path, isaac_dirs):
        summary_path = directory / "evaluation_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, value in {"checkpoint_sha256": run["checkpoint_sha256"],
                           "route_bank_sha256": manifest["route_bank_sha256"]}.items():
            if summary[key].lower() != value.lower():
                raise ValueError(f"Native {key} mismatch")
        for key, value in {"num_scenarios": count // 2, "task_success_scenarios": count // 2, "duration_s": 24., "exec_horizon": 4,
                           "num_inference_steps": 10, "seed": run["protocol"]["seed"], "posture_settle_steps": 50,
                           "pace_consistent_preview": False, "deterministic_resets": True,
                           "record_staging_failures": True, "profile_transition_margin_m": .25}.items():
            if summary[key] != value:
                raise ValueError(f"Native protocol mismatch: {key}")
        metadata = json.loads((directory / "timeseries_metadata.json").read_text(encoding="utf-8"))
        if metadata["dt_s"] != manifest["dt_s"]:
            raise ValueError("Native trace dt mismatch")
        with np.load(directory / "timeseries.npz", allow_pickle=False) as trace:
            for scenario in metadata["scenarios"]:
                ident, i = condition_id(scenario), scenario["index"]
                if ident not in expected or ident in native:
                    raise ValueError(f"Unexpected/duplicate native condition {ident}")
                c = expected[ident]
                reference = banks[(c["family"], c["repeat"])].xy
                native[ident] = {
                    "position": align_to_reference(trace["positions_xy"][:, i], scenario["path_xy"], reference),
                    **{key: trace[field][:, i].copy() for key, field in
                       (("valid", "valid"), ("height", "achieved_height_m"),
                        ("required_height", "required_height_m"), ("speed", "tangent_speed_m_s"))},
                    "base_failure": bool(trace["base_failure"][i]),
                    "failure_step": int(trace["failure_step"][i]),
                }
        summaries.append(summary)
        for filename in ("evaluation_summary.json", "timeseries_metadata.json", "timeseries.npz"):
            sources[str(directory / filename)] = sha(directory / filename)
    if set(native) != set(expected):
        raise ValueError("Missing native conditions")
    result = {"scope": manifest["scope"], "checkpoint_sha256": run["checkpoint_sha256"],
              "model_sha256": run["model_sha256"], "source_hashes": sources,
              "Isaac": {"attempts": count,
                        "arrived": sum(round(s["overall_task_arrival_rate"] * (count // 2)) for s in summaries),
                        "task_success": sum(task_success_count(s) for s in summaries),
                        "full_horizon_survival": sum(round(s["overall_survival_rate"] * (count // 2)) for s in summaries),
                        "mean_episode_active_cte_rmse_m": np.mean([s["overall_active_cross_track_rmse_m"] for s in summaries]),
                        "mean_episode_active_height_mae_m": np.mean([s["overall_active_height_mae_m"] for s in summaries]),
                        "full_horizon_failures": [{"id": k, "failure_step": v["failure_step"]}
                                                  for k, v in native.items() if v["base_failure"]]},
              "MuJoCo": {"attempts": count,
                         **{k: sum(e[k] for e in episodes) for k in ("arrived", "task_success", "adherence_success", "full_horizon_survival")},
                         "mean_episode_active_cte_rmse_m": finite_mean([e.get("tracking", {}).get("cross_track_rmse_m") for e in episodes]),
                         "mean_episode_active_height_mae_m": finite_mean([e.get("tracking", {}).get("height_mae_m") for e in episodes]),
                         "mean_episode_arrival_speed_error_mps": finite_mean([e.get("task", {}).get("mean_speed_error_m_s") for e in episodes if e["arrived"]]),
                         "arrival_speed_error_n": sum(e["arrived"] and e.get("task", {}).get("mean_speed_error_m_s") is not None for e in episodes)},
              "limitations": ["One inference seed per comparison; fixed development geometries, not sealed statistical evidence",
                              f"MuJoCo batch {count} versus native two batches {count // 2}; denoising noise is not paired",
                              "Global spawn pose removed using route frames only, never fitting achieved positions",
                              "Engine contact and actuator semantics differ; no dynamics tuning or retraining",
                              "Full 24 s survival includes post-arrival standing; first task success is separate",
                              "Synchronous simulation is not wall-clock realtime validation",
                              "Native adherence counts not reconstructed from incomplete arrival-origin metadata"]}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "comparison.json", result)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(mujoco_dir / "traces.npz", allow_pickle=False) as traces:
        fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
        profiles, panels = plt.subplots(4, 2, figsize=(9, 8), constrained_layout=True)
        for row, ident in enumerate(manifest["smoke_ids"]):
            i = next(i for i, e in enumerate(episodes) if e["id"] == ident)
            c = expected[ident]; path, _, _ = path_for(c, banks[(c["family"], c["repeat"])], manifest)
            n = native[ident]; valid = traces[f"episode_{i}_valid"]
            p = traces[f"episode_{i}_position"]
            ax = axes.flat[row]
            ax.plot(path[:, 0], path[:, 1], "--", color=".25", label="Reference")
            ax.plot(*n["position"][n["valid"]].T, color="#0072B2", lw=1.2, label="Isaac / PhysX")
            ax.plot(*p[valid, :2].T, color="#E69F00", lw=1.2, label="MuJoCo")
            ax.set_title(ident.replace("_", " "), fontsize=9)
            ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
            times = (np.arange(len(valid)) + 1) * manifest["dt_s"]
            for col, nkey, mkey in ((0, "height", "position"), (1, "speed", "tangent_speed")):
                panel = panels[row, col]
                values = p[:, 2] if col == 0 else traces[f"episode_{i}_{mkey}"]
                panel.plot(times[n["valid"]], n[nkey][n["valid"]], color="#0072B2", lw=1, label="Isaac / PhysX")
                panel.plot(times[valid], values[valid], color="#E69F00", lw=1, label="MuJoCo")
                if col == 0:
                    # Height switches occur at each engine's own progress, not necessarily equal time.
                    panel.plot(times[n["valid"]], n["required_height"][n["valid"]], "--", color="#0072B2", lw=.8, label="Required (Isaac)")
                    panel.plot(times[valid], traces[f"episode_{i}_required_height"][valid], "--", color="#E69F00", lw=.8, label="Required (MuJoCo)")
                else:
                    panel.axhline(c["speed_mps"], color=".25", ls="--", label="Requested mean")
                panel.set_xlabel("Active time [s]")
                panel.set_ylabel("Base height [m]" if col == 0 else "Tangent speed [m/s]")
            panels[row, 0].set_title(ident.replace("_", " "), fontsize=9)
        for ax in [*axes.flat, *panels.flat]:
            ax.grid(alpha=.2); ax.legend(fontsize=6, loc="best")
        fig.savefig(output / "geometry_comparison.png", dpi=180); plt.close(fig)
        profiles.savefig(output / "height_speed_comparison.png", dpi=180); plt.close(profiles)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True); parser.add_argument("--mujoco-dir", required=True)
    parser.add_argument("--isaac-dirs", nargs=2, required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.mujoco_dir, args.isaac_dirs, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
