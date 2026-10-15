"""Aggregate all declared robustness attempts and evolving timing, not trainings."""
import argparse
import json
from pathlib import Path
import numpy as np
from .scenarios import load_manifest
from .compare import compare, finite_mean
from .perturbations import PROTOCOLS
from .reporting import save_json
from .convert import sha


def analyze(root, manifest_path, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, _ = load_manifest(manifest_path)
    expected = {c["id"] for c in manifest["conditions"]}
    results = {}; sources = {}
    for variant in PROTOCOLS:
        mj = root/f"deployment_mujoco_{variant}_20261014"
        native = [root/f"deployment_isaac_{part}_{variant}_20261014" for part in ("constants", "transitions")]
        # Existing validator checks actor, bank, semantics and frame alignment.
        compare(manifest_path, mj, native, output/variant)
        rows = {"MuJoCo": [], "Isaac": []}
        for r in json.loads((mj/"episodes.json").read_text(encoding="utf-8")):
            rows["MuJoCo"].append(dict(id=r["id"], speed=r["speed_mps"], arrived=r["arrived"],
                success=r["task_success"], adherence=r["adherence_success"], survival=r["full_horizon_survival"],
                cte=r.get("tracking", {}).get("cross_track_rmse_m"), height=r.get("tracking", {}).get("height_mae_m")))
        for d in native:
            protocol = json.loads((d/"deployment_protocol.json").read_text(encoding="utf-8"))
            if protocol["sampler"] != "graph" or protocol["physical"]["protocol"]["name"] != variant:
                raise ValueError("Native deployment protocol mismatch")
            for r in json.loads((d/"episode_metrics.json").read_text(encoding="utf-8")):
                profile = r["transition_direction"] or ("walk" if np.isclose(r["requested_height"], .2932) else "crouch")
                cte = r["active_cross_track_rmse_m"]
                rows["Isaac"].append(dict(id=f"{r['path_shape']}_{r['requested_speed']:g}_{profile}", speed=r["requested_speed"],
                    arrived=r["task_arrived"], success=r["task_success"],
                    adherence=bool(r["task_success"] and cte is not None and cte <= .1),
                    survival=r["survived"], cte=cte, height=r["active_height_mae_m"]))
            for name in ("episode_metrics.json", "deployment_protocol.json"):
                sources[str(d/name)] = sha(d/name)
        run = json.loads((mj/"run_manifest.json").read_text(encoding="utf-8"))
        if run["sampler"] != "graph" or run["perturbation"] != variant:
            raise ValueError("MuJoCo deployment protocol mismatch")
        results[variant] = {}
        for engine, attempts in rows.items():
            if len(attempts) != 32 or {r["id"] for r in attempts} != expected:
                raise ValueError("Missing/duplicated robustness attempts")
            def summary(selected):
                return dict(n=len(selected), **{k:sum(bool(r[k]) for r in selected) for k in ("arrived", "success", "adherence", "survival")},
                            mean_cte_m=finite_mean([r["cte"] for r in selected]), mean_height_mae_m=finite_mean([r["height"] for r in selected]))
            results[variant][engine] = dict(all=summary(attempts),
                by_speed={str(v):summary([r for r in attempts if r["speed"] == v]) for v in (.35, .65)}, attempts=attempts)
    timing = {}
    for label, folder in (("standalone_enforced", "live_standalone_20261014"),
                          ("standalone_profile", "live_standalone_profile_20261014"),
                          ("native_load_profile", "live_native_load_profile_full_20261014")):
        path = root/folder/"live.json"
        timing[label] = json.loads(path.read_text(encoding="utf-8")); sources[str(path)] = sha(path)
        # The release-to-end value includes schedule backlog, not just work.
        rows = [r for r in timing[label]["records"] if not r["rejected"]]
        def percentiles(values):
            return dict(median_ms=float(np.median(values)), p99_ms=float(np.percentile(values,99)), max_ms=float(max(values))) if values else None
        timing[label]["actual_tick_work"] = percentiles([r.get("tick_work_ms", r["total_work_ms"]-r["release_lateness_ms"]) for r in rows])
        if label == "native_load_profile":
            loaded = [r for r in rows if r.get("load_active", False)]
            timing[label]["verified_load_slice"] = dict(n=len(loaded),
                sensor_to_target=percentiles([r["sensor_to_target_ms"] for r in loaded]),
                release_to_tick_end=percentiles([r["total_work_ms"] for r in loaded]),
                deadline_20ms_misses=sum(r["tick_deadline_missed"] for r in loaded))
    faults = {}
    for name in ("nan_target", "stale_state", "delay", "target_limit"):
        path = root/f"live_fault_{name}_20261014"/"live.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        expected_reason = {"nan_target":"invalid_target", "stale_state":"stale_state",
                           "delay":"stale_state", "target_limit":"target_outside_joint_limits"}[name]
        if result["supervision"] != "enforce" or result["failure"] != f"watchdog:{expected_reason}" or result["commands_sent"] != 0:
            raise ValueError(f"Injected fault was not blocked before dispatch: {name}")
        faults[name] = {k:result[k] for k in ("failure", "commands_sent", "fault_injection", "first_supervisor_fault")}
        sources[str(path)] = sha(path)
    for label, folder in (("host_cpu_standalone", "live_host_cpu_standalone_profile_20261014"),
                          ("host_cpu_native_load", "live_host_cpu_native_load_profile_20261014")):
        path=root/folder/"live.json"
        timing[label]=json.loads(path.read_text(encoding="utf-8")); sources[str(path)]=sha(path)
    # Physics is driven by simulated steps in these tests, not wall time.
    # Check that changing host buffer placement preserves this live trajectory.
    reference = {r["tick"]:r["position"] for r in timing["standalone_profile"]["records"] if not r["rejected"]}
    motion_errors = {}
    for label in ("host_cpu_standalone", "host_cpu_native_load"):
        selected = [r for r in timing[label]["records"] if not r["rejected"] and r["tick"] in reference]
        motion_errors[label] = dict(ticks=len(selected), max_position_abs_error_m=float(max(
            np.max(np.abs(np.asarray(r["position"])-reference[r["tick"]])) for r in selected)))
    save_json(output/"deployment_results.json", dict(robustness=results, timing=timing, faults=faults, source_hashes=sources,
        host_buffer_motion_check=motion_errors,
        scope="Fixed development robustness bank and simulator-only software supervision, not hardware certification"))
    plot_timing(timing, output)
    print(json.dumps({v:{e:r["all"] for e,r in engines.items()} for v,engines in results.items()}, indent=2))


def plot_timing(timing, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), constrained_layout=True)
    for key, color, label, style in (("standalone_profile", "#1F4E79", "GPU / standalone", "-"),
        ("native_load_profile", "#D97925", "GPU / PhysX", "-"),
        ("host_cpu_standalone", "#1F4E79", "Host buffers / standalone", "--"),
        ("host_cpu_native_load", "#D97925", "Host buffers / PhysX (580 ticks)", "--")):
        rows = [r for r in timing[key]["records"] if not r["rejected"]]
        if key == "native_load_profile":
            rows = [r for r in rows if r.get("load_active", False)]
        for ax, metric in zip(axes, ("sensor_to_target_ms", "total_work_ms")):
            values = np.sort([r[metric] for r in rows])
            ax.plot(values, np.arange(1,len(values)+1)/len(values), color=color, label=label, linestyle=style)
            ax.set_xscale("log"); ax.grid(alpha=.15); ax.set_ylabel("Empirical cumulative fraction")
            ax.set_xticks([2,5,10,20,40,80], labels=["2","5","10","20","40","80"])
            ax.minorticks_off(); ax.set_xlim(1.1,90)
    axes[0].set_xlabel("Sensor-to-CPU-target [ms]"); axes[1].set_xlabel("Release-to-tick-end [ms]")
    axes[0].axvline(20, color=".45", linestyle=":"); axes[1].axvline(20, color=".45", linestyle=":")
    axes[0].legend(fontsize=8)
    fig.suptitle("Local RTX laptop; neural inference uses CUDA in every curve", fontsize=10)
    fig.savefig(output/"live_timing.png", dpi=180); fig.savefig(output/"live_timing.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("scratch/wc_deployment"))
    p.add_argument("--manifest", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); analyze(a.root, a.manifest, a.output_dir)


if __name__ == "__main__":
    main()
