"""Replay frozen Isaac fixtures and profile the pinned FP32 WC runtime."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.diffusion_policy.runtime.contract import WC_V3_SHA256, load_runtime_contract
from scripts.diffusion_policy.runtime.controller import (
    DelayedHistory,
    Profile16Route,
    action_to_joint_target,
)
from scripts.diffusion_policy.runtime.fixtures import (
    load_fixture_bundle,
    restore_rng_state,
)


def _tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device).unsqueeze(0)


def _scalar(value: np.ndarray) -> float:
    return float(np.asarray(value).reshape(()))


def _route(record: dict[str, np.ndarray], metadata: dict[str, Any]) -> Profile16Route:
    return Profile16Route(
        path_w=record["path_w"].astype(np.float32, copy=True),
        cumulative_xy=record["path_cumulative_xy"].astype(np.float32, copy=True),
        yaws_w=record["path_yaw_w"].astype(np.float32, copy=True),
        progress_idx=int(np.asarray(record["path_progress_idx"]).item()),
        speed_start_arc=_scalar(record["speed_budget_start_arc_m"]),
        speed_target_arc=_scalar(record["speed_budget_target_arc_m"]),
        desired_mean_speed=_scalar(record["requested_mean_speed_mps"]),
    )


def _build_goal(record: dict[str, np.ndarray], metadata: dict[str, Any], contract) -> tuple[np.ndarray, int]:
    fixture = metadata["metadata"]
    env = fixture["environment"]
    route = _route(record, fixture)
    goal = route.build_goal(
        root_pos_w=record["root_pos_pre"],
        root_quat_wxyz=record["root_quat_pre"],
        step=int(np.asarray(record["step"]).item()),
        dt=float(env["policy_dt_s"]),
        goal_horizon_steps=int(fixture["goal_horizon_steps"]),
        v_req_clip=float(fixture.get("v_req_clip_mps", contract.values["v_req_clip_mps"])),
        speed_budget_max_mps=float(fixture["speed_budget_max_mps"]),
    )
    return goal, route.progress_idx


def _close(actual: np.ndarray | torch.Tensor, expected: np.ndarray, *, atol: float,
           rtol: float) -> tuple[float | None, bool]:
    value = actual.detach().cpu().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    expected = np.asarray(expected)
    if value.shape != expected.shape or not np.isfinite(value).all() or not np.isfinite(expected).all():
        return None, False
    error = float(np.max(np.abs(value - expected))) if value.size else 0.0
    return error, bool(np.allclose(value, expected, atol=atol, rtol=rtol))


def run_parity(contract, bundles: list[tuple[dict, list[dict[str, np.ndarray]]]], output: Path,
               sampler=None, sampler_name: str = "cached") -> dict:
    if (output / "parity.json").exists():
        raise FileExistsError(f"Refusing to overwrite parity report: {output / 'parity.json'}")
    device = contract.device
    sampler = sampler or contract.sampler
    result = {
        "checkpoint_sha256": contract.sha256,
        "backend": str(device),
        "dtype": "float32",
        "sampler": sampler_name,
        "goal_tolerance_atol": 1.0e-6,
        "goal_tolerance_rtol": 1.0e-5,
        "action_tolerance_atol": 1.0e-5,
        "action_tolerance_rtol": 1.0e-4,
        "target_tolerance_atol_rad": 1.0e-5,
        "target_tolerance_rtol": 1.0e-4,
        "fixtures": [],
    }
    original_cpu_rng = torch.get_rng_state().clone()
    original_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    failures = []
    try:
        for bundle_metadata, records in bundles:
            capture_meta = bundle_metadata["metadata"]
            if capture_meta.get("checkpoint_sha256", "").lower() != contract.sha256.lower():
                raise ValueError("Fixture and loaded actor SHA-256 differ.")
            if capture_meta.get("goal_representation") != "hindsight_geom_profile16":
                raise ValueError("Fixture does not contain profile16 goals.")
            env = capture_meta["environment"]
            if capture_meta.get("device") != str(device):
                raise ValueError(
                    f"Same-backend parity required; fixture used {capture_meta.get('device')}, replay uses {device}."
                )
            q_offset = torch.as_tensor(env["q_offset_rad"], dtype=torch.float32, device=device)
            clip = env["wrapper_clip_actions"]
            if clip is False:
                clip = 0.0
            elif clip is not None:
                clip = float(clip)
            one_bundle = {"context": capture_meta.get("scenario", {}).get("path_shape", "unknown"),
                          "samples": [], "max_goal_abs_error": 0.0,
                          "max_trajectory_abs_error": 0.0, "max_executable_chunk_abs_error": 0.0,
                          "max_action_abs_error_env": 0.0, "max_target_abs_error_rad": 0.0}
            for index, record in enumerate(records):
                if (bundle_metadata.get("capture_mode") == "sequential" and
                        int(np.asarray(record["selected_chunk_slot"]).item()) != 0):
                    # RNG states between replans are post-sampling states. Those
                    # rows are validated by run_sequential_replay, which carries
                    # the existing chunk instead of sampling again.
                    continue
                try:
                    computed_goal, progress = _build_goal(record, bundle_metadata, contract)
                    goal_error, goal_pass = _close(
                        computed_goal, record["current_goal_pre_step"], atol=1e-6, rtol=1e-5
                    )
                    if progress != int(np.asarray(record["path_progress_after_goal"]).item()):
                        raise AssertionError(f"progress index {progress} != recorded post-goal progress")
                    goal_hist = record["goal_history"]
                    p = _tensor(record["proprio_history"], device)
                    a = _tensor(record["action_history"], device)
                    g = _tensor(goal_hist, device)
                    restore_rng_state(record, device)
                    trajectory = sampler.predict_action_denormalized(
                        p, a, g, guidance_scale=float(capture_meta["guidance_scale"])
                    )
                    chunk = contract.policy.executable_chunk(trajectory, 4)
                    action, target = action_to_joint_target(
                        chunk[:, 0],
                        action_scale=float(env["action_scale"]),
                        q_offset=q_offset,
                        clip_actions=clip,
                    )
                    prediction_error, prediction_pass = _close(
                        trajectory[0], record["full_trajectory"], atol=1e-5, rtol=1e-4
                    )
                    chunk_error, chunk_pass = _close(
                        chunk[0], record["executable_chunk"], atol=1e-5, rtol=1e-4
                    )
                    action_error, action_pass = _close(action[0], record["env_action"], atol=1e-5, rtol=1e-4)
                    target_error, target_pass = _close(
                        target[0], record["joint_target_post_step"], atol=1e-5, rtol=1e-4
                    )
                    # Stored buffers are the exact model inputs; explicit shape/finiteness
                    # checks detect serialization or batch-axis changes before inference.
                    if p.shape != (1, 8, 30) or a.shape != (1, 8, 12) or g.shape != (1, 8, 16):
                        raise AssertionError("runtime fixture history layout changed")
                    if any(not torch.isfinite(x).all() for x in (p, a, g)):
                        raise AssertionError("runtime fixture contains nonfinite history values")
                    errors = {"goal": goal_error, "trajectory": prediction_error,
                              "executable_chunk": chunk_error,
                              "env_action": action_error, "joint_target_rad": target_error}
                    sample_pass = all((goal_pass, prediction_pass, chunk_pass, action_pass, target_pass))
                    one_bundle["samples"].append({"sample": index, "step": int(record["step"]),
                                                  "errors": errors, "pass": sample_pass})
                    for key, error in (("max_goal_abs_error", goal_error),
                                       ("max_trajectory_abs_error", prediction_error),
                                       ("max_executable_chunk_abs_error", chunk_error),
                                       ("max_action_abs_error_env", action_error),
                                       ("max_target_abs_error_rad", target_error)):
                        if error is not None:
                            one_bundle[key] = max(one_bundle[key], error)
                except Exception as exc:  # preserve each failing sample in parity.json
                    one_bundle["samples"].append({"sample": index, "step": int(record.get("step", -1)),
                                                  "pass": False, "error": f"{type(exc).__name__}: {exc}"})
            one_bundle["pass"] = bool(one_bundle["samples"]) and all(row["pass"] for row in one_bundle["samples"])
            result["fixtures"].append(one_bundle)
            if not one_bundle["pass"]:
                failures.append(one_bundle["context"])
    finally:
        torch.set_rng_state(original_cpu_rng)
        if original_cuda_rng is not None:
            torch.cuda.set_rng_state(original_cuda_rng, device)
    result["snapshot_count"] = sum(len(x["samples"]) for x in result["fixtures"])
    result["pass"] = bool(result["fixtures"]) and not failures
    output.mkdir(parents=True, exist_ok=True)
    (output / "parity.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def run_sequential_replay(contract, bundle: tuple[dict, list[dict[str, np.ndarray]]], output: Path,
                          sampler=None, sampler_name: str = "cached", history_device=None) -> dict:
    """Replay contiguous captured steps, rebuilding delayed histories and chunks."""
    bundle_metadata, records = bundle
    capture_meta = bundle_metadata["metadata"]
    if bundle_metadata.get("capture_mode") != "sequential":
        raise ValueError("Sequential replay requires a fixture captured with --runtime_fixture_sequential_steps.")
    steps = [int(np.asarray(row["step"]).item()) for row in records]
    if steps != list(range(len(records))):
        raise ValueError("Sequential fixture must contain every step from zero in order.")
    if len(records) < 8:
        raise ValueError("Sequential replay requires at least eight captured environment steps.")
    env = capture_meta["environment"]
    device = contract.device
    buffer_device = device if history_device is None else torch.device(history_device)
    sampler = sampler or contract.sampler
    route = _route(records[0], capture_meta)
    # Evaluator fixtures start at the first active policy tick, after reset
    # warmup and optional posture staging.  Their histories are therefore
    # already populated; starting this replay from zeros creates an artificial
    # mismatch in the first action chunk and every subsequent history.
    first_record = records[0]
    history = DelayedHistory(
        proprio=torch.as_tensor(first_record["proprio_history"], dtype=torch.float32,
                                device=buffer_device).unsqueeze(0).clone(),
        action=torch.as_tensor(first_record["action_history"], dtype=torch.float32,
                               device=buffer_device).unsqueeze(0).clone(),
        goal=torch.as_tensor(first_record["goal_history"], dtype=torch.float32,
                             device=buffer_device).unsqueeze(0).clone(),
        previous_action=torch.as_tensor(first_record["previous_action"], dtype=torch.float32,
                                        device=buffer_device).unsqueeze(0).clone(),
    )
    history.validate()
    q_offset = torch.as_tensor(env["q_offset_rad"], dtype=torch.float32, device=buffer_device)
    clip = env["wrapper_clip_actions"]
    clip = 0.0 if clip is False else (None if clip is None else float(clip))
    trajectory = None
    chunk = None
    rows = []
    original_cpu_rng = torch.get_rng_state().clone()
    original_cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    try:
        for index, record in enumerate(records):
            slot = index % 4
            hist_checks = []
            for name, actual, key in (
                ("proprio_history", history.proprio[0], "proprio_history"),
                ("action_history", history.action[0], "action_history"),
                ("goal_history", history.goal[0], "goal_history"),
                ("previous_action", history.previous_action[0], "previous_action"),
            ):
                err, passed = _close(actual, record[key], atol=0.0, rtol=0.0)
                hist_checks.append((name, err, passed))

            if slot == 0:
                restore_rng_state(record, device)
                trajectory = sampler.predict_action_denormalized(
                    history.proprio, history.action, history.goal,
                    guidance_scale=float(capture_meta["guidance_scale"]),
                )
                chunk = contract.policy.executable_chunk(trajectory, 4)
            if trajectory is None or chunk is None:
                raise RuntimeError("Chunk was not initialized at the first sequential step.")

            raw_action = chunk[:, slot]
            action, target = action_to_joint_target(
                raw_action, action_scale=float(env["action_scale"]), q_offset=q_offset, clip_actions=clip,
            )
            current_goal = route.build_goal(
                root_pos_w=record["root_pos_pre"], root_quat_wxyz=record["root_quat_pre"],
                step=index, dt=float(env["policy_dt_s"]),
                goal_horizon_steps=int(capture_meta["goal_horizon_steps"]),
                v_req_clip=float(capture_meta.get("v_req_clip_mps", contract.values["v_req_clip_mps"])),
                speed_budget_max_mps=float(capture_meta["speed_budget_max_mps"]),
            )
            action_error, action_pass = _close(action[0], record["env_action"], atol=1e-5, rtol=1e-4)
            target_error, target_pass = _close(
                target[0], record["joint_target_post_step"], atol=1e-5, rtol=1e-4
            )
            chunk_action_error, chunk_action_pass = _close(
                raw_action[0], record["action_from_chunk"], atol=1e-5, rtol=1e-4
            )
            trajectory_error, trajectory_pass = _close(
                trajectory[0], record["full_trajectory"], atol=1e-5, rtol=1e-4
            )
            chunk_error, chunk_pass = _close(
                chunk[0], record["executable_chunk"], atol=1e-5, rtol=1e-4
            )
            goal_error, goal_pass = _close(
                current_goal, record["current_goal_pre_step"], atol=1e-6, rtol=1e-5
            )
            progress_pass = route.progress_idx == int(np.asarray(record["path_progress_after_goal"]).item())
            hist_pass = all(item[2] for item in hist_checks)
            captured_slot = int(np.asarray(record["selected_chunk_slot"]).item())
            slot_pass = captured_slot == slot
            passed = (hist_pass and action_pass and target_pass and chunk_action_pass and
                      trajectory_pass and chunk_pass and goal_pass and progress_pass and slot_pass)
            rows.append({
                "step": index, "chunk_slot": slot, "pass": passed,
                "history_max_abs_error": max((x[1] or 0.0) for x in hist_checks),
                "action_max_abs_error": action_error,
                "joint_target_max_abs_error_rad": target_error,
                "chunk_action_max_abs_error": chunk_action_error,
                "trajectory_max_abs_error": trajectory_error,
                "chunk_max_abs_error": chunk_error,
                "goal_max_abs_error": goal_error,
                "slot_pass": slot_pass,
                "progress_pass": progress_pass,
                "history_pass": hist_pass,
            })
            history.commit_tick(
                _tensor(record["proprio_pre_step"], buffer_device), action,
                torch.as_tensor(current_goal, dtype=torch.float32, device=buffer_device).unsqueeze(0),
            )
    finally:
        torch.set_rng_state(original_cpu_rng)
        if original_cuda_rng is not None:
            torch.cuda.set_rng_state(original_cuda_rng, device)
    result = {
        "mode": "contiguous_policy_steps_from_reset",
        "history_device": str(buffer_device),
        "checkpoint_sha256": contract.sha256,
        "backend": str(device),
        "sampler": sampler_name,
        "step_count": len(rows),
        "complete_chunks": len(rows) // 4,
        "pass": bool(rows) and all(row["pass"] for row in rows),
        "steps": rows,
        "limitations": [
            "Uses recorded pre-step physical observations as inputs; does not rerun physics or a robot transport loop.",
            "Validates history shifts, replanning cadence, four chunk actions, goal updates and joint targets.",
        ],
    }
    if (output / "sequential_replay.json").exists():
        raise FileExistsError(f"Refusing to overwrite sequential replay: {output / 'sequential_replay.json'}")
    (output / "sequential_replay.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"median_ms": float(np.median(array)), "p95_ms": float(np.quantile(array, .95)),
            "p99_ms": float(np.quantile(array, .99)), "max_ms": float(np.max(array)),
            "deadline_80ms_exceedance_fraction": float(np.mean(array > 80.0))}


def _profile_context(contract, bundle_metadata: dict, record: dict[str, np.ndarray], *, warmup: int,
                     samples: int, context: str, sampler, sampler_name: str) -> tuple[list[dict], dict]:
    device = contract.device
    capture_meta = bundle_metadata["metadata"]
    env = capture_meta["environment"]
    q_offset = torch.as_tensor(env["q_offset_rad"], dtype=torch.float32, device=device)
    clip = env["wrapper_clip_actions"]
    if clip is False:
        clip = 0.0
    elif clip is not None:
        clip = float(clip)
    proprio = _tensor(record["proprio_history"], device)
    action_history = _tensor(record["action_history"], device)
    goal_history = _tensor(record["goal_history"], device)
    pre_proprio = _tensor(record["proprio_pre_step"], device)
    env_action_prev = _tensor(record["env_action"], device)
    current_goal = _tensor(record["current_goal_pre_step"], device)
    route = _route(record, capture_meta)

    def goal_call():
        route.progress_idx = int(np.asarray(record["path_progress_idx"]).item())
        return route.build_goal(
            root_pos_w=record["root_pos_pre"], root_quat_wxyz=record["root_quat_pre"],
            step=int(np.asarray(record["step"]).item()), dt=float(env["policy_dt_s"]),
            goal_horizon_steps=int(capture_meta["goal_horizon_steps"]),
            v_req_clip=float(capture_meta.get("v_req_clip_mps", contract.values["v_req_clip_mps"])),
            speed_budget_max_mps=float(capture_meta["speed_budget_max_mps"]),
        )

    def sampler_call():
        return sampler.predict_action_denormalized(
            proprio, action_history, goal_history, guidance_scale=float(capture_meta["guidance_scale"])
        )

    def mapping_call(trajectory):
        chunk = contract.policy.executable_chunk(trajectory, 4)
        return action_to_joint_target(chunk[:, 0], action_scale=float(env["action_scale"]),
                                      q_offset=q_offset, clip_actions=clip)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            mapping_call(sampler_call())
        sync()
        # Repeated-input software microbenchmark. The evolving history below is
        # timed separately; the sampler inputs intentionally stay fixed here.
        goal_ms = []
        history_ms = []
        sampler_ms = []
        map_ms = []
        full_ms = []
        cuda_events = []
        history = DelayedHistory(
            proprio.clone(), action_history.clone(), goal_history.clone(),
            _tensor(record["previous_action"], device).reshape(1, 12),
        )
        for _ in range(samples):
            t0 = time.perf_counter()
            goal_call()
            goal_ms.append((time.perf_counter() - t0) * 1000.0)

            sync()
            t0 = time.perf_counter()
            history.commit_tick(pre_proprio, env_action_prev, current_goal)
            sync()
            history_ms.append((time.perf_counter() - t0) * 1000.0)

            sync()
            t0 = time.perf_counter()
            if device.type == "cuda":
                sampler_start = torch.cuda.Event(enable_timing=True)
                sampler_end = torch.cuda.Event(enable_timing=True)
                sampler_start.record()
            trajectory = sampler_call()
            if device.type == "cuda":
                sampler_end.record()
                sampler_end.synchronize()
                cuda_events.append(float(sampler_start.elapsed_time(sampler_end)))
            sync()
            sampler_ms.append((time.perf_counter() - t0) * 1000.0)

            sync()
            t0 = time.perf_counter()
            mapping_call(trajectory)
            sync()
            map_ms.append((time.perf_counter() - t0) * 1000.0)

            sync()
            t0 = time.perf_counter()
            fresh_goal = goal_call()
            device_goal = torch.as_tensor(fresh_goal, dtype=torch.float32, device=device).unsqueeze(0)
            full_trajectory = sampler_call()
            mapped, _ = mapping_call(full_trajectory)
            history.commit_tick(pre_proprio, mapped, device_goal)
            sync()
            full_ms.append((time.perf_counter() - t0) * 1000.0)
    rows = []
    for index in range(samples):
        rows.append({"context": context, "sample": index, "goal_prepare_wall_ms": goal_ms[index],
                     "history_update_wall_ms": history_ms[index], "sampler_host_wall_ms": sampler_ms[index],
                     "action_map_wall_ms": map_ms[index] if map_ms else math.nan,
                     "sampler_cuda_event_ms": cuda_events[index] if cuda_events else math.nan,
                     "fixed_input_pipeline_host_wall_ms": full_ms[index]})
    summary = {"context": context, "sample_count": samples, "warmup_count": warmup,
               "goal_prepare": _percentiles(goal_ms), "history_update": _percentiles(history_ms),
               "sampler_host_wall": _percentiles(sampler_ms),
               "action_map_wall": _percentiles(map_ms) if map_ms else None,
               "sampler_cuda_event": _percentiles(cuda_events) if cuda_events else None,
               "fixed_input_pipeline_host_wall": _percentiles(full_ms),
               "sampler": sampler_name,
               "notes": "Fixed recorded model inputs; not an evolving controller or end-to-end simulation. "
                        "Synchronized wall and CUDA-event measurements are separate and non-additive."}
    return rows, summary


def run_benchmark(contract, bundles: list[tuple[dict, list[dict[str, np.ndarray]]]],
                  output: Path, *, warmup: int, samples: int, sampler=None,
                  sampler_name: str = "cached") -> dict:
    if warmup < 1 or samples < 20:
        raise ValueError("Use at least one warmup and 20 timed samples per real context.")
    protected = ("latency_samples.csv", "latency_summary.json", "latency_ecdf.png")
    existing = [output / name for name in protected if (output / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite benchmark output: {existing[0]}")
    sampler = sampler or contract.sampler
    all_rows, summaries = [], []
    for bundle_metadata, records in bundles:
        # Pick a middle snapshot so the context includes evolved state, not
        # only the reset-transient first inference.
        index = len(records) // 2
        record = records[index]
        context = str(bundle_metadata["metadata"].get("scenario", {}).get("path_shape", "unknown"))
        route_len = float(bundle_metadata["metadata"].get("scenario", {}).get("speed", -1.0))
        context = f"{context}_v{route_len:g}_sample{index:03d}"
        rows, summary = _profile_context(contract, bundle_metadata, record, warmup=warmup,
                                         samples=samples, context=context, sampler=sampler,
                                         sampler_name=sampler_name)
        all_rows.extend(rows)
        summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "latency_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "checkpoint_sha256": contract.sha256,
        "device": str(contract.device),
        "device_name": torch.cuda.get_device_name(contract.device) if contract.device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "dtype": "float32",
        "execution": f"{sampler_name} sampler, FP32, batch=1, no autocast/compile/export",
        "sampler": sampler_name,
        "nominal_replan_period_ms": contract.values["exec_horizon"] * contract.values["dt_s"] * 1000.0,
        "contexts": summaries,
        "limitations": ["p99 from 200 repeats is descriptive, not a tail-latency guarantee.",
                        "No simulator, viewer, actuator thread, transport, or hardware is timed.",
                        "CUDA event and synchronized wall-clock distributions are non-additive."],
    }
    (output / "latency_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(7.2, 4.3), constrained_layout=True)
    for context in sorted({row["context"] for row in all_rows}):
        values = np.sort(np.asarray([row["fixed_input_pipeline_host_wall_ms"] for row in all_rows
                                     if row["context"] == context], dtype=np.float64))
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, linewidth=1.8, label=context)
    ax.axvline(80.0, color="#aa3333", linestyle="--", linewidth=1.4, label="80 ms replan period")
    ax.set_xlabel("Fixed-input software-path wall time [ms]")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(output / "latency_ecdf.png", dpi=180)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True, nargs="+", help="Directories written by evaluator oracle capture.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-sha256", default=WC_V3_SHA256,
                        help="Pinned candidate SHA by default; pass 'none' only after review.")
    parser.add_argument("--sampler", choices=("native", "cached", "both"), default="both",
                        help="Compare original policy inference, cached runtime, or both.")
    parser.add_argument("--sampler-order", choices=("native,cached", "cached,native"),
                        default="native,cached", help="Serial order used when --sampler both.")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this Python environment.")
    expected_sha = None if args.expected_sha256.lower() == "none" else args.expected_sha256
    contract = load_runtime_contract(args.checkpoint, device, expected_sha256=expected_sha)
    bundles = [load_fixture_bundle(item) for item in args.fixture]
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite benchmark results: {output}")
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "native": contract.policy,
        "cached": contract.sampler,
    }
    selected = args.sampler_order.split(",") if args.sampler == "both" else [args.sampler]
    summaries = {}
    for name in selected:
        variant_output = output / name
        variant_output.mkdir()
        sampler = variants[name]
        if not args.skip_parity:
            parity = run_parity(contract, bundles, variant_output, sampler=sampler, sampler_name=name)
            if not parity["pass"]:
                print(f"FAIL: {name} parity; details: {variant_output / 'parity.json'}")
                return 2
            print(f"PASS: {name} same-backend parity over {parity['snapshot_count']} snapshots.")
        for bundle_metadata, records in bundles:
            if bundle_metadata.get("capture_mode") == "sequential":
                result = run_sequential_replay(
                    contract, (bundle_metadata, records), variant_output,
                    sampler=sampler, sampler_name=name,
                )
                print(f"{'PASS' if result['pass'] else 'FAIL'}: {name} sequential replay "
                      f"{result['step_count']} steps / {result['complete_chunks']} chunks.")
                if not result["pass"]:
                    return 3
        summaries[name] = run_benchmark(
            contract, bundles, variant_output, warmup=args.warmup, samples=args.samples,
            sampler=sampler, sampler_name=name,
        )
        print(f"Saved {name} benchmark in {variant_output}")
    if len(summaries) == 2:
        by_name = {name: {row["context"]: row["fixed_input_pipeline_host_wall"]
                          for row in summary["contexts"]} for name, summary in summaries.items()}
        comparison = {"measurement_order": selected, "contexts": {}}
        for context in sorted(set(by_name["native"]) & set(by_name["cached"])):
            native = by_name["native"][context]
            cached = by_name["cached"][context]
            comparison["contexts"][context] = {
                "native_median_ms": native["median_ms"], "cached_median_ms": cached["median_ms"],
                "median_relative_reduction": 1.0 - cached["median_ms"] / native["median_ms"],
                "native_p95_ms": native["p95_ms"], "cached_p95_ms": cached["p95_ms"],
                "notes": "Same fixture inputs; variants measured serially, not interleaved. Repeat in reverse order for a controlled A/B.",
            }
        (output / "sampler_comparison.json").write_text(
            json.dumps(comparison, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(f"Benchmark outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
