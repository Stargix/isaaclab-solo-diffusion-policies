"""Measure where the pinned WC-v3 inference path spends time.

This is a synchronized software microbenchmark, not an end-to-end robot timing.
It reports CPU route preparation, CPU->GPU/GPU->CPU copies, normalization,
transformer forwards, the whole reverse chain, and the chain work outside the
transformer. CUDA event and host wall measurements are intentionally separate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from scripts.diffusion_policy.runtime.benchmark import _build_goal, _route, _tensor
from scripts.diffusion_policy.runtime.contract import WC_V3_SHA256, load_runtime_contract
from scripts.diffusion_policy.runtime.fixtures import load_fixture_bundle


def _stats(values: list[float]) -> dict[str, float]:
    values = [float(value) for value in values]
    return {
        "median_ms": statistics.median(values),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
    }


def _copy_group_profile(arrays: dict[str, np.ndarray], device: torch.device, samples: int) -> dict:
    wall, device_ms = [], []
    for _ in range(samples):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        results = {
            name: torch.as_tensor(array, dtype=torch.float32, device=device)
            for name, array in arrays.items()
        }
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - start) * 1000.0)
        device_ms.append(float(begin.elapsed_time(end)))
        del results
    return {
        "bytes": sum(int(array.nbytes) for array in arrays.values()),
        "copies_per_call": len(arrays),
        "synchronized_host_wall": _stats(wall),
        "cuda_event": _stats(device_ms),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    if args.samples < 20 or args.warmup < 1:
        parser.error("Use at least 20 measured samples and one warm-up.")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This breakdown uses CUDA events; run it on cuda:0.")
    contract = load_runtime_contract(args.checkpoint, device, expected_sha256=WC_V3_SHA256)
    metadata, records = load_fixture_bundle(args.fixture)
    record = records[len(records) // 2]
    capture = metadata["metadata"]
    env_meta = capture["environment"]
    route = _route(record, capture)

    def build_goal():
        route.progress_idx = int(np.asarray(record["path_progress_idx"]).item())
        return _build_goal(record, metadata, contract)[0]

    goal_cpu_ms = []
    for _ in range(args.samples):
        start = time.perf_counter()
        build_goal()
        goal_cpu_ms.append((time.perf_counter() - start) * 1000.0)

    arrays = {
        "proprio_history": np.asarray(record["proprio_history"], dtype=np.float32),
        "action_history": np.asarray(record["action_history"], dtype=np.float32),
        "goal_history": np.asarray(record["goal_history"], dtype=np.float32),
    }
    transfers = _copy_group_profile(arrays, device, args.samples)
    transfers["input_bytes_by_history"] = {name: int(array.nbytes) for name, array in arrays.items()}

    proprio = _tensor(arrays["proprio_history"], device)
    action = _tensor(arrays["action_history"], device)
    goal = _tensor(arrays["goal_history"], device)
    policy = contract.policy
    sampler = contract.sampler
    original_condition = policy._normalized_condition
    original_model_output = policy._model_output
    measurements = []
    active_sample = None

    def timed_condition(*values):
        if active_sample is None:
            return original_condition(*values)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_condition(*values)
        end.record()
        active_sample["conditioning_events"].append((begin, end))
        return result

    def timed_model(sample, timestep, proprio_n, action_n, goal_n, *, trainable):
        if active_sample is None:
            return original_model_output(
                sample, timestep, proprio_n, action_n, goal_n, trainable=trainable
            )
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        result = original_model_output(
            sample, timestep, proprio_n, action_n, goal_n, trainable=trainable
        )
        end.record()
        active_sample["network_events"].append(("actor" if trainable else "base", begin, end))
        return result

    policy._normalized_condition = timed_condition
    policy._model_output = timed_model
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                sampler.predict_action_denormalized(proprio, action, goal)
            torch.cuda.synchronize(device)
            for _ in range(args.samples):
                current = {"conditioning_events": [], "network_events": []}
                active_sample = current
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                host_start = time.perf_counter()
                begin.record()
                sampler.predict_action_denormalized(proprio, action, goal)
                end.record()
                end.synchronize()
                current["sampler_cuda_ms"] = float(begin.elapsed_time(end))
                current["sampler_wall_ms"] = (time.perf_counter() - host_start) * 1000.0
                current["conditioning_cuda_ms"] = sum(
                    float(start.elapsed_time(stop)) for start, stop in current["conditioning_events"]
                )
                network_ms = {"base": 0.0, "actor": 0.0}
                for kind, start, stop in current["network_events"]:
                    network_ms[kind] += float(start.elapsed_time(stop))
                current["network_base_cuda_ms"] = network_ms["base"]
                current["network_actor_cuda_ms"] = network_ms["actor"]
                current["network_cuda_ms"] = sum(network_ms.values())
                current["chain_other_cuda_ms"] = max(
                    0.0, current["sampler_cuda_ms"] - current["conditioning_cuda_ms"] - current["network_cuda_ms"]
                )
                measurements.append(current)
    finally:
        policy._normalized_condition = original_condition
        policy._model_output = original_model_output

    target = contract.policy.executable_chunk(
        sampler.predict_action_denormalized(proprio, action, goal), 4
    )[:, 0]
    torch.cuda.synchronize(device)
    output_cpu_ms = []
    for _ in range(args.samples):
        start = time.perf_counter()
        target.detach().to("cpu")
        output_cpu_ms.append((time.perf_counter() - start) * 1000.0)

    component_names = (
        "sampler_cuda_ms", "sampler_wall_ms", "conditioning_cuda_ms",
        "network_base_cuda_ms", "network_actor_cuda_ms", "network_cuda_ms", "chain_other_cuda_ms",
    )
    components = {name: _stats([sample[name] for sample in measurements]) for name in component_names}
    median_chain = components["sampler_cuda_ms"]["median_ms"]
    components["median_fraction_of_reverse_chain"] = {
        name: components[name]["median_ms"] / median_chain if median_chain else 0.0
        for name in ("conditioning_cuda_ms", "network_base_cuda_ms", "network_actor_cuda_ms", "chain_other_cuda_ms")
    }
    output = {
        "checkpoint_sha256": contract.sha256,
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "fixture": capture.get("scenario", {}),
        "batch_size": 1,
        "inference_steps": contract.values["inference_steps"],
        "warmup_samples": args.warmup,
        "measured_samples": args.samples,
        "route_goal_cpu_preparation": _stats(goal_cpu_ms),
        "host_to_device_history_copies": transfers,
        "reverse_chain_and_network": components,
        "joint_action_device_to_host_copy": _stats(output_cpu_ms),
        "interpretation": {
            "network_base": "First five frozen/base denoiser transformer forwards, CUDA time.",
            "network_actor": "Last five fine-tuned actor transformer forwards, CUDA time.",
            "chain_other": "Whole sampler minus conditioning and transformer CUDA time; includes DDPM updates, random draws, and output denormalization.",
            "sampler_wall": "Synchronized fixed-input software sampler call including host dispatch; does not include simulator, sensor transport, or actuator publication.",
        "transfers": "Three NumPy/CPU-to-CUDA history copies in one grouped call and a separate CUDA-to-CPU 12-joint action copy. A physical robot transport stack is not measured.",
            "percentages": "Fractions of median sampler CUDA-event time; not additive with wall time or CPU route preparation.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
