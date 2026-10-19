"""Gate CUDA graph inference against frozen oracle fixtures, then benchmark serially."""
import argparse
import json
from pathlib import Path
import torch
from .contract import load_runtime_contract
from .cuda_graph import CudaGraphDPPOSampler
from .fixtures import load_fixture_bundle
from .benchmark import run_parity, run_sequential_replay, run_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--fixture", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--order", choices=("cached,graph", "graph,cached"), default="cached,graph")
    parser.add_argument("--samples", type=int, default=200); parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args(); output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    bundles = [load_fixture_bundle(p) for p in args.fixture]
    # Separate instances: graph causality metadata must not modify the reference.
    contracts = {name: load_runtime_contract(args.checkpoint, args.device) for name in ("cached", "graph")}
    samplers = {"cached": contracts["cached"].sampler,
                "graph": CudaGraphDPPOSampler(contracts["graph"].policy)}
    results = {}
    for name in args.order.split(","):
        destination = output / name; destination.mkdir()
        contract, sampler = contracts[name], samplers[name]
        parity = run_parity(contract, bundles, destination, sampler=sampler, sampler_name=name)
        if not parity["pass"]:
            raise RuntimeError(f"{name}: original oracle parity failed; do not deploy")
        for bundle in bundles:
            if bundle[0].get("capture_mode") == "sequential":
                replay = run_sequential_replay(contract, bundle, destination, sampler=sampler, sampler_name=name)
                if not replay["pass"]:
                    raise RuntimeError(f"{name}: sequential replay failed; do not deploy")
        results[name] = run_benchmark(contract, bundles, destination, sampler=sampler,
                                     sampler_name=name, samples=args.samples, warmup=args.warmup)
        print(f"{name}: parity and timing complete", flush=True)
    (output / "optimization.json").write_text(json.dumps({"order": args.order, "results": results,
        "graph_startup_s": samplers["graph"].startup_s, "torch_threads": torch.get_num_threads(),
        "scope": "Fixed-input batch1 software pipeline; not sensor/actuator transport or robot timing"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
