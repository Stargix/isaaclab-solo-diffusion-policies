"""Frozen WC-v3 controller in MuJoCo, synchronous simulated time, no training."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from scripts.diffusion_policy.runtime.contract import load_runtime_contract
from scripts.diffusion_policy.runtime.controller import DelayedHistory, ChunkExecutor, Profile16Route, action_to_joint_target
from scripts.diffusion_policy.train.conditioning.goal_builder import (
    advance_path_progress, build_geometric_height_profile_goal_from_path,
)
from scripts.diffusion_policy.evaluation.metrics import project_trajectory_to_polyline
from .backend import MujocoBackend
from .convert import OFFSET, sha
from .scenarios import load_manifest, path_for
from .reporting import save_json, episode_metrics, summarize, plots


def run_batch(conditions, manifest, banks, contract, args, output):
    batch = len(conditions); device = contract.device
    backends = [MujocoBackend(args.model, args.dependency_dir, validation_path=args.validation) for _ in conditions]
    from .perturbations import configure_mujoco, impulse_mujoco
    perturbation = getattr(args, "perturbation", "nominal")
    audits = [configure_mujoco(b, perturbation) for b in backends]
    save_json(output / "physical_perturbation.json", audits)
    prepared = [path_for(c, banks[(c["family"], c["repeat"])], manifest) for c in conditions]
    histories = DelayedHistory.zeros(batch, device=device)
    sampler = contract.sampler
    if getattr(args, "sampler", "cached") == "graph":
        from scripts.diffusion_policy.runtime.cuda_graph import CudaGraphDPPOSampler
        sampler = CudaGraphDPPOSampler(contract.policy, batch_size=batch)
    executor = ChunkExecutor(sampler)
    offsets = torch.tensor(OFFSET, dtype=torch.float32, device=device)
    failures = [None] * batch; failure_times = [None] * batch
    last_states = [b.read_state() for b in backends]
    staging_progress = [0] * batch
    torch.manual_seed(manifest["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(manifest["seed"])

    def pre_step_goals(step=None, routes=None):
        states = [last_states[i] if failures[i] is not None else b.read_state() for i, b in enumerate(backends)]
        goals = []
        for i, state in enumerate(states):
            path, arc, yaw = prepared[i]
            if routes is None:
                staging_progress[i] = advance_path_progress(path, state.position, staging_progress[i])
                goal = build_geometric_height_profile_goal_from_path(
                    path, arc, yaw, state.position, state.quaternion, goal_horizon_steps=100,
                    dt=.02, speed=0., start_idx=staging_progress[i], v_avg_clip=contract.values["v_req_clip_mps"])
            else:
                goal = routes[i].build_goal(root_pos_w=state.position, root_quat_wxyz=state.quaternion,
                                           step=step, v_req_clip=contract.values["v_req_clip_mps"])
            goals.append(goal)
        return states, torch.tensor(np.stack([s.proprio() for s in states]), device=device), torch.tensor(np.stack(goals), device=device)

    def execute(actions, phase):
        actual, targets = action_to_joint_target(actions, action_scale=.25, q_offset=offsets, clip_actions=None)
        targets_cpu = targets.cpu().numpy()
        result = []
        for i, b in enumerate(backends):
            if failures[i] is not None:
                result.append((last_states[i], None)); continue
            state, diagnostics = b.step_policy_tick(targets_cpu[i])
            if diagnostics.failure:
                failures[i] = f"{phase}:{diagnostics.failure}"; failure_times[i] = state.time_s
                # A failed numerical state must not poison unrelated batched histories.
                state = last_states[i]
            else:
                last_states[i] = state
            result.append((state, diagnostics))
        return actual, result

    for _ in range(manifest["warmup_steps"]):
        _, proprio, goals = pre_step_goals()
        actions, _ = execute(torch.zeros(batch, 12, device=device), "warmup")
        histories.commit_tick(proprio, actions, goals)
    for _ in range(manifest["posture_settle_steps"]):
        action, _, _ = executor.action(histories.proprio, histories.action, histories.goal)
        _, proprio, goals = pre_step_goals()
        actual, _ = execute(action, "staging")
        histories.commit_tick(proprio, actual, goals)
    executor.reset()
    starts = [s.position.copy() for s in last_states]
    start_times = [b.data.time for b in backends]
    routes = []
    for c, (path, arc, yaw), start in zip(conditions, prepared, starts):
        start_arc = float(project_trajectory_to_polyline(start[None, :2], path[:, :2])[0][0])
        routes.append(Profile16Route(path, arc, yaw, 0, start_arc,
                                     min(start_arc + c["speed_mps"] * manifest["duration_s"], float(arc[-1])),
                                     c["speed_mps"]))
    steps = round(manifest["duration_s"] / manifest["dt_s"])
    traces = [{"position": np.zeros((steps, 3)), "yaw": np.zeros(steps), "valid": np.zeros(steps, bool),
               "required_height": np.zeros(steps), "tangent_speed": np.zeros(steps),
               "torque_max": np.zeros(steps), "saturated_fraction": np.zeros(steps),
               "base_force": np.zeros(steps), "conditioning_speed": np.zeros(steps)} for _ in conditions]
    started = time.perf_counter()
    for step in range(steps):
        if all(f is not None for f in failures):
            break
        action, _, _ = executor.action(histories.proprio, histories.action, histories.goal)
        _, proprio, goals = pre_step_goals(step, routes)
        for i, backend in enumerate(backends):
            if failures[i] is None:
                impulse_mujoco(backend, perturbation, step)
        actual, results = execute(action, "active")
        histories.commit_tick(proprio, actual, goals)
        goal_cpu = goals.cpu().numpy()
        for i, (state, diagnostics) in enumerate(results):
            path, arc, yaw = prepared[i]; trace = traces[i]
            index = advance_path_progress(path, state.position, routes[i].progress_idx)
            tangent_index = min(index + 1, len(path) - 1)
            tangent = path[tangent_index, :2] - path[max(0, tangent_index - 1), :2]
            norm = np.linalg.norm(tangent); tangent = tangent / norm if norm > 1e-6 else np.zeros(2)
            trace["position"][step] = state.position
            trace["yaw"][step] = Rotation.from_quat(state.quaternion, scalar_first=True).as_euler("xyz")[2]
            trace["valid"][step] = failures[i] is None
            trace["required_height"][step] = path[index, 2]
            trace["tangent_speed"][step] = state.linear_velocity_world[:2] @ tangent
            trace["conditioning_speed"][step] = goal_cpu[i, -1]
            if diagnostics:
                trace["torque_max"][step] = diagnostics.torque_max_nm
                trace["saturated_fraction"][step] = diagnostics.saturated_fraction
                trace["base_force"][step] = diagnostics.base_force_max_n
        if step % 200 == 0:
            print(f"MuJoCo tick {step}/{steps}, alive {sum(f is None for f in failures)}/{batch}", flush=True)
    episodes = []
    for i, c in enumerate(conditions):
        failure_time = failure_times[i] - start_times[i] if failure_times[i] is not None and failures[i].startswith("active:") else None
        episodes.append(episode_metrics(c, prepared[i][0], prepared[i][2], traces[i], starts[i],
                                        failures[i], failure_time, manifest, contract.checkpoint.get("dppo_task_config", {})))
    return episodes, traces, [p[0] for p in prepared], time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--validation", required=True); parser.add_argument("--manifest", required=True)
    parser.add_argument("--dependency-dir"); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0"); parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sampler", choices=("cached", "graph"), default="cached")
    parser.add_argument("--perturbation", choices=("nominal", "mass_friction", "lateral_impulse"), default="nominal")
    parser.add_argument("--condition-ids", nargs="+", help="Explicit declared subset; written in run provenance")
    args = parser.parse_args(); output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    manifest, banks = load_manifest(args.manifest)
    if args.seed is not None:
        if args.seed not in manifest.get("evaluation_seeds", [manifest["seed"]]):
            raise ValueError("Seed was not declared in the frozen manifest")
        manifest = dict(manifest, seed=args.seed)
    contract = load_runtime_contract(args.checkpoint, args.device)
    conditions = [c for c in manifest["conditions"] if not args.smoke or c["id"] in manifest["smoke_ids"]]
    if args.condition_ids:
        missing = set(args.condition_ids) - {c["id"] for c in conditions}
        if missing:
            raise ValueError(f"Unknown declared condition IDs: {missing}")
        conditions = [c for c in conditions if c["id"] in args.condition_ids]
    output.mkdir(parents=True, exist_ok=True)
    import subprocess
    save_json(output / "run_manifest.json", {"checkpoint_sha256": contract.sha256, "model_sha256": sha(args.model),
              "validation_sha256": sha(args.validation), "scenario_manifest_sha256": sha(args.manifest),
              "runtime_contract": contract.values, "conditions": conditions, "protocol": manifest,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
              "source_hashes": {str(p): sha(p) for folder in (Path(__file__).parent, Path(__file__).parent.parent / "runtime")
                                for p in folder.rglob("*.py")},
              "precision": "FP32", "simulated_time_not_wall_time": True,
              "sampler": args.sampler, "perturbation": args.perturbation,
              "limitations": ["Development transfer bank, not sealed generalization evidence",
                              "Same targets, not identical exogenous denoising noise across engines",
                              "Peak substep base force versus PhysX contact-history filtering is not an exact detector match"]})
    episodes, traces, paths, wall = run_batch(conditions, manifest, banks, contract, args, output)
    save_json(output / "episodes.json", episodes)
    summary = summarize(episodes); summary["active_loop_wall_s"] = wall
    save_json(output / "summary.json", summary)
    np.savez_compressed(output / "traces.npz", **{f"episode_{i}_{name}": array for i, trace in enumerate(traces) for name, array in trace.items()})
    plots(output, conditions, paths, traces, manifest)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
