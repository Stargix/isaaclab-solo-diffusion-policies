"""Evolving sensor-to-target timing with real MuJoCo physics and paced ticks."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from .contract import load_runtime_contract
from .cuda_graph import CudaGraphDPPOSampler
from .controller import DelayedHistory, ChunkExecutor, Profile16Route, action_to_joint_target
from .safety import CommandSupervisor, SafetyFault
from scripts.diffusion_policy.sim2sim.backend import MujocoBackend
from scripts.diffusion_policy.sim2sim.scenarios import load_manifest, path_for
from scripts.diffusion_policy.sim2sim.reporting import save_json
from scripts.diffusion_policy.sim2sim.convert import OFFSET, sha
from scripts.diffusion_policy.evaluation.metrics import project_trajectory_to_polyline


def stats(values):
    if not values:
        return None
    return dict(median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                p99_ms=float(np.percentile(values, 99)), max_ms=float(max(values)))


def run(args):
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    contract = load_runtime_contract(args.checkpoint, "cuda:0")
    sampler = CudaGraphDPPOSampler(contract.policy)
    if args.host_buffers == "cpu":
        from .host_buffers import HostBufferSampler
        sampler = HostBufferSampler(sampler)
    history_device = torch.device("cpu") if args.host_buffers == "cpu" else contract.device
    backend = MujocoBackend(args.model, args.dependency_dir, validation_path=args.validation)
    manifest, banks = load_manifest(args.manifest)
    condition = next(c for c in manifest["conditions"] if c["id"] == "coherent_smooth_0.35_walk")
    path, arc, yaw = path_for(condition, banks[(condition["family"], 0)], manifest)
    hist = DelayedHistory.zeros(1, device=history_device)
    executor = ChunkExecutor(sampler)
    offset = torch.tensor(OFFSET, device=history_device, dtype=torch.float32)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    route = Profile16Route(path, arc, yaw, 0, 0., float(arc[-1]), .35)
    # Same nominal staging, outside measured active phase.
    for tick in range(75):
        state = backend.read_state()
        proprio = torch.tensor(state.proprio()[None], device=history_device)
        from scripts.diffusion_policy.train.conditioning.goal_builder import build_geometric_height_profile_goal_from_path
        goal = build_geometric_height_profile_goal_from_path(path, arc, yaw, state.position, state.quaternion,
                goal_horizon_steps=100, dt=.02, speed=0., start_idx=0)
        goals = torch.tensor(goal[None], device=history_device)
        action = torch.zeros(1, 12, device=history_device) if tick < 25 else executor.action(hist.proprio, hist.action, hist.goal)[0]
        actual, target = action_to_joint_target(action, action_scale=.25, q_offset=offset, clip_actions=None)
        _, diagnostic = backend.step_policy_tick(target[0].cpu().numpy())
        if diagnostic.failure:
            raise RuntimeError(f"Staging failure: {diagnostic.failure}")
        hist.commit_tick(proprio, actual, goals)
    executor.reset()
    start = backend.read_state().position.copy()
    route.speed_start_arc = float(project_trajectory_to_polyline(start[None, :2], path[:, :2])[0][0])
    guard = CommandSupervisor(backend.model.jnt_range[backend.joints, 0], backend.model.jnt_range[backend.joints, 1])
    records = []; failure = None; first_supervisor_fault = None; plan_stamp = None; start_wall = time.perf_counter(); sent = 0
    started_unix = time.time(); load_active = False
    for tick in range(round(args.duration_s/.02)):
        release = start_wall + tick*.02
        delay = release-time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        sensor_start = time.perf_counter()
        if args.load_heartbeat and tick % 50 == 0:
            try:
                heartbeat = json.loads(Path(args.load_heartbeat).read_text(encoding="utf-8"))
                load_active = bool(heartbeat["active"] and 0 <= time.time()-heartbeat["wall_time_s"] < 10.)
            except (OSError, ValueError, KeyError):
                load_active = False
        state = backend.read_state()
        goal = route.build_goal(root_pos_w=state.position, root_quat_wxyz=state.quaternion, step=tick)
        proprio = torch.tensor(state.proprio()[None], device=history_device)
        goals = torch.tensor(goal[None], device=history_device)
        replan = executor.chunk is None or executor.next_slot == 0
        if replan:
            plan_stamp = sensor_start
        action, slot, _ = executor.action(hist.proprio, hist.action, hist.goal)
        actual, target = action_to_joint_target(action, action_scale=.25, q_offset=offset, clip_actions=None)
        cpu_target = target[0].cpu().numpy()
        if tick == args.fault_tick and args.inject_fault != "none":
            if args.inject_fault == "nan_target": cpu_target[:] = np.nan
            elif args.inject_fault == "stale_state": sensor_start -= .1
            elif args.inject_fault == "delay": time.sleep(.09)
            elif args.inject_fault == "target_limit": cpu_target[0] = guard.upper[0]+.01
        target_ready = time.perf_counter()
        try:
            admitted = guard.admit(q=state.q, dq=state.dq, target=cpu_target, now=target_ready,
                                  state_stamp=sensor_start, plan_stamp=plan_stamp, sequence=tick)
        except SafetyFault as error:
            if first_supervisor_fault is None:
                first_supervisor_fault = dict(tick=tick, reason=str(error), commands_sent=sent,
                    sensor_age_ms=(target_ready-sensor_start)*1000, plan_age_ms=(target_ready-plan_stamp)*1000)
            if args.supervision == "enforce":
                failure = f"watchdog:{error}"
                records.append(dict(tick=tick, rejected=True, reason=str(error), commands_sent=sent))
                break  # Stop physics; do not issue an unvalidated motor action.
            # Explicit simulator-only profiler: continue original dynamics after
            # recording that the enforced supervisor would already have stopped.
            # This mode cannot be used as a safety gate or in a robot adapter.
            admitted = cpu_target.copy()
        physics_start = time.perf_counter()
        after, diagnostic = backend.step_policy_tick(admitted); sent += 1
        physics_end = time.perf_counter()
        hist.commit_tick(proprio, actual, goals)
        if args.host_buffers != "cpu":
            torch.cuda.synchronize(contract.device)
        tick_end = time.perf_counter()
        records.append(dict(tick=tick, rejected=False, replan=replan, slot=slot,
            load_active=load_active,
            sensor_to_target_ms=(target_ready-sensor_start)*1000, physics_ms=(physics_end-physics_start)*1000,
            total_work_ms=(tick_end-(start_wall+tick*.02))*1000,
            tick_work_ms=(tick_end-sensor_start)*1000,
            release_lateness_ms=(sensor_start-release)*1000,
            tick_deadline_missed=tick_end>release+.02, position=after.position.tolist()))
        if diagnostic.failure:
            failure = f"physics:{diagnostic.failure}"; break
    accepted = [r for r in records if not r["rejected"]]
    report = dict(checkpoint_sha256=contract.sha256, model_sha256=sha(args.model), manifest_sha256=sha(args.manifest),
        source_sha256=sha(__file__), condition=condition, load_label=args.load_label,
        load_log=args.load_log, fault_injection=args.inject_fault, failure=failure,
        fault_tick=args.fault_tick,
        supervision=args.supervision, first_supervisor_fault=first_supervisor_fault,
        host_buffers=args.host_buffers,
        started_unix_s=started_unix, ended_unix_s=time.time(), load_heartbeat=args.load_heartbeat,
        load_active_tick_fraction=float(np.mean([r.get("load_active", False) for r in accepted])) if accepted else 0.,
        expected_ticks=round(args.duration_s/.02), commands_sent=sent, records=records,
        sensor_to_target=stats([r["sensor_to_target_ms"] for r in accepted]),
        replan_sensor_to_target=stats([r["sensor_to_target_ms"] for r in accepted if r["replan"]]),
        full_tick_work=stats([r["total_work_ms"] for r in accepted]),
        tick_work=stats([r["tick_work_ms"] for r in accepted]),
        timing_definitions={"sensor_to_target": "read-start to CPU target, current observations and delayed model histories",
                            "full_tick_work": "Historical name: scheduled release to tick end, includes accumulated lateness",
                            "tick_work": "actual tick start to end; excludes accumulated release lateness"},
        deadline_20ms_misses=sum(r["tick_deadline_missed"] for r in accepted),
        deadline_80ms_replan_misses=sum(r["sensor_to_target_ms"]>.08*1000 for r in accepted if r["replan"]),
        scope="Paced evolving MuJoCo sensor-to-CPU-target loop; no robot transport, actuator thread or realtime certification",
        fault_response=("Latched rejection and terminated simulated episode; no further commands after fault"
                        if args.supervision == "enforce" else "Observation-only profiler; not safety-gated control"))
    save_json(output / "live.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="records"}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model", "validation", "manifest", "output-dir", "dependency-dir"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--duration-s", type=float, default=24.)
    p.add_argument("--load-label", default="standalone")
    p.add_argument("--load-log", default=None)
    p.add_argument("--load-heartbeat", default=None)
    p.add_argument("--inject-fault", choices=("none", "nan_target", "stale_state", "delay", "target_limit"), default="none")
    p.add_argument("--fault-tick", type=int, default=100)
    p.add_argument("--supervision", choices=("enforce", "observe"), default="enforce")
    p.add_argument("--host-buffers", choices=("gpu", "cpu"), default="gpu")
    a = p.parse_args()
    if a.supervision == "observe" and a.inject_fault != "none":
        p.error("Injected-fault validation requires enforced supervision")
    run(a)


if __name__ == "__main__":
    main()
