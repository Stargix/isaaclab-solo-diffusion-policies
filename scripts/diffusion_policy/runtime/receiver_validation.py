"""Independent paced MuJoCo plant and inference worker; simulator only.

Fixed 15-ms dispatch phase is an explicit new execution delay, not an
algebra-preserving replacement for the synchronous task evaluator. No motors
or network interface are opened. Fault response terminates the simulation.
"""
import argparse
from dataclasses import asdict
from threading import Thread, Lock, Event
import time
import platform
import numpy as np
import torch
from .contract import load_runtime_contract
from .controller import DelayedHistory, ChunkExecutor, Profile16Route, action_to_joint_target
from .host_buffers import HostBufferSampler
from .worker import InferenceWorker
from .safety import CommandSupervisor
from .live import stats
from scripts.diffusion_policy.sim2sim.backend import MujocoBackend
from scripts.diffusion_policy.sim2sim.scenarios import load_manifest, path_for
from scripts.diffusion_policy.sim2sim.convert import OFFSET, sha
from scripts.diffusion_policy.sim2sim.reporting import save_json
from scripts.diffusion_policy.evaluation.metrics import project_trajectory_to_polyline
from scripts.diffusion_policy.train.conditioning.goal_builder import build_geometric_height_profile_goal_from_path


class PacedPlant:
    def __init__(self, backend, target):
        self.backend = backend
        self.target = target.copy()
        self.lock = Lock()
        self.stop = Event()
        self.snapshot = (backend.read_state(), time.perf_counter())
        self.failure = None
        self.records = []
        self.thread = Thread(target=self._run, name="simulation-plant", daemon=True)

    def _run(self):
        origin = time.perf_counter()
        step = 0
        while not self.stop.is_set():
            release = origin + step * .005
            if self.stop.wait(max(0., release-time.perf_counter())):
                break
            begun = time.perf_counter()
            with self.lock:
                target = self.target.copy()
            state, diagnostic = self.backend.step_policy_tick(target, substeps=1)
            ended = time.perf_counter()
            with self.lock:
                self.snapshot = (state, begun)
                self.records.append(dict(step=step, lateness_ms=(begun-release)*1000,
                                         work_ms=(ended-begun)*1000, sim_time_s=state.time_s))
                if diagnostic.failure:
                    self.failure = diagnostic.failure
                    self.stop.set()
            step += 1

    def read(self):
        with self.lock:
            return self.snapshot, self.failure

    def dispatch(self, target):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Plant stopped")
            self.target = target.copy()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)
        if self.thread.is_alive():
            raise RuntimeError("Plant thread did not stop")


def run(args):
    from pathlib import Path
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    contract = load_runtime_contract(args.checkpoint, args.device)
    if contract.device.type == "cuda":
        from .cuda_graph import CudaGraphDPPOSampler
        sampler = HostBufferSampler(CudaGraphDPPOSampler(contract.policy))
    else:
        sampler = HostBufferSampler(contract.sampler)
    backend = MujocoBackend(args.model, args.dependency_dir, validation_path=args.validation)
    manifest, banks = load_manifest(args.manifest)
    case = next(c for c in manifest['conditions'] if c['id'] == 'coherent_smooth_0.35_walk')
    path, arc, yaw = path_for(case, banks[(case['family'], 0)], manifest)
    hist = DelayedHistory.zeros(1, device=torch.device('cpu'))
    executor = ChunkExecutor(sampler)
    offset = torch.tensor(OFFSET, dtype=torch.float32)
    torch.manual_seed(42)
    if contract.device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    target = OFFSET.copy()
    # Original synchronous posture staging is outside paced measurements.
    for tick in range(75):
        state = backend.read_state()
        goal = build_geometric_height_profile_goal_from_path(path, arc, yaw, state.position,
            state.quaternion, goal_horizon_steps=100, dt=.02, speed=0., start_idx=0)
        action = torch.zeros(1,12) if tick < 25 else executor.action(hist.proprio,hist.action,hist.goal)[0]
        actual, qtarget = action_to_joint_target(action, action_scale=.25,q_offset=offset,clip_actions=None)
        target = qtarget[0].numpy()
        _, d = backend.step_policy_tick(target)
        if d.failure:
            raise RuntimeError(f'Staging failure: {d.failure}')
        hist.commit_tick(torch.tensor(state.proprio()[None]),actual,torch.tensor(goal[None]))
    start_arc = float(project_trajectory_to_polyline(state.position[None,:2],path[:,:2])[0][0])
    route = Profile16Route(path,arc,yaw,0,start_arc,float(arc[-1]),.35)
    guard = CommandSupervisor(backend.model.jnt_range[backend.joints,0],backend.model.jnt_range[backend.joints,1])
    if args.worker_mode=='process':
        from .process_worker import ProcessInferenceWorker
        worker=ProcessInferenceWorker(contract.path,contract.device,(hist.proprio,hist.action,hist.goal),
                                      injected_delay_s=args.inject_delay_ms/1000.)
    else:
        worker = InferenceWorker(sampler,injected_delay_s=args.inject_delay_ms/1000.)
        worker.warmup((hist.proprio,hist.action,hist.goal),device=contract.device)
    plant = PacedPlant(backend,target)
    records = []; fault = None; plan = None
    origin = time.perf_counter()
    plant.thread.start()
    try:
        for tick in range(round(args.duration_s/.02)):
            release = origin+tick*.02
            time.sleep(max(0.,release-time.perf_counter()))
            (state, stamp), failure = plant.read()
            if failure:
                raise RuntimeError(f'physics:{failure}')
            acquired = time.perf_counter()
            goal = route.build_goal(root_pos_w=state.position,root_quat_wxyz=state.quaternion,step=tick)
            if tick % 4 == 0:
                worker.submit(tick, time.perf_counter(), (hist.proprio,hist.action,hist.goal))
                plan = worker.receive(tick,release+.015)
            dispatch_at = release+.015
            time.sleep(max(0.,dispatch_at-time.perf_counter()))
            now = time.perf_counter()
            if now > release+.02:
                raise TimeoutError('receiver_deadline')
            # State freshness is checked from independent sensor/plant timestamp.
            (latest, latest_stamp), failure = plant.read()
            if failure:
                raise RuntimeError(f'physics:{failure}')
            action = torch.tensor(plan.actions[tick % 4].copy()[None])
            actual,qtarget = action_to_joint_target(action,action_scale=.25,q_offset=offset,clip_actions=None)
            admitted = guard.admit(q=latest.q,dq=latest.dq,target=qtarget[0].numpy(),now=now,
                state_stamp=latest_stamp,plan_stamp=plan.requested_at,sequence=tick)
            dispatch_time=time.perf_counter()
            if dispatch_time > release+.02:
                raise TimeoutError('receiver_validation_overran_deadline')
            plant.dispatch(admitted)
            hist.commit_tick(torch.tensor(state.proprio()[None]),actual,torch.tensor(goal[None]))
            records.append(dict(tick=tick,slot=tick%4,plan_sequence=plan.sequence,
                dispatch_lateness_ms=(dispatch_time-dispatch_at)*1000,state_age_ms=(dispatch_time-latest_stamp)*1000,
                plan_age_ms=(dispatch_time-plan.requested_at)*1000,position=latest.position.tolist(),
                inference_ms=(plan.ready_at-plan.requested_at)*1000 if tick%4==0 else None,
                acquisition_lateness_ms=(acquired-release)*1000))
    except Exception as error:
        fault = dict(tick=len(records),reason=f'{type(error).__name__}:{error}',commands_sent=len(records))
    finally:
        plant.close()
        worker_stopped = worker.close()
    result = dict(checkpoint_sha256=contract.sha256,device=str(contract.device),
        model_sha256=sha(args.model),validation_sha256=sha(args.validation),manifest_sha256=sha(args.manifest),
        source_sha256={str(p.relative_to(Path(__file__).parents[3])):sha(p)
                       for folder in (Path(__file__).parent,Path(__file__).parent.parent/'sim2sim')
                       for p in folder.glob('*.py')},
        torch_version=torch.__version__,numpy_version=np.__version__,torch_threads=torch.get_num_threads(),
        processor=platform.processor(),gpu=torch.cuda.get_device_name(contract.device) if contract.device.type=='cuda' else None,
        expected_ticks=round(args.duration_s/.02),commands_sent=len(records),fault=fault,
        worker_stopped=worker_stopped,worker_mode=args.worker_mode,dispatch_phase_ms=15,inject_delay_ms=args.inject_delay_ms,
        records=records,plant_records=plant.records,
        inference=stats([r['inference_ms'] for r in records if r['inference_ms'] is not None]),
        dispatch_lateness=stats([r['dispatch_lateness_ms'] for r in records]),
        scope='Local computer only. Best-effort Python scheduling with independently paced discrete MuJoCo plant. No robot transport, hardware safety or hard realtime claim.')
    save_json(out/'receiver.json',result)
    print({k:v for k,v in result.items() if k not in ('records','plant_records')},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','model','validation','manifest','dependency-dir','output-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--duration-s',type=float,default=24.)
    p.add_argument('--inject-delay-ms',type=float,default=0.)
    p.add_argument('--worker-mode',choices=('process','thread'),default='process')
    a=p.parse_args()
    if not np.isfinite([a.duration_s,a.inject_delay_ms]).all() or a.duration_s<=0 or a.inject_delay_ms<0:
        p.error('Require finite positive duration and nonnegative delay')
    run(a)


if __name__=='__main__':
    main()
