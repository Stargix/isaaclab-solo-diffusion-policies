"""Process-owned inference so Python model dispatch cannot hold receiver GIL."""
import multiprocessing as mp
from queue import Empty, Full
import time
import numpy as np
import torch
from .worker import Plan


def _child(checkpoint,device,histories,rng,requests,results):
    try:
        from .contract import load_runtime_contract
        from .host_buffers import HostBufferSampler
        torch.set_num_threads(1)
        contract=load_runtime_contract(checkpoint,device)
        if contract.device.type=='cuda':
            from .cuda_graph import CudaGraphDPPOSampler
            sampler=HostBufferSampler(CudaGraphDPPOSampler(contract.policy))
        else:
            sampler=HostBufferSampler(contract.sampler)
        inputs=tuple(torch.tensor(v,dtype=torch.float32) for v in histories)
        devices=[contract.device.index or 0] if contract.device.type=='cuda' else []
        with torch.inference_mode(),torch.random.fork_rng(devices=devices):
            for _ in range(3):sampler.predict_action_denormalized(*inputs)
        torch.set_rng_state(torch.tensor(rng['cpu'],dtype=torch.uint8))
        if devices:torch.cuda.set_rng_state(torch.tensor(rng['cuda'],dtype=torch.uint8),contract.device)
        results.put(('ready',contract.sha256))
        with torch.inference_mode():
            while True:
                item=requests.get()
                if item is None:return
                sequence,stamp,histories,delay,replay_rng=item
                # Oracle-only replay can restore environment-interleaved noise.
                # Normal execution owns its independent continuous RNG stream.
                if replay_rng is not None:
                    torch.set_rng_state(torch.tensor(replay_rng['cpu'],dtype=torch.uint8))
                    if devices:torch.cuda.set_rng_state(torch.tensor(replay_rng['cuda'],dtype=torch.uint8),contract.device)
                if delay:time.sleep(delay)
                inputs=tuple(torch.tensor(v,dtype=torch.float32) for v in histories)
                trajectory=sampler.predict_action_denormalized(*inputs)
                actions=sampler.executable_chunk(trajectory,4)[0].numpy().copy()
                if actions.shape!=(4,12) or not np.isfinite(actions).all():
                    raise ValueError('Invalid process plan')
                results.put(Plan(sequence,stamp,time.perf_counter(),actions))
    except Exception as error:
        results.put(('error',type(error).__name__,str(error)))


class ProcessInferenceWorker:
    def __init__(self,checkpoint,device,history,*,injected_delay_s=0.):
        context=mp.get_context('spawn')
        self.requests=context.Queue(maxsize=1);self.results=context.Queue(maxsize=1)
        self.delay=injected_delay_s
        rng={'cpu':torch.get_rng_state().numpy().copy()}
        if torch.device(device).type=='cuda':rng['cuda']=torch.cuda.get_rng_state(device).cpu().numpy().copy()
        self.process=context.Process(target=_child,args=(str(checkpoint),str(device),
            tuple(x.numpy().copy() for x in history),rng,self.requests,self.results),daemon=True)
        self.process.start()
        try:
            ready=self.results.get(timeout=120.)
            if ready[0]!='ready':raise RuntimeError(f'Worker initialization failed: {ready}')
        except Exception:
            self.close();raise

    def submit(self,sequence,requested_at,history,*,replay_rng=None):
        if not self.process.is_alive():raise RuntimeError('Inference process is not alive')
        values=tuple(x.detach().numpy().copy() for x in history)
        self.requests.put_nowait((sequence,requested_at,values,self.delay,replay_rng))

    def receive(self,sequence,deadline):
        timeout=deadline-time.perf_counter()
        if timeout<=0:raise TimeoutError('dispatch_deadline')
        try:result=self.results.get(timeout=timeout)
        except Empty as error:raise TimeoutError('plan_not_ready') from error
        if not isinstance(result,Plan):raise RuntimeError(f'Inference process error: {result}')
        if result.sequence!=sequence:raise ValueError('Wrong process plan sequence')
        if result.ready_at>deadline or time.perf_counter()>deadline:raise TimeoutError('late_plan')
        result.actions.setflags(write=False)
        return result

    def close(self):
        try:self.requests.put_nowait(None)
        except Full:pass
        self.process.join(timeout=2.)
        if self.process.is_alive():
            # Only this object's own child; the simulated receiver is stopped.
            self.process.terminate();self.process.join(timeout=2.)
        stopped=not self.process.is_alive()
        self.requests.close();self.results.close()
        return stopped
