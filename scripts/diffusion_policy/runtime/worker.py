"""Single-owner inference worker with copied inputs and timestamped plans.

The receiver consumes each slot once. This is a software interface, not a
hardware watchdog or a hard realtime scheduler.
"""
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread
import time
import numpy as np
import torch


@dataclass(frozen=True)
class Plan:
    sequence: int
    requested_at: float
    ready_at: float
    actions: np.ndarray


class InferenceWorker:
    def __init__(self, sampler, *, injected_delay_s=0.):
        self.sampler = sampler
        self.delay = injected_delay_s
        self.requests = Queue(maxsize=1)
        self.results = Queue(maxsize=1)
        self.thread = Thread(target=self._run, name="policy-inference", daemon=True)
        self.thread.start()

    def submit(self, sequence, requested_at, history):
        # The receiver retains ownership of mutable history buffers.
        self.requests.put_nowait((sequence, requested_at,
                                 tuple(x.detach().clone() for x in history)))

    def warmup(self, history, *, device, repeats=3):
        """Initialize the worker's device context before enabling the receiver.

        Restore RNG after warmup so it does not change the active noise stream.
        No other caller may use the model/generator concurrently.
        """
        device = torch.device(device)
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            for index in range(repeats):
                sequence = -index-1
                self.submit(sequence,time.perf_counter(),history)
                self.receive(sequence,time.perf_counter()+30.)

    def _run(self):
        with torch.inference_mode():
            while True:
                item = self.requests.get()
                if item is None:
                    return
                sequence, stamp, histories = item
                try:
                    if self.delay:
                        time.sleep(self.delay)
                    trajectory = self.sampler.predict_action_denormalized(*histories)
                    chunk = self.sampler.executable_chunk(trajectory, 4)
                    actions = chunk[0].detach().cpu().numpy().copy()
                    if actions.shape != (4, 12) or not np.isfinite(actions).all():
                        raise ValueError("Invalid four-action inference result")
                    actions.setflags(write=False)
                    self.results.put(Plan(sequence, stamp, time.perf_counter(), actions))
                except Exception as exc:
                    self.results.put(exc)

    def receive(self, sequence, deadline):
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            raise TimeoutError("dispatch_deadline")
        try:
            result = self.results.get(timeout=timeout)
        except Empty as exc:
            raise TimeoutError("plan_not_ready") from exc
        if isinstance(result, Exception):
            raise RuntimeError("Inference failed") from result
        if result.sequence != sequence:
            raise ValueError("Plan sequence does not match receiver")
        if result.ready_at > deadline or time.perf_counter() > deadline:
            raise TimeoutError("late_plan")
        return result

    def close(self):
        # A blocked native GPU call cannot be safely cancelled from Python.
        # The simulation receiver has already stopped sending commands.
        try:
            self.requests.put_nowait(None)
        except Exception:
            pass
        self.thread.join(timeout=2.)
        return not self.thread.is_alive()
