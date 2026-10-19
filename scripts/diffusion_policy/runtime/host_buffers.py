"""CPU history/chunk boundary: transfer only when replanning, no policy changes."""
import torch


class HostBufferSampler:
    def __init__(self, sampler):
        self.sampler = sampler

    @torch.inference_mode()
    def predict_action_denormalized(self, proprio, action, goal, *, guidance_scale=1., generator=None):
        values = (proprio, action, goal)
        if any(v.device.type != "cpu" or v.dtype != torch.float32 for v in values):
            raise ValueError("Host histories require CPU FP32 tensors")
        trajectory = self.sampler.predict_action_denormalized(
            *(v.to(self.sampler.device) for v in values), guidance_scale=guidance_scale, generator=generator)
        # Blocking transfer completes the plan before it is exposed to the CPU
        # executor. Four distinct slots are retained; no repeated stale action.
        return trajectory.cpu()

    def executable_chunk(self, trajectory, num_actions):
        return self.sampler.executable_chunk(trajectory, num_actions)
