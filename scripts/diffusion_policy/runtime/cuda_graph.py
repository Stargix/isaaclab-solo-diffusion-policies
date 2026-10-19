"""Opt-in FP32/K10 CUDA graph execution; stochastic draws remain eager and ordered."""
from functools import partial
import time
import torch
from .sampler import CachedDPPOSampler


class CudaGraphDPPOSampler(CachedDPPOSampler):
    """Fixed-batch, single-thread runtime. Construct/warm before enabling control.

    Weights/equations/noise draw shapes are unchanged. Long-lived input/output
    buffers are private; returned trajectories are cloned so chunk slots cannot
    be overwritten by a later replay. No graph RNG state or mixed precision.
    """

    @torch.inference_mode()
    def __init__(self, policy, *, batch_size=1):
        super().__init__(policy)
        if self.device.type != "cuda" or batch_size < 1 or policy.training:
            raise ValueError("CUDA graph requires an eval-mode CUDA policy and a positive fixed batch")
        scheduler = policy.policy.noise_scheduler
        if scheduler.config.thresholding:
            raise ValueError("Dynamic thresholding is outside the pinned graph contract")
        self.batch_size = batch_size
        started = time.perf_counter()
        self.inputs = [torch.zeros(batch_size, 8, d, device=self.device, dtype=torch.float32) for d in (30, 12, 16)]
        shape = (batch_size, policy.cfg.prediction_horizon, policy.cfg.action_dim)
        self.noises = [torch.zeros(shape, device=self.device, dtype=torch.float32) for _ in range(10)]
        stats = policy.normalizer_stats
        self.stats = [torch.as_tensor(v, dtype=torch.float32, device=self.device) for v in
                      (stats.proprio.mean, stats.proprio.std, stats.action.min, stats.action.max,
                       stats.goal.mean, stats.goal.std)]
        self.timesteps = [torch.full((batch_size,), step[0], device=self.device, dtype=torch.long)
                          for step in self.steps]
        # TransformerDecoder's automatic causal-mask detection reads a CUDA
        # boolean on the host. Supply the *same* verified causal flag explicitly.
        self.original_forwards = []
        for wrapper in (policy.base_model, policy.policy.model):
            model = wrapper.model
            expected = torch.triu(torch.full((16, 16), float("-inf")), diagonal=1)
            if model.encoder_only or not isinstance(model.decoder, torch.nn.TransformerDecoder) or not torch.equal(model.mask.cpu(), expected):
                raise ValueError("Expected the pinned 16-token causal decoder mask")
            decoder = model.decoder
            self.original_forwards.append((decoder, decoder.forward))
        try:
            # Validate both decoders before mutating either owned instance.
            for decoder, forward in self.original_forwards:
                decoder.forward = partial(forward, tgt_is_causal=True, memory_is_causal=False)
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        self._reverse()
                torch.cuda.current_stream(self.device).wait_stream(stream)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=stream):
                    self.static_output = self._reverse()
                torch.cuda.synchronize(self.device)
        except Exception:
            for decoder, forward in self.original_forwards:
                decoder.forward = forward
            raise
        self.startup_s = time.perf_counter() - started

    def _reverse(self):
        p_mean, p_std, a_min, a_max, g_mean, g_std = self.stats
        proprio_n = (self.inputs[0] - p_mean) / p_std
        action_n = 2.0 * (self.inputs[1] - a_min) / (a_max - a_min) - 1.0
        goal_n = (self.inputs[2] - g_mean) / g_std
        sample = self.noises[0]
        noise_index = 1
        scheduler = self.policy.policy.noise_scheduler
        for i, (timestep, alpha_t, alpha_previous, variance, trainable) in enumerate(self.steps):
            model_output = self.policy._model_output(sample, self.timesteps[i], proprio_n, action_n, goal_n, trainable=trainable)
            beta_t = 1.0 - alpha_t
            beta_previous = 1.0 - alpha_previous
            current_alpha = alpha_t / alpha_previous
            current_beta = 1.0 - current_alpha
            clean = (sample - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            if scheduler.config.clip_sample:
                limit = float(scheduler.config.clip_sample_range)
                clean = clean.clamp(-limit, limit)
            mean = alpha_previous.sqrt() * current_beta / beta_t * clean
            mean = mean + current_alpha.sqrt() * beta_previous / beta_t * sample
            if timestep > 0:
                sample = mean + variance.sqrt() * self.noises[noise_index]
                noise_index += 1
            else:
                sample = mean
        sample = sample.clamp(-1.0, 1.0)
        return .5 * (sample + 1.0) * (a_max - a_min) + a_min

    @torch.inference_mode()
    def predict_action_denormalized(self, proprio_hist, action_hist, goal_hist, *, guidance_scale=1., generator=None):
        if guidance_scale != 1. or self.policy.training:
            raise ValueError("Frozen eval-mode graph requires guidance 1")
        for source, target in zip((proprio_hist, action_hist, goal_hist), self.inputs):
            if source.shape != target.shape or source.device != target.device or source.dtype != torch.float32:
                raise ValueError("Graph histories must match its fixed FP32 batch/device/shapes")
            target.copy_(source)
        # Same ten eager draws, same shape and order as CachedDPPOSampler.
        # Models are in eval mode and make no RNG calls between those draws.
        for noise in self.noises:
            noise.copy_(torch.randn(noise.shape, device=self.device, dtype=torch.float32, generator=generator))
        self.graph.replay()
        return self.static_output.clone()
