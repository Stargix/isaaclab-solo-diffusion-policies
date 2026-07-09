"""TransformerForDiffusion adapted from the Diffusion Policy codebase.

Reference implementation:
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/transformer_for_diffusion.py

This local copy keeps the tested architecture/API used by Diffusion Policy and
DiffuseLoco, while removing package-level dependencies that do not exist in this
Isaac Lab workspace.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding used for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class TransformerForDiffusion(nn.Module):
    """Time-series diffusion transformer used by Diffusion Policy."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        horizon: int,
        n_obs_steps: int | None = None,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        causal_attn: bool = False,
        time_as_cond: bool = True,
        obs_as_cond: bool = False,
        n_cond_layers: int = 0,
    ) -> None:
        super().__init__()

        if n_obs_steps is None:
            n_obs_steps = horizon

        tokens = horizon
        cond_tokens = 1
        if not time_as_cond:
            tokens += 1
            cond_tokens -= 1
        obs_as_cond = cond_dim > 0 or obs_as_cond
        if obs_as_cond:
            if not time_as_cond:
                raise ValueError("obs_as_cond requires time_as_cond=True.")
            cond_tokens += n_obs_steps

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, tokens, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb) if obs_as_cond else None

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = cond_tokens <= 0
        if not encoder_only:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, cond_tokens, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4 * n_emb,
                    dropout=p_drop_attn,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_cond_layers)
            else:
                self.encoder = nn.Sequential(nn.Linear(n_emb, 4 * n_emb), nn.Mish(), nn.Linear(4 * n_emb, n_emb))

            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layer)

        if causal_attn:
            mask = (torch.triu(torch.ones(tokens, tokens)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)

            if time_as_cond and obs_as_cond:
                target_positions, source_positions = torch.meshgrid(
                    torch.arange(tokens),
                    torch.arange(cond_tokens),
                    indexing="ij",
                )
                memory_mask = target_positions >= (source_positions - 1)
                memory_mask = (
                    memory_mask.float().masked_fill(memory_mask == 0, float("-inf")).masked_fill(memory_mask == 1, 0.0)
                )
                self.register_buffer("memory_mask", memory_mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.T = tokens
        self.T_cond = cond_tokens
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        self.apply(self._init_weights)
        logger.info("TransformerForDiffusion parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module: nn.Module) -> None:
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def get_optim_groups(self, weight_decay: float = 1.0e-3) -> list[dict]:
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, _ in module.named_parameters():
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param_name.endswith("bias") or param_name.startswith("bias"):
                    no_decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif param_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        no_decay.add("pos_emb")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {name: param for name, param in self.named_parameters()}
        inter_params = decay & no_decay
        if inter_params:
            raise RuntimeError(f"Parameters in both decay/no_decay sets: {inter_params}")
        missing = param_dict.keys() - (decay | no_decay)
        if missing:
            raise RuntimeError(f"Parameters not separated into decay/no_decay sets: {missing}")

        return [
            {"params": [param_dict[name] for name in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[name] for name in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 1.0e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.get_optim_groups(weight_decay=weight_decay), lr=learning_rate, betas=betas)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        input_emb = self.input_emb(sample)
        if self.encoder_only:
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            pos = self.pos_emb[:, : token_embeddings.shape[1], :]
            x = self.drop(token_embeddings + pos)
            x = self.encoder(src=x, mask=self.mask)
            x = x[:, 1:, :]
        else:
            cond_embeddings = time_emb
            if self.obs_as_cond:
                if cond is None:
                    raise ValueError("cond must be provided when obs_as_cond=True.")
                cond_embeddings = torch.cat([cond_embeddings, self.cond_obs_emb(cond)], dim=1)
            cond_pos = self.cond_pos_emb[:, : cond_embeddings.shape[1], :]
            memory = self.encoder(self.drop(cond_embeddings + cond_pos))

            token_embeddings = input_emb
            pos = self.pos_emb[:, : token_embeddings.shape[1], :]
            x = self.drop(token_embeddings + pos)
            x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask, memory_mask=self.memory_mask)

        return self.head(self.ln_f(x))

