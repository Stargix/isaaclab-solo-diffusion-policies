"""Direct, conditioned action-chunk Transformer for the deterministic BC control.

This intentionally mirrors the conditioning and decoder choices of
``TransformerForDiffusion`` without accepting future ground-truth actions or a
diffusion timestep.  Learned query tokens are the direct-regression analogue
of the denoising trajectory tokens.
"""

from __future__ import annotations

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


class TransformerChunkPolicy(nn.Module):
    """Decode an action horizon from observed proprioception, actions and goals."""

    def __init__(
        self,
        *,
        proprio_dim: int,
        action_hist_dim: int,
        goal_dim: int,
        action_dim: int,
        history: int,
        prediction_horizon: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        p_drop_emb: float,
        p_drop_attn: float,
    ) -> None:
        super().__init__()
        self.history = history
        self.prediction_horizon = prediction_horizon
        self.action_dim = action_dim
        self.io_encoder = self._make_cond_mlp(proprio_dim + action_hist_dim, d_model)
        self.goal_encoder = self._make_cond_mlp(goal_dim, d_model)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 2 * history, d_model))
        self.query_emb = nn.Parameter(torch.zeros(1, prediction_horizon, d_model))
        self.query_pos_emb = nn.Parameter(torch.zeros(1, prediction_horizon, d_model))
        self.drop = nn.Dropout(p_drop_emb)
        self.memory_encoder = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.Mish(), nn.Linear(4 * d_model, d_model)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=p_drop_attn,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, action_dim)

        target_mask = torch.triu(
            torch.full((prediction_horizon, prediction_horizon), float("-inf")), diagonal=1
        )
        self.register_buffer("target_mask", target_mask)
        # Query j may use matching history through j.  Queries beyond the
        # observed eight tokens see all observed history; no future action is a
        # decoder input at train or test time.
        memory_mask = torch.full((prediction_horizon, 2 * history), float("-inf"))
        for query_index in range(prediction_horizon):
            visible = min(query_index + 1, history)
            memory_mask[query_index, :visible] = 0.0
            memory_mask[query_index, history : history + visible] = 0.0
        self.register_buffer("memory_mask", memory_mask)
        self.apply(self._init_weights)
        logger.info("TransformerChunkPolicy parameters: %e", sum(p.numel() for p in self.parameters()))

    @staticmethod
    def _make_cond_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Mish(), nn.Linear(out_dim, out_dim))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            for name in ("in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"):
                weight = getattr(module, name)
                if weight is not None:
                    nn.init.normal_(weight, mean=0.0, std=0.02)
            for name in ("in_proj_bias", "bias_k", "bias_v"):
                bias = getattr(module, name)
                if bias is not None:
                    nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        elif isinstance(module, TransformerChunkPolicy):
            for parameter in (module.cond_pos_emb, module.query_emb, module.query_pos_emb):
                nn.init.normal_(parameter, mean=0.0, std=0.02)

    def configure_optimizers(
        self,
        *,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.Optimizer:
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if name.endswith("bias") or "ln_f" in name or "norm" in name or name.endswith("pos_emb") or name.endswith("query_emb"):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=learning_rate,
            betas=betas,
        )

    def forward(self, proprio_hist: torch.Tensor, action_hist: torch.Tensor, goal_hist: torch.Tensor) -> torch.Tensor:
        if proprio_hist.shape[1] != self.history or action_hist.shape[1] != self.history or goal_hist.shape[1] != self.history:
            raise ValueError("All conditioning histories must match the configured history length.")
        io_tokens = self.io_encoder(torch.cat((proprio_hist, action_hist), dim=-1))
        goal_tokens = self.goal_encoder(goal_hist)
        memory = torch.cat((io_tokens, goal_tokens), dim=1)
        memory = self.memory_encoder(self.drop(memory + self.cond_pos_emb))
        query = self.drop(self.query_emb + self.query_pos_emb).expand(proprio_hist.shape[0], -1, -1)
        decoded = self.decoder(
            tgt=query,
            memory=memory,
            tgt_mask=self.target_mask,
            memory_mask=self.memory_mask,
        )
        return self.head(self.ln_f(decoded))
