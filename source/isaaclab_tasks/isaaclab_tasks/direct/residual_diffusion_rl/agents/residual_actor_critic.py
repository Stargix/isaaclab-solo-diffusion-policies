"""Actor-critic initialization for residual control around a frozen prior."""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.modules import ActorCritic


class ResidualActorCritic(ActorCritic):
    """Start the deterministic residual exactly at the identity correction.

    RSL-RL's exploration standard deviation remains trainable and non-zero,
    but the actor mean initially outputs zero for every observation.  This
    preserves the competent frozen prior at iteration zero instead of applying
    an arbitrary correction from a randomly initialized output layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        output_layer = self.actor[-2] if isinstance(self.actor[-1], nn.Unflatten) else self.actor[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError(
                "ResidualActorCritic expected the actor to end in a linear output layer"
            )
        nn.init.zeros_(output_layer.weight)
        if output_layer.bias is not None:
            nn.init.zeros_(output_layer.bias)
