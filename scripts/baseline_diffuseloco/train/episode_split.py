"""Episode-level split utilities for the DiffuseLoco baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeSplit:
    train_demo_indices: tuple[int, ...]
    val_demo_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "train_demo_indices": list(self.train_demo_indices),
            "val_demo_indices": list(self.val_demo_indices),
        }


def split_episode_indices(num_demos: int, val_fraction: float, seed: int) -> EpisodeSplit:
    if num_demos < 2:
        raise ValueError("At least two complete demonstrations are required for an episode-level split.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between 0 and 1.")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(num_demos)
    val_count = min(num_demos - 1, max(1, int(round(num_demos * val_fraction))))
    val = tuple(sorted(int(i) for i in shuffled[:val_count]))
    train = tuple(sorted(int(i) for i in shuffled[val_count:]))
    if set(train) & set(val):
        raise RuntimeError("Episode split overlap detected.")
    return EpisodeSplit(train, val)

