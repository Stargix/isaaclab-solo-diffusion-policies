"""Configuration dataclasses for diffusion-policy training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .data.obs_utils import ACTION_HIST_DIM, GOAL_DIM, PROPRIO_DIM


@dataclass
class DatasetConfig:
    hdf5_paths: list[str]
    history: int = 8
    prediction_horizon: int = 16
    execution_offset: int = 8
    step_stride: int = 1
    symmetry_mode: str = "quadruped"
    max_stats_samples: int = 20_000
    val_fraction: float = 0.05


@dataclass
class ModelConfig:
    proprio_dim: int = PROPRIO_DIM
    action_hist_dim: int = ACTION_HIST_DIM
    goal_dim: int = GOAL_DIM
    action_dim: int = 12
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    p_drop_emb: float = 0.0
    p_drop_attn: float = 0.3
    separate_goal_conditioning: bool = True


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 10
    num_inference_steps: int | None = None
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    cfg_dropout_prob: float = 0.0


@dataclass
class OptimConfig:
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-3
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    lr_warmup_steps: int = 10_000
    num_workers: int = 0
    seed: int = 42
    mixed_precision: bool = True
    save_every: int = 10
    log_every: int = 50


@dataclass
class TrainConfig:
    dataset: DatasetConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    optim: OptimConfig
    output_dir: str
    run_name: str = "solo12_diffusion_policy"
    wandb_project: str | None = None
    wandb_entity: str | None = None
    schema_version: int = 3
    policy_kind: str = "diffuseloco_velocity_height_ddpm"

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_DEFAULTS = ModelConfig()
DIFFUSION_DEFAULTS = DiffusionConfig()
OPTIM_DEFAULTS = OptimConfig()
DATASET_DEFAULTS = DatasetConfig(hdf5_paths=[])

TRAINING_CONFIG_KEYS = frozenset(
    {
        "history",
        "prediction_horizon",
        "execution_offset",
        "step_stride",
        "symmetry_mode",
        "max_stats_samples",
        "val_fraction",
        "d_model",
        "nhead",
        "num_layers",
        "p_drop_emb",
        "p_drop_attn",
        "diffusion_steps",
        "num_inference_steps",
        "beta_schedule",
        "cfg_dropout_prob",
        "batch_size",
        "epochs",
        "lr",
        "weight_decay",
        "grad_clip_norm",
        "ema_decay",
        "lr_warmup_steps",
        "num_workers",
        "seed",
        "save_every",
        "log_every",
    }
)


def load_training_config_overrides(path: str | Path) -> dict[str, Any]:
    """Load optional training hyperparameter overrides from a JSON file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Training config must be a JSON object: {path}")

    unknown = sorted(set(data) - TRAINING_CONFIG_KEYS)
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {unknown}")

    return data


def resolve_inference_steps(cli_override: int | None, diffusion: DiffusionConfig | dict[str, Any]) -> int:
    """Resolve denoising steps for deployment: CLI > checkpoint > train timesteps."""

    if cli_override is not None:
        return cli_override

    if isinstance(diffusion, dict):
        infer_steps = diffusion.get("num_inference_steps")
        train_steps = int(diffusion["num_train_timesteps"])
    else:
        infer_steps = diffusion.num_inference_steps
        train_steps = diffusion.num_train_timesteps

    if infer_steps is not None:
        return int(infer_steps)
    return train_steps
