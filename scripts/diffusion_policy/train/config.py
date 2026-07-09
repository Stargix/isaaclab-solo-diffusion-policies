"""Configuration dataclasses for diffusion-policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DatasetConfig:
    hdf5_paths: list[str]
    history: int = 8
    action_horizon: int = 4
    min_segment_steps: int = 50
    max_segment_steps: int = 150
    segment_stride: int = 10
    dt: float = 0.02
    v_req_clip: float = 2.0
    symmetry_mode: str = "quadruped"
    max_stats_samples: int = 20000
    val_fraction: float = 0.05


@dataclass
class ModelConfig:
    obs_dim: int = 42
    goal_dim: int = 11
    action_dim: int = 12
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 100
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"
    variance_type: str = "fixed_small"
    clip_sample: bool = True
    cfg_dropout_prob: float = 0.2


@dataclass
class OptimConfig:
    batch_size: int = 256
    epochs: int = 200
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    grad_clip_norm: float = 1.0
    ema_decay: float = 0.9999
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

    def to_dict(self) -> dict:
        return asdict(self)

