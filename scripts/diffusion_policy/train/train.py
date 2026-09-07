#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Solo12 Diffusion Policy from merged HDF5 demonstrations.

Example:
    python scripts/diffusion_policy/train/train.py \
        --datasets scripts/diffusion_policy/data/datasets/combined_A.hdf5 \
        --output_dir scripts/diffusion_policy/runs/baseline_A \
        --symmetry_mode quadruped
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.ema_model import EMAModel
    from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
    from train.config import (
        DATASET_DEFAULTS,
        DIFFUSION_DEFAULTS,
        MODEL_DEFAULTS,
        OPTIM_DEFAULTS,
        DatasetConfig,
        DiffusionConfig,
        ModelConfig,
        OptimConfig,
        TrainConfig,
        load_training_config_overrides,
    )
    from train.conditioning.goal_builder import GOAL_SCHEMA_NAME, REFERENCE_GOAL_SCHEMA_NAME, goal_dimension, goal_schema_name, REFERENCE_GOAL_REPRESENTATIONS
    from train.data.dataset import SpatialHindsightDataset
    from train.data.episode_split import split_episode_indices
    from train.runtime.checkpoint import load_training_checkpoint
else:  # pragma: no cover
    from ..model.ema_model import EMAModel
    from ..model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
    from .config import (
        DATASET_DEFAULTS,
        DIFFUSION_DEFAULTS,
        MODEL_DEFAULTS,
        OPTIM_DEFAULTS,
        DatasetConfig,
        DiffusionConfig,
        ModelConfig,
        OptimConfig,
        TrainConfig,
        load_training_config_overrides,
    )
    from .conditioning.goal_builder import GOAL_SCHEMA_NAME, REFERENCE_GOAL_SCHEMA_NAME, goal_dimension, goal_schema_name, REFERENCE_GOAL_REPRESENTATIONS
    from .data.dataset import SpatialHindsightDataset
    from .data.episode_split import split_episode_indices
    from .runtime.checkpoint import load_training_checkpoint


def _find_cli_value(argv: list[str], flag: str) -> str | None:
    for index, arg in enumerate(argv):
        if arg == flag and index + 1 < len(argv):
            return argv[index + 1]
    return None


def parse_args() -> argparse.Namespace:
    config_path = _find_cli_value(sys.argv[1:], "--config")
    config_defaults = load_training_config_overrides(config_path) if config_path else {}

    parser = argparse.ArgumentParser(description="Train Solo12 Diffusion Policy.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON file with hyperparameter overrides.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Merged HDF5 files to train on.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and config.")
    parser.add_argument("--run_name", default="solo12_diffusion_policy", help="Run name for logs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume model/EMA/optimizer/scheduler from a compatible checkpoint.")

    parser.add_argument("--history", type=int, default=DATASET_DEFAULTS.history)
    parser.add_argument("--prediction_horizon", type=int, default=DATASET_DEFAULTS.prediction_horizon)
    parser.add_argument("--execution_offset", type=int, default=DATASET_DEFAULTS.execution_offset)
    parser.add_argument("--goal_horizon_steps", type=int, default=DATASET_DEFAULTS.goal_horizon_steps)
    parser.add_argument(
        "--waypoint_time_offsets_s",
        type=float,
        nargs=3,
        default=DATASET_DEFAULTS.waypoint_time_offsets_s,
        metavar=("T1", "T2", "T3"),
        help="Temporal preview offsets in seconds; all must precede the terminal horizon.",
    )
    parser.add_argument(
        "--goal_representation",
        choices=REFERENCE_GOAL_REPRESENTATIONS,
        default=DATASET_DEFAULTS.goal_representation,
        help=("path11 is the legacy temporal preview; hindsight_geom_avg12 uses achieved geometric waypoints "
              "plus terminal pose and average speed; hindsight_geom_profile16 adds spatially aligned "
              "task-height previews; holonomic_se2_32 exposes route SE(2) tokens; "
              "path_guidance_se2_36 uses a stored guide plus terminal pose."),
    )
    parser.add_argument("--step_stride", type=int, default=DATASET_DEFAULTS.step_stride, help="Temporal stride to sub-sample step windows.")
    parser.add_argument("--v_req_clip", type=float, default=DATASET_DEFAULTS.v_req_clip)
    parser.add_argument(
        "--goal_source",
        choices=["achieved", "reference"],
        default=DATASET_DEFAULTS.goal_source,
        help="achieved = legacy hindsight; reference = explicit route stored in the HDF5.",
    )
    parser.add_argument(
        "--waypoint_noise_std_m",
        type=float,
        default=DATASET_DEFAULTS.waypoint_noise_std_m,
        help=("Training-only Gaussian jitter in metres applied to the three intermediate path11 waypoints. "
              "The terminal pose is never changed."),
    )
    parser.add_argument(
        "--waypoint_noise_clip_m",
        type=float,
        default=DATASET_DEFAULTS.waypoint_noise_clip_m,
        help="Per-waypoint radial clip for --waypoint_noise_std_m; zero disables clipping.",
    )
    parser.add_argument(
        "--include_padded_starts",
        action="store_true",
        default=DATASET_DEFAULTS.include_padded_starts,
        help="Include reset samples with left-padded state/action/goal histories.",
    )
    parser.add_argument(
        "--startup_sample_multiplier",
        type=int,
        default=DATASET_DEFAULTS.startup_sample_multiplier,
        help="Repeat reset/start anchors so they remain visible among 20-second episodes.",
    )
    parser.add_argument("--symmetry_mode", choices=["none", "mirror", "quadruped"], default=DATASET_DEFAULTS.symmetry_mode)
    parser.add_argument("--max_stats_samples", type=int, default=DATASET_DEFAULTS.max_stats_samples)
    parser.add_argument("--val_fraction", type=float, default=DATASET_DEFAULTS.val_fraction)

    parser.add_argument("--d_model", type=int, default=MODEL_DEFAULTS.d_model)
    parser.add_argument("--nhead", type=int, default=MODEL_DEFAULTS.nhead)
    parser.add_argument("--num_layers", type=int, default=MODEL_DEFAULTS.num_layers)
    parser.add_argument("--p_drop_emb", type=float, default=MODEL_DEFAULTS.p_drop_emb)
    parser.add_argument("--p_drop_attn", type=float, default=MODEL_DEFAULTS.p_drop_attn)

    parser.add_argument("--diffusion_steps", type=int, default=DIFFUSION_DEFAULTS.num_train_timesteps)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Denoising steps at deploy time; defaults to diffusion_steps.",
    )
    parser.add_argument("--beta_schedule", default=DIFFUSION_DEFAULTS.beta_schedule)
    parser.add_argument("--cfg_dropout_prob", type=float, default=DIFFUSION_DEFAULTS.cfg_dropout_prob)
    parser.add_argument("--batch_size", type=int, default=OPTIM_DEFAULTS.batch_size)
    parser.add_argument("--epochs", type=int, default=OPTIM_DEFAULTS.epochs)
    parser.add_argument("--lr", type=float, default=OPTIM_DEFAULTS.learning_rate)
    parser.add_argument("--weight_decay", type=float, default=OPTIM_DEFAULTS.weight_decay)
    parser.add_argument("--grad_clip_norm", type=float, default=OPTIM_DEFAULTS.grad_clip_norm)
    parser.add_argument("--ema_decay", type=float, default=OPTIM_DEFAULTS.ema_decay)
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=OPTIM_DEFAULTS.lr_warmup_steps,
        help="Linear LR warmup steps before cosine decay (DiffuseLoco default: 10000).",
    )
    parser.add_argument("--num_workers", type=int, default=OPTIM_DEFAULTS.num_workers)
    parser.add_argument("--seed", type=int, default=OPTIM_DEFAULTS.seed)
    parser.add_argument("--save_every", type=int, default=OPTIM_DEFAULTS.save_every)
    parser.add_argument("--log_every", type=int, default=OPTIM_DEFAULTS.log_every)
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision.")

    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    if args.num_inference_steps is None:
        args.num_inference_steps = args.diffusion_steps
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace) -> TrainConfig:
    is_reference = args.goal_source == "reference"
    if args.waypoint_noise_std_m < 0.0 or args.waypoint_noise_clip_m < 0.0:
        raise ValueError("Waypoint-noise magnitudes must be non-negative.")
    if args.waypoint_noise_std_m > 0.0 and (args.goal_source != "achieved" or args.goal_representation != "path11"):
        raise ValueError(
            "Waypoint jitter is the simple hindsight-path augmentation and requires "
            "--goal_source achieved --goal_representation path11."
        )
    if args.goal_representation in {"hindsight_geom_avg12", "hindsight_geom_profile16"} and args.goal_source != "achieved":
        raise ValueError(f"{args.goal_representation} requires --goal_source achieved.")
    return TrainConfig(
        dataset=DatasetConfig(
            hdf5_paths=args.datasets,
            history=args.history,
            prediction_horizon=args.prediction_horizon,
            execution_offset=args.execution_offset,
            goal_horizon_steps=args.goal_horizon_steps,
            waypoint_time_offsets_s=tuple(args.waypoint_time_offsets_s),
            step_stride=args.step_stride,
            v_req_clip=args.v_req_clip,
            goal_source=args.goal_source,
            goal_representation=args.goal_representation,
            waypoint_noise_std_m=args.waypoint_noise_std_m,
            waypoint_noise_clip_m=args.waypoint_noise_clip_m,
            include_padded_starts=args.include_padded_starts,
            startup_sample_multiplier=args.startup_sample_multiplier,
            symmetry_mode=args.symmetry_mode,
            max_stats_samples=args.max_stats_samples,
            val_fraction=args.val_fraction,
        ),
        model=ModelConfig(
            goal_dim=goal_dimension(args.goal_representation),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=4 * args.d_model,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
        ),
        diffusion=DiffusionConfig(
            num_train_timesteps=args.diffusion_steps,
            num_inference_steps=args.num_inference_steps,
            beta_schedule=args.beta_schedule,
            cfg_dropout_prob=args.cfg_dropout_prob,
        ),
        optim=OptimConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            ema_decay=args.ema_decay,
            lr_warmup_steps=args.lr_warmup_steps,
            num_workers=args.num_workers,
            seed=args.seed,
            mixed_precision=not args.no_amp,
            save_every=args.save_every,
            log_every=args.log_every,
        ),
        output_dir=args.output_dir,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        schema_version=(8 if args.goal_representation == "hindsight_geom_profile16"
                        else 7 if args.goal_representation == "hindsight_geom_avg12"
                        else 6 if args.goal_representation == "path_guidance_se2_36"
                        else 5 if args.goal_representation == "holonomic_se2_32"
                        else 4 if is_reference else 3),
        policy_kind=("spatial_hindsight_height_profile_ddpm" if args.goal_representation == "hindsight_geom_profile16"
                     else "spatial_hindsight_geometry_ddpm" if args.goal_representation == "hindsight_geom_avg12"
                     else "holonomic_reference_path_ddpm" if args.goal_representation == "holonomic_se2_32"
                     else "path_guidance_terminal_ddpm" if args.goal_representation == "path_guidance_se2_36"
                     else "spatial_reference_path_ddpm" if is_reference else "spatial_time_preview_ddpm"),
        goal_schema=goal_schema_name(args.goal_representation, reference=is_reference),
    )


def maybe_init_wandb(cfg: TrainConfig):
    if cfg.wandb_project is None:
        return None
    import wandb

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        config=cfg.to_dict(),
    )
    return wandb


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay (DiffuseLoco-style, step-based)."""

    warmup_steps = max(0, min(int(warmup_steps), max(total_steps - 1, 0)))
    cosine_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(cosine_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_policy(cfg: TrainConfig) -> Solo12DiffusionPolicy:
    policy_cfg = Solo12DiffusionPolicyConfig(
        proprio_dim=cfg.model.proprio_dim,
        action_hist_dim=cfg.model.action_hist_dim,
        goal_dim=cfg.model.goal_dim,
        history=cfg.dataset.history,
        prediction_horizon=cfg.dataset.prediction_horizon,
        execution_offset=cfg.dataset.execution_offset,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        p_drop_emb=cfg.model.p_drop_emb,
        p_drop_attn=cfg.model.p_drop_attn,
        separate_goal_conditioning=cfg.model.separate_goal_conditioning,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        variance_type=cfg.diffusion.variance_type,
        clip_sample=cfg.diffusion.clip_sample,
        cfg_dropout_prob=cfg.diffusion.cfg_dropout_prob,
        num_inference_steps=cfg.diffusion.num_inference_steps or cfg.diffusion.num_train_timesteps,
    )
    return Solo12DiffusionPolicy(policy_cfg)


class NoisyHindsightWaypointDataset(Dataset):
    """Training-only jitter for hindsight waypoint observations.

    Each history token gets one bounded XY displacement shared by its three
    intermediate waypoints.  This keeps the guide path coherent while the
    achieved terminal pose (the final five path11 values) and every action
    label remain untouched.  Validation deliberately uses clean conditions.
    """

    def __init__(self, dataset: Dataset, *, std_m: float, clip_m: float) -> None:
        self.dataset = dataset
        self.std_m = float(std_m)
        self.clip_m = float(clip_m)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        goal = item["goal_hist"].clone()
        # path11 = [three waypoint XY pairs, clean terminal XY/Z/yaw/speed].
        noise = torch.randn((goal.shape[0], 1, 2), dtype=goal.dtype) * self.std_m
        if self.clip_m > 0.0:
            norm = torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1.0e-8)
            noise = noise * torch.clamp(self.clip_m / norm, max=1.0)
        goal[:, :6] += noise.expand(-1, 3, -1).reshape(goal.shape[0], 6)
        return {**item, "goal_hist": goal}


@torch.no_grad()
def evaluate(policy: Solo12DiffusionPolicy, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> float:
    policy.eval()
    losses = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        losses.append(float(policy.compute_loss(batch).item()))
    policy.train()
    return float(np.mean(losses)) if losses else float("nan")


def save_checkpoint(
    path: Path,
    *,
    cfg: TrainConfig,
    policy: Solo12DiffusionPolicy,
    ema: EMAModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    normalizer_stats: dict,
    val_loss: float,
    split_manifest: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": cfg.schema_version,
            "policy_kind": cfg.policy_kind,
            "config": cfg.to_dict(),
            "model_state_dict": policy.state_dict(),
            "ema_model_state_dict": ema.averaged_model.state_dict(),
            "noise_scheduler_config": dict(policy.noise_scheduler.config),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "ema_optimization_step": ema.optimization_step,
            "epoch": epoch,
            "global_step": global_step,
            "normalizer_stats": normalizer_stats,
            "val_loss": val_loss,
            "split_manifest": split_manifest,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    set_seed(cfg.optim.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(
        f"[INFO] d_model={cfg.model.d_model} layers={cfg.model.num_layers} "
        f"nhead={cfg.model.nhead} K_train={cfg.diffusion.num_train_timesteps} "
        f"K_infer={cfg.diffusion.num_inference_steps} trajectory={cfg.dataset.prediction_horizon} "
        f"execute_from={cfg.dataset.execution_offset}"
    )

    device = torch.device(args.device)
    dataset = SpatialHindsightDataset(
        cfg.dataset.hdf5_paths,
        history=cfg.dataset.history,
        prediction_horizon=cfg.dataset.prediction_horizon,
        execution_offset=cfg.dataset.execution_offset,
        goal_horizon_steps=cfg.dataset.goal_horizon_steps,
        waypoint_time_offsets_s=cfg.dataset.waypoint_time_offsets_s,
        step_stride=cfg.dataset.step_stride,
        dt=cfg.dataset.dt,
        v_req_clip=cfg.dataset.v_req_clip,
        goal_source=cfg.dataset.goal_source,
        goal_representation=cfg.dataset.goal_representation,
        include_padded_starts=cfg.dataset.include_padded_starts,
        startup_sample_multiplier=cfg.dataset.startup_sample_multiplier,
        symmetry_mode=cfg.dataset.symmetry_mode,
    )
    episode_split = split_episode_indices(len(dataset.demos), cfg.dataset.val_fraction, cfg.optim.seed)
    train_indices = dataset.sample_indices_for_demos(episode_split.train_demo_indices)
    val_indices = dataset.sample_indices_for_demos(episode_split.val_demo_indices)
    normalizer_stats = dataset.build_normalizer_stats(
        episode_split.train_demo_indices,
        max_stats_samples=cfg.dataset.max_stats_samples,
        seed=cfg.optim.seed,
    ).to_dict()
    train_dataset: Dataset = Subset(dataset, train_indices)
    if cfg.dataset.waypoint_noise_std_m > 0.0:
        train_dataset = NoisyHindsightWaypointDataset(
            train_dataset,
            std_m=cfg.dataset.waypoint_noise_std_m,
            clip_m=cfg.dataset.waypoint_noise_clip_m,
        )
    val_dataset = Subset(dataset, val_indices)
    split_manifest = {
        "seed": cfg.optim.seed,
        "val_fraction": cfg.dataset.val_fraction,
        "train": dataset.manifest_for_demos(episode_split.train_demo_indices),
        "validation": dataset.manifest_for_demos(episode_split.val_demo_indices),
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(split_manifest, file, indent=2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=False,
        num_workers=cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
    )

    policy = build_policy(cfg).to(device)
    policy.set_normalizer_stats(normalizer_stats)
    ema = EMAModel(deepcopy(policy), max_value=cfg.optim.ema_decay, power=0.75)
    ema.averaged_model.to(device)
    ema.averaged_model.set_normalizer_stats(normalizer_stats)
    optimizer = policy.configure_optimizers(
        learning_rate=cfg.optim.learning_rate,
        weight_decay=cfg.optim.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.optim.epochs
    lr_scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=cfg.optim.lr_warmup_steps,
        total_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.optim.mixed_precision and device.type == "cuda")
    wandb = maybe_init_wandb(cfg)

    print(
        f"[INFO] demos={len(dataset.demos)} samples={len(dataset)} "
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"goal_source={cfg.dataset.goal_source} padded_starts={cfg.dataset.include_padded_starts} "
        f"symmetry={cfg.dataset.symmetry_mode} steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"lr_warmup={cfg.optim.lr_warmup_steps} device={device}"
    )

    global_step = 0
    best_val = float("inf")
    start_epoch = 1
    if args.resume is not None:
        resume = load_training_checkpoint(args.resume, device, expected_policy_kind=cfg.policy_kind)
        previous_cfg = resume["config"]
        for section in ("dataset", "model", "diffusion"):
            if previous_cfg[section] != cfg.to_dict()[section]:
                raise ValueError(f"Resume {section} config does not match the requested run.")
        policy.load_state_dict(resume["model_state_dict"])
        ema.averaged_model.load_state_dict(resume["ema_model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        lr_scheduler.load_state_dict(resume["lr_scheduler_state_dict"])
        if "scaler_state_dict" in resume:
            scaler.load_state_dict(resume["scaler_state_dict"])
        global_step = int(resume["global_step"])
        start_epoch = int(resume["epoch"]) + 1
        best_val = float(resume.get("val_loss", float("inf")))
        ema.optimization_step = int(resume.get("ema_optimization_step", global_step))
        if start_epoch > cfg.optim.epochs:
            raise ValueError(
                f"Checkpoint completed epoch {start_epoch - 1}, but requested epochs={cfg.optim.epochs}."
            )
        print(f"[INFO] Resuming at epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}")

    for epoch in range(start_epoch, cfg.optim.epochs + 1):
        policy.train()
        epoch_start = time.time()
        train_losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.optim.mixed_precision and device.type == "cuda"):
                loss = policy.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optim.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            ema.step(policy)

            global_step += 1
            train_losses.append(float(loss.item()))
            if global_step % cfg.optim.log_every == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[TRAIN] epoch={epoch} step={global_step} loss={loss.item():.6f} lr={lr:.3e}")
                if wandb is not None:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": global_step})

            # Validation and checkpointing every 5000 steps
            if global_step % 5000 == 0:
                ema.averaged_model.set_normalizer_stats(normalizer_stats)
                val_loss = evaluate(ema.averaged_model, val_loader, device, max_batches=100)
                recent_train_loss = float(np.mean(train_losses[-100:])) if train_losses else float("nan")
                print(f"[VAL-STEP] step={global_step} train_loss_recent={recent_train_loss:.6f} val_loss={val_loss:.6f}")
                if wandb is not None:
                    wandb.log({"val/loss": val_loss, "train/loss_recent": recent_train_loss, "step": global_step})

                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        output_dir / "best.pt",
                        cfg=cfg,
                        policy=policy,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        normalizer_stats=normalizer_stats,
                        val_loss=val_loss,
                        split_manifest=split_manifest,
                    )
                    print(f"[INFO] New best val_loss={val_loss:.6f} saved to best.pt")

                # Save periodic checkpoint every 10000 steps
                if global_step % 10000 == 0:
                    save_checkpoint(
                        output_dir / f"checkpoint_step_{global_step:06d}.pt",
                        cfg=cfg,
                        policy=policy,
                        ema=ema,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        normalizer_stats=normalizer_stats,
                        val_loss=val_loss,
                        split_manifest=split_manifest,
                    )

        ema.averaged_model.set_normalizer_stats(normalizer_stats)
        val_loss = evaluate(ema.averaged_model, val_loader, device, max_batches=100)
        train_loss = float(np.mean(train_losses))
        elapsed = time.time() - epoch_start
        print(f"[EPOCH] {epoch}/{cfg.optim.epochs} train={train_loss:.6f} val={val_loss:.6f} time={elapsed:.1f}s")
        if wandb is not None:
            wandb.log({"epoch": epoch, "train/epoch_loss": train_loss, "val/loss": val_loss, "step": global_step})

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                output_dir / "best.pt",
                cfg=cfg,
                policy=policy,
                ema=ema,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                normalizer_stats=normalizer_stats,
                val_loss=val_loss,
                split_manifest=split_manifest,
            )
        if epoch % cfg.optim.save_every == 0 or epoch == cfg.optim.epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                cfg=cfg,
                policy=policy,
                ema=ema,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                normalizer_stats=normalizer_stats,
                val_loss=val_loss,
                split_manifest=split_manifest,
            )


if __name__ == "__main__":
    main()
