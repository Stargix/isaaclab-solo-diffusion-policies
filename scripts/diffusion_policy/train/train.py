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
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.ema_model import EMAModel
    from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
    from train.config import DatasetConfig, DiffusionConfig, ModelConfig, OptimConfig, TrainConfig
    from train.dataset import LocomotionHindsightDataset
else:  # pragma: no cover
    from ..model.ema_model import EMAModel
    from ..model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
    from .config import DatasetConfig, DiffusionConfig, ModelConfig, OptimConfig, TrainConfig
    from .dataset import LocomotionHindsightDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Solo12 Diffusion Policy.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Merged HDF5 files to train on.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and config.")
    parser.add_argument("--run_name", default="solo12_diffusion_policy", help="Run name for logs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--action_horizon", type=int, default=4)
    parser.add_argument("--min_segment_steps", type=int, default=50)
    parser.add_argument("--max_segment_steps", type=int, default=150)
    parser.add_argument("--segment_stride", type=int, default=10)
    parser.add_argument("--v_req_clip", type=float, default=2.0)
    parser.add_argument("--symmetry_mode", choices=["none", "mirror", "quadruped"], default="quadruped")
    parser.add_argument("--val_fraction", type=float, default=0.05)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--beta_schedule", default="squaredcos_cap_v2")
    parser.add_argument("--cfg_dropout_prob", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision.")

    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        dataset=DatasetConfig(
            hdf5_paths=args.datasets,
            history=args.history,
            action_horizon=args.action_horizon,
            min_segment_steps=args.min_segment_steps,
            max_segment_steps=args.max_segment_steps,
            segment_stride=args.segment_stride,
            v_req_clip=args.v_req_clip,
            symmetry_mode=args.symmetry_mode,
            val_fraction=args.val_fraction,
        ),
        model=ModelConfig(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=4 * args.d_model,
            dropout=args.dropout,
        ),
        diffusion=DiffusionConfig(
            num_train_timesteps=args.diffusion_steps,
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


def build_policy(cfg: TrainConfig) -> Solo12DiffusionPolicy:
    policy_cfg = Solo12DiffusionPolicyConfig(
        history=cfg.dataset.history,
        action_horizon=cfg.dataset.action_horizon,
        d_model=cfg.model.d_model,
        nhead=cfg.model.nhead,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        num_train_timesteps=cfg.diffusion.num_train_timesteps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        beta_schedule=cfg.diffusion.beta_schedule,
        prediction_type=cfg.diffusion.prediction_type,
        variance_type=cfg.diffusion.variance_type,
        clip_sample=cfg.diffusion.clip_sample,
        cfg_dropout_prob=cfg.diffusion.cfg_dropout_prob,
    )
    return Solo12DiffusionPolicy(policy_cfg)


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
    epoch: int,
    global_step: int,
    normalizer_stats: dict,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": cfg.to_dict(),
            "model_state_dict": policy.state_dict(),
            "ema_model_state_dict": ema.averaged_model.state_dict(),
            "noise_scheduler_config": dict(policy.noise_scheduler.config),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "normalizer_stats": normalizer_stats,
            "val_loss": val_loss,
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

    device = torch.device(args.device)
    dataset = LocomotionHindsightDataset(
        cfg.dataset.hdf5_paths,
        history=cfg.dataset.history,
        action_horizon=cfg.dataset.action_horizon,
        min_segment_steps=cfg.dataset.min_segment_steps,
        max_segment_steps=cfg.dataset.max_segment_steps,
        segment_stride=cfg.dataset.segment_stride,
        dt=cfg.dataset.dt,
        v_req_clip=cfg.dataset.v_req_clip,
        symmetry_mode=cfg.dataset.symmetry_mode,
        max_stats_samples=cfg.dataset.max_stats_samples,
    )
    normalizer_stats = dataset.get_normalizer_stats().to_dict()

    val_size = max(1, int(len(dataset) * cfg.dataset.val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.optim.seed),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.optim.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
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
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optim.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.optim.mixed_precision and device.type == "cuda")
    wandb = maybe_init_wandb(cfg)

    print(
        f"[INFO] demos={len(dataset.demos)} samples={len(dataset)} train={train_size} val={val_size} "
        f"symmetry={cfg.dataset.symmetry_mode} scheduler=diffusers device={device}"
    )

    global_step = 0
    best_val = float("inf")
    for epoch in range(1, cfg.optim.epochs + 1):
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
                        epoch=epoch,
                        global_step=global_step,
                        normalizer_stats=normalizer_stats,
                        val_loss=val_loss,
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
                        epoch=epoch,
                        global_step=global_step,
                        normalizer_stats=normalizer_stats,
                        val_loss=val_loss,
                    )

        lr_scheduler.step()
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
                epoch=epoch,
                global_step=global_step,
                normalizer_stats=normalizer_stats,
                val_loss=val_loss,
            )
        if epoch % cfg.optim.save_every == 0 or epoch == cfg.optim.epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                cfg=cfg,
                policy=policy,
                ema=ema,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                global_step=global_step,
                normalizer_stats=normalizer_stats,
                val_loss=val_loss,
            )


if __name__ == "__main__":
    main()
