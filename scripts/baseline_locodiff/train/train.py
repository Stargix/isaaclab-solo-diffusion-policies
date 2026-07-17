"""Train Solo12 SDE/EDM Diffusion Policy."""

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
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig
    from model.ema_model import EMAModel
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
    from train.data.dataset import LocoDiffCommandSkillDataset
    from train.data.conditioning import CONDITION_MODES, goal_dim_for, policy_kind_for
    from train.data.episode_split import split_episode_indices


def _find_cli_value(args: list[str], flag: str) -> str | None:
    try:
        idx = args.index(flag)
        if idx + 1 < len(args):
            return args[idx + 1]
    except ValueError:
        pass
    return None


def parse_args() -> argparse.Namespace:
    config_path = _find_cli_value(sys.argv[1:], "--config")
    config_defaults = load_training_config_overrides(config_path) if config_path else {}

    parser = argparse.ArgumentParser(description="Train Solo12 SDE/EDM Diffusion Policy.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON file with hyperparameter overrides.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Merged HDF5 files to train on.")
    parser.add_argument("--output_dir", required=True, help="Directory for checkpoints and config.")
    parser.add_argument("--run_name", default="solo12_diffusion_policy", help="Run name for logs.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--history", type=int, default=DATASET_DEFAULTS.history)
    parser.add_argument("--prediction_horizon", type=int, default=DATASET_DEFAULTS.prediction_horizon)
    parser.add_argument("--execution_offset", type=int, default=DATASET_DEFAULTS.execution_offset)
    parser.add_argument("--step_stride", type=int, default=DATASET_DEFAULTS.step_stride, help="Temporal stride to sub-sample step windows.")
    parser.add_argument("--symmetry_mode", choices=["none", "mirror", "quadruped"], default=DATASET_DEFAULTS.symmetry_mode)
    parser.add_argument("--max_stats_samples", type=int, default=DATASET_DEFAULTS.max_stats_samples)
    parser.add_argument("--val_fraction", type=float, default=DATASET_DEFAULTS.val_fraction)
    parser.add_argument(
        "--condition_mode", choices=CONDITION_MODES, default=DATASET_DEFAULTS.condition_mode,
        help="command_skill reproduces the paper; velocity_height is the continuous-height ablation.",
    )

    parser.add_argument("--d_model", type=int, default=MODEL_DEFAULTS.d_model)
    parser.add_argument("--nhead", type=int, default=MODEL_DEFAULTS.nhead)
    parser.add_argument("--num_layers", type=int, default=MODEL_DEFAULTS.num_layers)
    parser.add_argument("--p_drop_emb", type=float, default=MODEL_DEFAULTS.p_drop_emb)
    parser.add_argument("--p_drop_attn", type=float, default=MODEL_DEFAULTS.p_drop_attn)

    parser.add_argument("--num_inference_steps", type=int, default=DIFFUSION_DEFAULTS.num_inference_steps)
    parser.add_argument("--sigma_data", type=float, default=DIFFUSION_DEFAULTS.sigma_data)
    parser.add_argument("--sigma_min", type=float, default=DIFFUSION_DEFAULTS.sigma_min)
    parser.add_argument("--sigma_max", type=float, default=DIFFUSION_DEFAULTS.sigma_max)
    parser.add_argument("--rho", type=float, default=DIFFUSION_DEFAULTS.rho)
    parser.add_argument("--log_sigma_loc", type=float, default=DIFFUSION_DEFAULTS.log_sigma_loc)
    parser.add_argument("--log_sigma_scale", type=float, default=DIFFUSION_DEFAULTS.log_sigma_scale)
    parser.add_argument("--noise_distribution", choices=["log_logistic"], default=DIFFUSION_DEFAULTS.noise_distribution)
    parser.add_argument("--sampler", choices=["euler", "heun"], default=DIFFUSION_DEFAULTS.sampler)
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
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Validate data, one forward/backward pass and three-step sampling, then exit.",
    )

    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_entity", default=None)

    if config_defaults:
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    return args


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
            prediction_horizon=args.prediction_horizon,
            execution_offset=args.execution_offset,
            step_stride=args.step_stride,
            symmetry_mode=args.symmetry_mode,
            max_stats_samples=args.max_stats_samples,
            val_fraction=args.val_fraction,
            condition_mode=args.condition_mode,
        ),
        model=ModelConfig(
            goal_dim=goal_dim_for(args.condition_mode),
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=4 * args.d_model,
            p_drop_emb=args.p_drop_emb,
            p_drop_attn=args.p_drop_attn,
        ),
        diffusion=DiffusionConfig(
            num_inference_steps=args.num_inference_steps,
            sigma_data=args.sigma_data,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            rho=args.rho,
            log_sigma_loc=args.log_sigma_loc,
            log_sigma_scale=args.log_sigma_scale,
            noise_distribution=args.noise_distribution,
            sampler=args.sampler,
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
        policy_kind=policy_kind_for(args.condition_mode),
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
        sigma_data=cfg.diffusion.sigma_data,
        sigma_min=cfg.diffusion.sigma_min,
        sigma_max=cfg.diffusion.sigma_max,
        rho=cfg.diffusion.rho,
        log_sigma_loc=cfg.diffusion.log_sigma_loc,
        log_sigma_scale=cfg.diffusion.log_sigma_scale,
        noise_distribution=cfg.diffusion.noise_distribution,
        sampler=cfg.diffusion.sampler,
        num_inference_steps=cfg.diffusion.num_inference_steps,
    )
    return Solo12DiffusionPolicy(policy_cfg)


@torch.no_grad()
def evaluate(
    policy: Solo12DiffusionPolicy,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
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
            "sde_config": cfg.to_dict()["diffusion"],
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
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
        f"nhead={cfg.model.nhead} trajectory={cfg.dataset.prediction_horizon} "
        f"execute_from={cfg.dataset.execution_offset} solver={cfg.diffusion.sampler} "
        f"solver_steps={cfg.diffusion.num_inference_steps} condition={cfg.dataset.condition_mode}"
    )

    device = torch.device(args.device)
    dataset = LocoDiffCommandSkillDataset(
        cfg.dataset.hdf5_paths,
        history=cfg.dataset.history,
        prediction_horizon=cfg.dataset.prediction_horizon,
        execution_offset=cfg.dataset.execution_offset,
        step_stride=cfg.dataset.step_stride,
        symmetry_mode=cfg.dataset.symmetry_mode,
        condition_mode=cfg.dataset.condition_mode,
    )
    episode_split = split_episode_indices(len(dataset.demos), cfg.dataset.val_fraction, cfg.optim.seed)
    train_indices = dataset.sample_indices_for_demos(episode_split.train_demo_indices)
    val_indices = dataset.sample_indices_for_demos(episode_split.val_demo_indices)
    normalizer_stats = dataset.build_normalizer_stats(
        episode_split.train_demo_indices,
        max_stats_samples=cfg.dataset.max_stats_samples,
        seed=cfg.optim.seed,
    ).to_dict()
    train_dataset = Subset(dataset, train_indices)
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

    preflight_batch = next(iter(train_loader))
    preflight_batch = {key: value.to(device) for key, value in preflight_batch.items()}
    optimizer.zero_grad(set_to_none=True)
    preflight_loss = policy.compute_loss(preflight_batch)
    if not torch.isfinite(preflight_loss):
        raise RuntimeError(f"Non-finite preflight loss: {preflight_loss.item()}")
    preflight_loss.backward()
    if not all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in policy.parameters()):
        raise RuntimeError("Non-finite gradient detected during preflight.")
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        sampled = policy.predict_action_denormalized(
            preflight_batch["proprio_hist"][:2],
            preflight_batch["action_hist"][:2],
            preflight_batch["goal_hist"][:2],
        )
    if sampled.shape != (2, cfg.dataset.prediction_horizon, cfg.model.action_dim):
        raise RuntimeError(f"Unexpected sampled action shape: {tuple(sampled.shape)}")
    if not torch.isfinite(sampled).all():
        raise RuntimeError("Non-finite action detected during preflight sampling.")
    print(f"[PREFLIGHT] loss={preflight_loss.item():.6f} shapes and gradients OK")
    if args.preflight_only:
        print("[PREFLIGHT] complete; training was not started.")
        return

    print(
        f"[INFO] demos={len(dataset.demos)} samples={len(dataset)} "
        f"train={len(train_dataset)} val={len(val_dataset)} "
        f"symmetry={cfg.dataset.symmetry_mode} steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"lr_warmup={cfg.optim.lr_warmup_steps} device={device}"
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
                epoch=epoch,
                global_step=global_step,
                normalizer_stats=normalizer_stats,
                val_loss=val_loss,
                split_manifest=split_manifest,
            )


if __name__ == "__main__":
    main()
