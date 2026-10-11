"""Exhaustively check eligible anchors and smoke-test the real loader/model."""
import argparse
import json
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from prepare import ROOT, INDEX, OUT, verify
from selection import eligible_anchors, HISTORY, HORIZON, DT, LOCAL_SPEED_TICKS

sys.path.insert(0, str(ROOT / 'scripts/diffusion_policy'))
from train.data.dataset import SpatialHindsightDataset
from train.data.episode_split import split_episode_indices
from train.conditioning.geometry import cumulative_xy_lengths
from model.solo12_diffusion_policy import Solo12DiffusionPolicy, Solo12DiffusionPolicyConfig


def main(*, context_only=False):
    manifest = verify()
    report = {'sources': {}, 'all_anchors_valid': True}
    values = {skill: [] for skill in manifest['sources']}
    example = None
    walk_min_local, walk_min_raw, walk_min_command, walk_first_tick = float('inf'), float('inf'), float('inf'), 999
    encoded_walk_min, encoded_pronk_max = float('inf'), 0.
    with h5py.File(INDEX, 'r') as file:
        counts = {skill: [0, 0] for skill in values}
        for name, demo in file['data'].items():
            skill = str(demo.attrs['selection_skill'])
            obs = demo['obs']
            root, command = obs['root_pos_w'][:], obs['command_speed'][:]
            anchors = demo['training_anchor_steps'][:]
            unique = np.unique(anchors)
            if not np.all(np.isin(unique, eligible_anchors(root, command, skill))):
                raise ValueError(f'{name}: ineligible training anchor; abort')
            # Independent numerical check using the trainer's actual float32
            # cumulative geometry, for EVERY retained historical goal token.
            state = np.unique(np.maximum(unique[:, None] - HISTORY + np.arange(HISTORY), 0))
            encoded = cumulative_xy_lengths(root).astype(np.float64)
            goal_speed = ((encoded[state + HORIZON] - encoded[state]) / (HORIZON * DT)).astype(np.float32)
            if skill == 'walk':
                encoded_walk_min = min(encoded_walk_min, float(goal_speed.min()))
                if np.any(goal_speed.astype(np.float64) < .6):
                    raise ValueError(f'{name}: actual trainer goal speed below 0.6')
            elif skill == 'pronk':
                encoded_pronk_max = max(encoded_pronk_max, float(goal_speed.max()))
                if np.any(goal_speed > .5):
                    raise ValueError(f'{name}: actual trainer pronk goal above 0.5')
            counts[skill][0] += len(unique)
            counts[skill][1] += len(anchors)
            xy = root[:, :2].astype(np.float64)
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
            v = (arc[100:] - arc[:-100]) / 2
            values[skill].extend(v[np.maximum(unique - 1, 0)])
            if skill == 'walk':
                local = (arc[LOCAL_SPEED_TICKS:] - arc[:-LOCAL_SPEED_TICKS]) / (LOCAL_SPEED_TICKS * DT)
                raw = np.diff(arc) / DT
                for anchor in unique:
                    start, end = int(anchor) - HISTORY - 1, int(anchor) - 1 + HORIZON
                    walk_min_local = min(walk_min_local, float(local[start:end-LOCAL_SPEED_TICKS+1].min()))
                    walk_min_raw = min(walk_min_raw, float(raw[start:end].min()))
                    walk_min_command = min(walk_min_command, float(np.linalg.norm(command[start:end,:2], axis=1).min()))
                    walk_first_tick = min(walk_first_tick, start)
            if skill == 'walk' and example is None:
                # First entry in the seeded demo order, not a handpicked success.
                example = (name, root, command, unique)
        for skill, (unique, weighted) in counts.items():
            expected = manifest['sources'][skill]
            if unique != expected['unique_anchors'] or weighted != expected['weighted_anchors']:
                raise ValueError(f'{skill}: anchor-count mismatch')
            report['sources'][skill] = {'unique_anchors': unique, 'weighted_anchors': weighted}
    report['walk_context'] = {'min_local_0p20s_mean_speed_mps': walk_min_local,
                              'min_instantaneous_0p02s_speed_mps': walk_min_raw,
                              'min_planar_command_mps': walk_min_command,
                              'earliest_used_physical_tick': walk_first_tick,
                              'padded_walk_starts': 0}
    report['actual_trainer_goal_tokens'] = {'min_walk_speed_mps': encoded_walk_min,
                                           'max_pronk_speed_mps': encoded_pronk_max}
    output = ROOT / 'scripts/diffusion_policy/data/audits/walk_crouch_pronk_speed_partition_v1'
    output.mkdir(parents=True, exist_ok=True)
    colors = {'walk':'#2878B5', 'crouch':'#59A14F', 'pronk':'#F28E2B'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for skill, speed in values.items():
        axes[0].hist(speed, bins=np.linspace(0, 1.5, 76), density=True,
                     histtype='step', color=colors[skill], label=skill, linewidth=1.6)
    axes[0].axvspan(.5, .6, color='.85', alpha=.6)
    axes[0].set(xlabel='Achieved hindsight mean speed [m/s]', ylabel='Density of unique anchors',
                title='Explicit slow / fast demonstration support')
    axes[0].legend()
    name, root, command, anchors = example
    speed = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) / DT
    local = np.convolve(speed, np.ones(LOCAL_SPEED_TICKS) / LOCAL_SPEED_TICKS, mode='valid')
    time = (np.arange(len(local)) + LOCAL_SPEED_TICKS / 2) * DT
    mask = np.zeros(len(local), dtype=bool)
    for anchor in anchors:
        mask[int(anchor)-HISTORY-1:int(anchor)-1+HORIZON-LOCAL_SPEED_TICKS+1] = True
    axes[1].plot(time, local, color='.65', lw=1, label='Original motion (context only)')
    axes[1].plot(time, np.where(mask, local, np.nan), color=colors['walk'], lw=2,
                 label='Physical context of retained walk anchors')
    axes[1].axhline(.6, color='black', ls='--', lw=1, label='0.6 m/s floor')
    axes[1].set(xlabel='Original episode time [s]', ylabel='Local XY speed, 0.20 s mean [m/s]',
                title='Walk: startup excluded, not relabelled fast')
    axes[1].legend(fontsize=7, loc='lower right')
    for axis in axes:
        axis.grid(alpha=.2)
        axis.spines[['top','right']].set_visible(False)
    fig.savefig(output / 'speed_partition_support.png', dpi=180)
    plt.close(fig)
    report['example'] = name
    report['index_sha256'] = manifest['index']['sha256']
    if context_only:
        (output / 'context_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
        return

    # Use the trainer's real indexing, augmentation and normalization. This is
    # one CPU forward/loss calculation, not a local training/optimizer update.
    dataset = SpatialHindsightDataset([str(INDEX)], goal_representation='hindsight_geom_profile16',
                                      include_padded_starts=True, symmetry_mode='quadruped')
    if len(dataset.samples) != 2007000:
        raise ValueError('Loader did not honor the weighted anchor index.')
    split = split_episode_indices(len(dataset.demos), .05, 42)
    for label, ids in (('train', split.train_demo_indices), ('validation', split.val_demo_indices)):
        selected = set(ids)
        counts = {skill: 0 for skill in values}
        for sample in dataset.samples:
            if sample.demo_idx in selected:
                skill = dataset.skill_names[int(dataset.demos[sample.demo_idx].skill_idx[0])]
                counts[skill] += 1
        if not all(counts.values()):
            raise ValueError('Episode split lost an expert.')
        report[label + '_weighted_anchors'] = counts
    stats = dataset.build_normalizer_stats(split.train_demo_indices, max_stats_samples=64, seed=42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    policy = Solo12DiffusionPolicy(Solo12DiffusionPolicyConfig(
        goal_dim=16, d_model=128, nhead=4, num_layers=4))
    policy.set_normalizer_stats(stats.to_dict())
    chosen = []
    for skill in values:
        for i, sample in enumerate(dataset.samples):
            if dataset.skill_names[int(dataset.demos[sample.demo_idx].skill_idx[0])] == skill:
                chosen.append(i * 4)
                break
    batch = {key: torch.stack([dataset[i][key] for i in chosen]) for key in dataset[0]}
    with torch.no_grad():
        loss = policy.compute_loss(batch)
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite smoke-test diffusion loss.')
    if float(batch['goal_hist'][0, :, -1].min()) < .6:
        raise ValueError('Real walk goal tokens below 0.6.')
    report.update(loader_samples=len(dataset), raw_weighted_samples=len(dataset.samples),
                  cpu_smoke_loss=float(loss), local_optimizer_steps=0,
                  episode_split_overlap=len(set(split.train_demo_indices) & set(split.val_demo_indices)))
    (output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-only', action='store_true', help='Recheck selection/plot only; skip the already verified model smoke test.')
    args = parser.parse_args()
    main(context_only=args.context_only)
