"""Seal a balanced eligible-anchor index; never alter the original episodes."""
import argparse
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'walk_crouch_pronk_v1'))
from prepare_data import PROTOCOL, OUT as SOURCE_OUT, sha256, verify_artifacts
from selection import (eligible_anchors, WALK_MIN_MPS, PRONK_MAX_MPS,
                       HISTORY, HORIZON, FUTURE_ACTIONS, DT, LOCAL_SPEED_TICKS, SETTLING_TICKS)

OUT = ROOT / 'scripts/diffusion_policy/data/datasets/walk_crouch_pronk_speed_partition_v1'
INDEX = OUT / 'walk_crouch_pronk_speed_partition.hdf5'
CONTRACT = json.loads((HERE / 'selection_contract.json').read_text(encoding='utf-8'))


def validate_contract():
    expected = {'walk_min_achieved_speed_mps': WALK_MIN_MPS,
                'pronk_max_hindsight_speed_mps': PRONK_MAX_MPS,
                'local_speed_window_ticks': LOCAL_SPEED_TICKS,
                'walk_settling_ticks_after_command_change': SETTLING_TICKS,
                'history_ticks': HISTORY, 'hindsight_ticks': HORIZON,
                'future_action_ticks': FUTURE_ACTIONS, 'control_dt_s': DT,
                'walk_near_rest_exclusion_mps': .02,
                'check_delayed_action_history_extra_tick': True,
                'walk_padded_starts': False, 'pronk_crouch_padded_starts': True}
    if any(CONTRACT[key] != value for key, value in expected.items()):
        raise ValueError('Declared selection contract does not match the implemented selector.')


def verify():
    validate_contract()
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(HERE / 'selection_contract.json') != manifest['selection_contract_sha256']:
        raise ValueError('Selection contract changed after preparation.')
    if sha256(ROOT / PROTOCOL['train_config']) != manifest['train_config_sha256']:
        raise ValueError('Frozen Phase-A training configuration changed.')
    for item in [manifest['index'], *manifest['sources'].values()]:
        if sha256(ROOT / item['path']) != item['sha256']:
            raise ValueError(f'Artifact changed: {item["path"]}')
    if set(manifest['sources']) != {'walk', 'crouch', 'pronk'}:
        raise ValueError('Wrong expert set.')
    if {v['weighted_anchors'] for v in manifest['sources'].values()} != {669000}:
        raise ValueError('Wrong balanced epoch budget.')
    return manifest


def prepare():
    validate_contract()
    if INDEX.exists() or (OUT / 'manifest.json').exists():
        raise FileExistsError('Prepared index exists; verify it, never overwrite.')
    original = verify_artifacts()
    pools = {}
    summaries = {}
    for skill, item in original['sources'].items():
        print(f'Selecting {skill}...', flush=True)
        entries, speeds = [], []
        with h5py.File(ROOT / item['path'], 'r') as file:
            for name in sorted(file['data'], key=lambda n: int(n.split('_')[-1])):
                obs = file['data'][name]['obs']
                root, command = obs['root_pos_w'][:], obs['command_speed'][:]
                anchors = eligible_anchors(root, command, skill)
                if not len(anchors):
                    continue
                entries.append((name, anchors))
                # These are latest historical goal speeds, not command speeds.
                xy = root[:, :2].astype(np.float64)
                arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
                v = (arc[100:] - arc[:-100]) / 2
                speeds.extend(v[np.maximum(anchors - 1, 0)])
        if not entries:
            raise ValueError(f'No eligible examples for {skill}; do not relax thresholds silently.')
        pools[skill] = entries
        summaries[skill] = {'eligible_demos': len(entries),
                           'unique_anchors': sum(len(a) for _, a in entries),
                           'latest_goal_speed_mps': dict(zip(('min', 'p05', 'median', 'p95', 'max'),
                              (float(v) for v in np.percentile(speeds, [0, 5, 50, 95, 100]))))}
    target = max(v['unique_anchors'] for v in summaries.values())
    if target != 669000:
        raise ValueError('Unexpected original source budget.')
    entries = []
    rng = np.random.default_rng(CONTRACT['balance_seed'])
    for skill, pool in pools.items():
        total = summaries[skill]['unique_anchors']
        repeats, remainder = divmod(target, total)
        weights = np.full(total, repeats, dtype=np.int16)
        if remainder:
            weights[rng.choice(total, remainder, replace=False)] += 1
        cursor = 0
        for name, anchors in pool:
            weighted = np.repeat(anchors, weights[cursor:cursor + len(anchors)])
            entries.append((skill, name, weighted.astype(np.int16)))
            cursor += len(anchors)
        summaries[skill].update(weighted_anchors=target, min_repeats=repeats,
                                max_repeats=repeats + int(remainder > 0))
    order = np.random.default_rng(CONTRACT['demo_shuffle_seed']).permutation(len(entries))
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(INDEX, 'x') as output:
        data = output.create_group('data')
        data.attrs.update(skill_names=np.asarray(['walk', 'crouch', 'pronk'], dtype='S32'),
                          control_rate_hz=50., condition_schema='velocity_xyyaw_plus_desired_base_height_v1',
                          convention='aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]; last_action[t] = actions[t-1] (last_action[0] = 0).',
                          storage_mode='relative_external_links',
                          anchor_selection_schema='explicit_weighted_training_anchors_v1')
        data.attrs['linked_sources_json'] = json.dumps([
            {'path': Path(os.path.relpath(ROOT / item['path'], OUT)).as_posix(), 'sha256': item['sha256']}
            for item in original['sources'].values()], sort_keys=True)
        for number, position in enumerate(order):
            skill, source_name, anchors = entries[int(position)]
            group = data.create_group(f'demo_{number}')
            relative = Path(os.path.relpath(ROOT / original['sources'][skill]['path'], OUT)).as_posix()
            for key in ('obs', 'actions', 'dones'):
                group[key] = h5py.ExternalLink(relative, f'/data/{source_name}/{key}')
            group['skill_idx'] = np.full(299, ('walk', 'crouch', 'pronk').index(skill), dtype=np.int8)
            group['training_anchor_steps'] = anchors
            group.attrs.update(num_samples=299, source_demo_name=source_name, selection_skill=skill)
    manifest = {'experiment': CONTRACT['experiment'], 'selection_contract': CONTRACT,
                'selection_contract_sha256': sha256(HERE / 'selection_contract.json'),
                'train_config_sha256': original['train_config_sha256'],
                'sources': {k: {'path': v['path'], 'sha256': v['sha256'], **summaries[k]}
                            for k, v in original['sources'].items()},
                'index': {'path': INDEX.relative_to(ROOT).as_posix(), 'sha256': sha256(INDEX)},
                'weighted_anchors': target * 3, 'unique_anchors': sum(v['unique_anchors'] for v in summaries.values())}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify()
        print('Partition index, sources, contract and configuration hashes verified.')
    else:
        prepare()
