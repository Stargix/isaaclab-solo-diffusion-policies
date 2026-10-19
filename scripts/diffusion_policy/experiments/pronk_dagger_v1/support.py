"""Explicit sampler revision and non-destructive action support diagnostics.

Min/max normalization is affine, not a physical actuator limit. DDPM's optional
clean-sample clipping imposes that range; disabling it allows real corrective
labels outside the original dataset range without changing their normalization.
Only this isolated experiment uses the revised sampler, in both training arms.
"""
from copy import deepcopy
import json
from pathlib import Path

import h5py
import numpy as np


PROTOCOL = 'pronk_dagger_v1_fixed_exposure_v3'


def unclipped_checkpoint(checkpoint):
    result = dict(checkpoint)
    result['config'] = deepcopy(checkpoint['config'])
    result['config']['diffusion']['clip_sample'] = False
    result['noise_scheduler_config'] = dict(checkpoint['noise_scheduler_config'])
    result['noise_scheduler_config']['clip_sample'] = False
    # diffusers thresholding is a separate output clipping mechanism.
    if result['noise_scheduler_config'].get('thresholding',False):
        raise ValueError('Dynamic thresholding is not supported by this sampler revision')
    return result


def audit_action_support(dataset, stats, path):
    """Report overlapping windows separately from independent frame/components.

    Never remove windows, alter labels or infer safety from a data percentile.
    Counts concern raw, unaugmented recordings; all existing symmetry transforms
    remain enabled in training. The affine normalizer remains exactly frozen.
    """
    lower = np.asarray(stats.action.min)
    upper = np.asarray(stats.action.max)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(upper<=lower):
        raise ValueError('Invalid frozen action normalization range')
    data=[]
    for demo in dataset.demos:
        actions=np.asarray(demo.actions)
        if not np.isfinite(actions).all():
            raise ValueError(f'Nonfinite recorded action in {demo.demo_name}')
        bad=(actions<lower-1e-5)|(actions>upper+1e-5)
        frame_bad=bad.any(axis=1)
        data.append(dict(bad=bad,prefix=np.r_[0,np.cumsum(frame_bad)],
                         touched=np.zeros(len(actions),dtype=bool)))
    audit=dict(windows=len(dataset.samples),past_only_windows=0,future_only_windows=0,
               past_and_future_windows=0,early_windows=0,early_future_outside_windows=0,
               excluded_windows=0,independent_outside_frames=0,outside_components=0,
               expert_outside_frames=0,student_outside_frames=0,unknown_source_outside_frames=0,
               maximum_raw_excess_by_joint=np.zeros_like(lower,dtype=float).tolist(),
               per_demo=[],scope='raw unaugmented target windows; no filtering')
    with h5py.File(path,'r') as f:
        takeovers={i:f['data'][demo.demo_name].attrs.get('takeover_step')
                   for i,demo in enumerate(dataset.demos)}
        for sample in dataset.samples:
            d=data[sample.demo_idx]
            anchor=sample.anchor_step
            lo=max(0,anchor-dataset.execution_offset)
            hi=min(len(d['touched']),anchor+dataset.future_horizon)
            past=d['prefix'][anchor]-d['prefix'][lo]>0
            future=d['prefix'][hi]-d['prefix'][anchor]>0
            audit['past_only_windows']+=int(past and not future)
            audit['future_only_windows']+=int(future and not past)
            audit['past_and_future_windows']+=int(past and future)
            d['touched'][lo:hi]=True
            takeover=takeovers[sample.demo_idx]
            early=takeover is not None and takeover<=anchor<takeover+16
            audit['early_windows']+=int(early)
            audit['early_future_outside_windows']+=int(early and future)
        max_excess=np.zeros_like(lower,dtype=float)
        for demo,d in zip(dataset.demos,data):
            indices=np.flatnonzero(d['bad'].any(axis=1)&d['touched'])
            audit['independent_outside_frames']+=len(indices)
            audit['outside_components']+=int(d['bad'][d['touched']].sum())
            if len(indices):
                actions=demo.actions[indices]
                max_excess=np.maximum(max_excess,np.maximum(lower-actions,actions-upper).max(axis=0))
                group=f['data'][demo.demo_name]
                if 'diagnostics/expert_control' in group:
                    expert=group['diagnostics/expert_control'][:].astype(bool)
                    audit['expert_outside_frames']+=int(expert[indices].sum())
                    audit['student_outside_frames']+=int((~expert[indices]).sum())
                else:
                    audit['unknown_source_outside_frames']+=len(indices)
                audit['per_demo'].append(dict(demo=demo.demo_name,outside_frame_indices=indices.tolist()))
    audit['maximum_raw_excess_by_joint']=max_excess.tolist()
    audit['outside_window_fraction']=(audit['past_only_windows']+audit['future_only_windows']+
                                      audit['past_and_future_windows'])/max(1,audit['windows'])
    return audit


def validate_reused_collection(directory, base_sha, expert_sha):
    """Accept only a complete round-zero collection from the frozen original prior.

    Reuse is explicit and read-only; neither the failed run nor its outputs are
    overwritten. The runner records both file hashes before beginning fitting.
    """
    from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import gate_passes
    directory=Path(directory).resolve()
    report=json.loads((directory/'collection.json').read_text(encoding='utf-8'))
    if (report.get('base_sha256')!=base_sha or report.get('expert_sha256')!=expert_sha
            or report.get('dagger_round')!=0 or report.get('seed')!=443
            or report.get('mode')!='takeover_collection'):
        raise ValueError('Reused collection does not match the frozen round-zero inputs/protocol')
    rows=report.get('attempts',[])
    if (len(rows)!=72 or {(r['cohort'],r['env']) for r in rows}
            != {(c,i) for c in range(4) for i in range(18)}
            or not report.get('gate_passed')
            or not gate_passes([r for r in rows if r['cohort']==0])):
        raise ValueError('Reused collection is incomplete or its physical gate did not pass')
    accepted=[r for r in rows if r.get('saved')]
    if len({r['demo'] for r in accepted})!=len(accepted):
        raise ValueError('Duplicate saved episode names')
    hdf5=directory/'recovery.hdf5'
    with h5py.File(hdf5,'r') as f:
        if set(f['data'])!={r['demo'] for r in accepted}:
            raise ValueError('Collection report does not match recorded episodes')
        for row in accepted:
            if not (row['survived'] and row['cycles']>=2 and abs(row['mean_vx']-row['speed'])<=.10):
                raise ValueError('Saved episode did not meet the physical acceptance criterion')
            demo=f['data'][row['demo']]
            expected_start=0 if row['env']%2==0 else 25
            expected_takeover=expected_start+(20,30,40)[(row['env']%6)//2]
            if (row.get('student_start_step')!=expected_start or row['takeover_step']!=expected_takeover
                    or int(demo.attrs['cohort'])!=row['cohort']
                    or int(demo.attrs['takeover_step'])!=expected_takeover):
                raise ValueError('Reused student exposure schedule does not match round zero')
            actions=demo['actions'][:]
            control=demo['diagnostics/expert_control'][:].astype(bool)
            expected=(np.arange(300)<expected_start)|(np.arange(300)>=expected_takeover)
            if actions.shape!=(300,12) or not np.isfinite(actions).all() or not np.array_equal(control,expected):
                raise ValueError('Invalid recorded actions or expert-control mask')
            anchors=demo['training_anchor_steps'][:]
            if not len(anchors) or any(t<0 or t+100>300 or not control[t:t+100].all() for t in anchors):
                raise ValueError('Invalid expert-controlled hindsight anchors')
    return hdf5


def validate_round0_fits(directory, collection, original, base_sha, nominal_sha):
    """Reuse completed matched round-zero fits, never unfinished optimizer state."""
    from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import load_checkpoint, sha256
    from scripts.diffusion_policy.experiments.pronk_dagger_v1.protocol import UPDATES, BATCH_SIZE, LEARNING_RATE
    directory=Path(directory).resolve()
    collection_sha=sha256(collection)

    def equal_stats(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            return a.keys()==b.keys() and all(equal_stats(a[k],b[k]) for k in a)
        return np.array_equal(np.asarray(a),np.asarray(b))

    expected_diffusion=deepcopy(original['config']['diffusion'])
    expected_diffusion['clip_sample']=False
    paths={}
    for arm in ('recovery','nominal'):
        path=directory/f'round_0_{arm}'/'last.pt'
        ck=load_checkpoint(path)
        experiment=ck.get('experiment',{})
        if (experiment.get('protocol') not in ('pronk_dagger_v1_support_v2',PROTOCOL)
                or experiment.get('arm')!=arm or experiment.get('parent_sha256')!=base_sha
                or experiment.get('updates')!=UPDATES or experiment.get('lr')!=LEARNING_RATE
                or experiment.get('batch_size')!=BATCH_SIZE
                or experiment.get('clip_sample') is not False
                or experiment.get('fixed_normalizer') is not True
                or experiment.get('nominal_sha256')!=nominal_sha):
            raise ValueError(f'Cannot reuse incomplete or incompatible round-zero fit: {path}')
        audits=experiment.get('collection_audits',[])
        if len(audits)!=1 or audits[0].get('sha256')!=collection_sha:
            raise ValueError(f'Round-zero fit used different correction data: {path}')
        if (ck.get('schema_version')!=original.get('schema_version')
                or ck.get('policy_kind')!=original.get('policy_kind')
                or ck['config']['model']!=original['config']['model']
                or ck['config']['diffusion']!=expected_diffusion
                or ck['noise_scheduler_config'].get('clip_sample') is not False
                or not equal_stats(ck['normalizer_stats'],original['normalizer_stats'])):
            raise ValueError(f'Round-zero model/normalization/sampler mismatch: {path}')
        dataset={k:v for k,v in ck['config']['dataset'].items() if k!='hdf5_paths'}
        expected_dataset={k:v for k,v in original['config']['dataset'].items() if k!='hdf5_paths'}
        if dataset!=expected_dataset:
            raise ValueError(f'Round-zero conditioning/history mismatch: {path}')
        # The next fit loads this EMA strictly; verify tensor validity before reuse.
        import torch
        expected=original['ema_model_state_dict']
        actual=ck['ema_model_state_dict']
        if (actual.keys()!=expected.keys() or any(actual[k].shape!=expected[k].shape
                or not torch.isfinite(actual[k]).all() for k in actual)):
            raise ValueError(f'Invalid round-zero EMA weights: {path}')
        paths[arm]=path
    return paths


if __name__=='__main__':
    import argparse
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
    from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import load_checkpoint, sha256
    from scripts.diffusion_policy.experiments.pronk_intervention_v1.finetune import make_dataset
    from scripts.diffusion_policy.train.data.normalization import NormalizerStats
    parser=argparse.ArgumentParser(description='Read-only support audit; no Kit, filtering or training.')
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--collection',type=Path,required=True,help='Recovery HDF5 file')
    parser.add_argument('--output',type=Path,help='Optional new JSON file; refuses overwrite')
    args=parser.parse_args()
    checkpoint=load_checkpoint(args.checkpoint)
    dataset=make_dataset(args.collection,checkpoint)
    audit=audit_action_support(dataset,NormalizerStats.from_dict(checkpoint['normalizer_stats']),args.collection)
    audit.update(checkpoint_sha256=sha256(args.checkpoint),collection_sha256=sha256(args.collection))
    rendered=json.dumps(audit,indent=2)
    if args.output is not None:
        with args.output.open('x',encoding='utf-8') as output:
            output.write(rendered+'\n')
    print(rendered)
