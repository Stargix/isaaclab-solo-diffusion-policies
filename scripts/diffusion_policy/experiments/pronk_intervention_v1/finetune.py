"""Fixed-budget nominal control versus takeover-recovery fine-tuning.

Reuse the original network, diffusion loss, normalization, history, augmentation
and EMA implementation. Only training distribution changes between arms.
Both start from the frozen nominal EMA, with new Adam and EMA schedules.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
import time
import numpy as np
import torch
import h5py

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import (
    ROOT, BASE_SHA, NOMINAL, DEFAULT_OUTPUT, load_checkpoint, sha256)
from scripts.diffusion_policy.train.data.dataset import SpatialHindsightDataset
from scripts.diffusion_policy.train.data.normalization import NormalizerStats
from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
from scripts.diffusion_policy.model.ema_model import EMAModel
from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint


def make_dataset(path, ck):
    c=deepcopy(ck['config']['dataset'])
    for k in ('hdf5_paths','max_stats_samples','val_fraction','waypoint_noise_std_m','waypoint_noise_clip_m'):
        c.pop(k,None)
    return SpatialHindsightDataset([str(path)],**c)


def cache(ds, demo_indices, count, rng):
    indices=ds.sample_indices_for_demos(demo_indices)
    if not indices:raise ValueError('Empty episode split')
    selected=rng.choice(indices,size=min(count,len(indices)),replace=False)
    samples=[ds[int(i)] for i in selected]
    return {k:torch.stack([s[k] for s in samples]) for k in samples[0]}


def retain_representable_targets(ds, stats):
    """Keep raw recordings; explicitly mask rare unsupported target windows.

    Both past and future denoising targets are checked. Never clip labels or
    silently widen the frozen sampler. Reject a material support mismatch.
    """
    kept=[]
    excluded=[]
    for sample in ds.samples:
        demo=ds.demos[sample.demo_idx]
        lo=max(0,sample.anchor_step-ds.execution_offset)
        hi=min(len(demo.actions),sample.anchor_step+ds.future_horizon)
        target=demo.actions[lo:hi]
        if np.any(target<stats.action.min-1e-5) or np.any(target>stats.action.max+1e-5):
            excluded.append(dict(demo=demo.demo_name,anchor=int(sample.anchor_step)))
        else:
            kept.append(sample)
    audit=dict(original_anchors=len(ds.samples),excluded_anchors=excluded,
               excluded_fraction=len(excluded)/max(1,len(ds.samples)))
    if audit['excluded_fraction']>.01:
        raise ValueError('More than 1% of target windows exceed the frozen action range; do not mask a material support mismatch.')
    ds.samples=kept
    return audit


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=('nominal','recovery'),required=True)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'best.pt')
    ap.add_argument('--collection',type=Path,default=DEFAULT_OUTPUT)
    ap.add_argument('--preflight_only',action='store_true')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    report=json.loads((args.collection/'collection.json').read_text())
    if not report['gate_passed'] or len({r['cohort'] for r in report['attempts']}) != 4:
        raise ValueError('Complete four-cohort collection with a passed gate is required.')
    ck=load_checkpoint(args.checkpoint,BASE_SHA)
    output=args.collection/('fit_'+args.arm)
    if output.exists():raise FileExistsError(output)
    recovery=make_dataset(args.collection/'recovery.hdf5',ck)
    with h5py.File(args.collection/'recovery.hdf5','r') as f:
        train_recovery=[i for i,d in enumerate(recovery.demos) if int(f['data'][d.demo_name].attrs['cohort'])<3]
        val_recovery=[i for i,d in enumerate(recovery.demos) if int(f['data'][d.demo_name].attrs['cohort'])==3]
    if len(train_recovery)<30 or len(val_recovery)<10:
        raise ValueError('Insufficient train/recovery-validation episodes; do not silently reuse training episodes.')
    stats=NormalizerStats.from_dict(ck['normalizer_stats'])
    range_audit=retain_representable_targets(recovery,stats)
    if args.preflight_only:
        print(json.dumps(dict(preflight='passed',recovery_train_episodes=len(train_recovery),
                              recovery_validation_episodes=len(val_recovery),action_range=range_audit)),
              flush=True)
        return
    output.mkdir()
    rng=np.random.default_rng(642)
    train_manifest=ck['split_manifest']['train']
    val_manifest=ck['split_manifest']['validation']
    train_names=[train_manifest[int(i)]['demo_name'] for i in rng.choice(len(train_manifest),32,replace=False)]
    val_names=[val_manifest[int(i)]['demo_name'] for i in rng.choice(len(val_manifest),8,replace=False)]
    subset=output/'nominal_subset.hdf5'
    with h5py.File(NOMINAL,'r') as source,h5py.File(subset,'w') as dest:
        group=dest.create_group('data')
        for k,v in source['data'].attrs.items():group.attrs[k]=v
        for n in train_names+val_names:source.copy(source['data'][n],group,name=n)
    nominal=make_dataset(subset,ck)
    nominal_range_audit=retain_representable_targets(nominal,stats)
    trn=[i for i,d in enumerate(nominal.demos) if d.demo_name in train_names]
    val=[i for i,d in enumerate(nominal.demos) if d.demo_name in val_names]
    provenance=dict(arm=args.arm,base_sha256=BASE_SHA,collection_sha256=sha256(args.collection/'recovery.hdf5'),
                    fixed_normalizer=True,initialization='original EMA',updates=600,batch_size=128,
                    recovery_fraction=0. if args.arm=='nominal' else .5,lr=1e-5,warmup_updates=50,
                    original_train_episodes=train_names,original_validation_episodes=val_names,
                    recovery_train_episodes=[recovery.demos[i].demo_name for i in train_recovery],
                    recovery_validation_episodes=[recovery.demos[i].demo_name for i in val_recovery],
                    primary_checkpoint='last.pt at fixed update 600; not selected by physical evaluation')
    provenance['action_range_audit']=dict(recovery=range_audit,nominal=nominal_range_audit,
        note='Rare target-window masking added after the range audit; exploratory, not a preregistered paper comparison. No raw recordings or labels modified.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(provenance),flush=True)
    torch.manual_seed(42)
    device=torch.device(args.device)
    model=Solo12DiffusionPolicy(model_config_from_checkpoint(ck['config'])).to(device)
    model.load_state_dict(ck['ema_model_state_dict']);model.set_normalizer_stats(stats)
    ema=EMAModel(deepcopy(model),max_value=.9999,power=.75)
    opt=model.configure_optimizers(learning_rate=1e-5,weight_decay=.001)
    old=cache(nominal,trn,2048,rng)
    new=cache(recovery,train_recovery,2048,rng)
    validations={'nominal':cache(nominal,val,128,rng),'recovery':cache(recovery,val_recovery,128,rng)}
    start=time.monotonic()
    with (output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,601):
            model.train()
            n=128 if args.arm=='nominal' else 64
            old_idx=torch.randint(len(old['actions']),(n,))
            batch={k:v[old_idx] for k,v in old.items()}
            if args.arm=='recovery':
                new_idx=torch.randint(len(new['actions']),(64,))
                batch={k:torch.cat((v,new[k][new_idx])) for k,v in batch.items()}
            batch={k:v.to(device) for k,v in batch.items()}
            for group in opt.param_groups:group['lr']=1e-5*min(step/50.,1.)
            opt.zero_grad(set_to_none=True)
            loss=model.compute_loss(batch)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step();ema.step(model)
            if step%100==0 or step==1:
                with torch.inference_mode(),torch.random.fork_rng(devices=[device.index or 0]):
                    torch.manual_seed(842)
                    values={name:float(ema.averaged_model.compute_loss({k:v.to(device) for k,v in b.items()})) for name,b in validations.items()}
                row=dict(update=step,train_loss=float(loss.detach()),validation=values,elapsed_s=time.monotonic()-start)
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    config=deepcopy(ck['config'])
    config['dataset']['hdf5_paths']=[str(subset.resolve()),str((args.collection/'recovery.hdf5').resolve())]
    config['output_dir']=str(output.resolve())
    config['optim'].update(learning_rate=1e-5,epochs=0,lr_warmup_steps=50,batch_size=128)
    saved={k:ck[k] for k in ('schema_version','policy_kind','noise_scheduler_config')}
    saved.update(config=config,model_state_dict=model.state_dict(),ema_model_state_dict=ema.averaged_model.state_dict(),
                 normalizer_stats=ck['normalizer_stats'],epoch=0,global_step=600,
                 experiment=provenance,val_loss=sum(values.values())/len(values))
    torch.save(saved,output/'last.pt')
    print('CHECKPOINT',output/'last.pt',flush=True)


if __name__=='__main__':main()
