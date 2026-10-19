"""Full-window aggregated imitation; fixed budget and frozen normalization."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import load_checkpoint, sha256
from scripts.diffusion_policy.experiments.pronk_intervention_v1.finetune import make_dataset
from scripts.diffusion_policy.experiments.pronk_dagger_v1.support import PROTOCOL, audit_action_support, unclipped_checkpoint
from scripts.diffusion_policy.experiments.pronk_dagger_v1.protocol import UPDATES, BATCH_SIZE, LEARNING_RATE, early_anchor
from scripts.diffusion_policy.model.solo12_diffusion_policy import Solo12DiffusionPolicy
from scripts.diffusion_policy.model.ema_model import EMAModel
from scripts.diffusion_policy.train.data.normalization import NormalizerStats
from scripts.dppo_diffusion_rl.checkpointing import model_config_from_checkpoint


def batch_from(pool, count, rng, device):
    """Sample fresh windows from complete pools, including existing augmentation."""
    if not pool:
        raise ValueError('Empty sampling pool')
    items = []
    for _ in range(count):
        ds, indices = pool[int(rng.integers(len(pool)))]
        items.append(ds[int(indices[int(rng.integers(len(indices)))])])
    return {k: torch.stack([x[k] for x in items]).to(device) for k in items[0]}


def recovery_pools(paths, ck, stats):
    train, early, validation, audits = [], [], [], []
    for path in paths:
        ds = make_dataset(path, ck)
        audits.append(dict(path=str(path), sha256=sha256(path), **audit_action_support(ds,stats,path)))
        with h5py.File(path, 'r') as f:
            tr = [i for i,d in enumerate(ds.demos) if int(f['data'][d.demo_name].attrs['cohort']) < 3]
            va = [i for i,d in enumerate(ds.demos) if int(f['data'][d.demo_name].attrs['cohort']) == 3]
            takeover = {i:int(f['data'][d.demo_name].attrs['takeover_step']) for i,d in enumerate(ds.demos)}
        if len(tr) < 30 or len(va) < 10:
            raise ValueError('Need >=30 training and >=10 held-out accepted episodes per collection')
        ti = ds.sample_indices_for_demos(tr)
        vi = ds.sample_indices_for_demos(va)
        ei = [i for i in ti if early_anchor(ds.samples[i // ds._symmetry_count].anchor_step,
              takeover[ds.samples[i // ds._symmetry_count].demo_idx])]
        if not ei or not vi:
            raise ValueError('Missing early corrective windows or validation')
        train.append((ds, ti)); early.append((ds, ei)); validation.append((ds, vi))
    return train, early, validation, audits


def validation_metrics(model, batches):
    """Denoising loss plus actually sampled next-four-action error (raw units)."""
    model.eval()
    values = {}
    with torch.inference_mode():
        for name, batch in batches.items():
            loss = float(model.compute_loss(batch))
            pred = model.predict_action_denormalized(batch['proprio_hist'], batch['action_hist'], batch['goal_hist'])
            offset = model.cfg.execution_offset
            error = pred[:,offset:offset+4] - batch['actions'][:,offset:offset+4]
            values[name] = dict(epsilon_loss=loss, sampled_action_rmse=float(error.square().mean().sqrt()))
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--expected_sha256', required=True)
    ap.add_argument('--nominal', type=Path, required=True)
    ap.add_argument('--collections', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--arm', choices=('recovery','nominal'), required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--preflight_only', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ck = unclipped_checkpoint(load_checkpoint(args.checkpoint, args.expected_sha256))
    stats = NormalizerStats.from_dict(ck['normalizer_stats'])
    recovery, early, heldout, audits = recovery_pools(args.collections, ck, stats)
    nominal = make_dataset(args.nominal, ck)
    nominal_audit = audit_action_support(nominal,stats,args.nominal)
    train_names = {x['demo_name'] for x in ck['split_manifest']['train']}
    val_names = {x['demo_name'] for x in ck['split_manifest']['validation']}
    if train_names & val_names:
        raise ValueError('Original nominal splits overlap')
    tr = nominal.sample_indices_for_demos([i for i,d in enumerate(nominal.demos) if d.demo_name in train_names])
    va = nominal.sample_indices_for_demos([i for i,d in enumerate(nominal.demos) if d.demo_name in val_names])
    if not tr or not va:
        raise ValueError('Original episode splits not found')
    provenance = dict(protocol=PROTOCOL, arm=args.arm, parent_sha256=args.expected_sha256,
                      updates=UPDATES, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                      collection_audits=audits, nominal_range_audit=nominal_audit,
                      nominal_sha256=sha256(args.nominal), fixed_normalizer=True,
                      clip_sample=False, support_policy='retain all finite labels; no range-based exclusion',
                      mixture=('100% nominal' if args.arm=='nominal' else
                               '50% nominal, 25% all corrective, 25% first 16 corrective anchors'),
                      selection='last.pt at fixed update budget, not selected on rollout performance')
    if args.preflight_only:
        sample = batch_from([(nominal,tr)],2,np.random.default_rng(42),'cpu')
        correction = batch_from(early,2,np.random.default_rng(42),'cpu')
        for batch in (sample,correction):
            if any(not torch.isfinite(value).all() for value in batch.values()):
                raise ValueError('Nonfinite training window')
            if batch['actions'].shape != (2,16,12):
                raise ValueError('Unexpected action sequence contract')
        print(json.dumps(provenance)); return
    args.output.mkdir(parents=True)
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    device = torch.device(args.device)
    torch.manual_seed(42); rng = np.random.default_rng(642)
    model = Solo12DiffusionPolicy(model_config_from_checkpoint(ck['config'])).to(device)
    model.load_state_dict(ck['ema_model_state_dict']); model.set_normalizer_stats(stats)
    ema = EMAModel(deepcopy(model), max_value=.9999, power=.75)
    opt = model.configure_optimizers(learning_rate=LEARNING_RATE, weight_decay=.001)
    nominal_pool = [(nominal, tr)]
    # Fixed tensors, fixed validation RNG; no validation episodes enter sampling pools.
    batches = {'nominal':batch_from([(nominal,va)],64,rng,device)}
    for i,pool in enumerate(heldout):
        batches[f'recovery_round_{i}'] = batch_from([pool],64,rng,device)
    start = time.monotonic()
    with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,UPDATES+1):
            model.train()
            batch = batch_from(nominal_pool,128 if args.arm=='nominal' else 64,rng,device)
            if args.arm=='recovery':
                others = [batch_from(recovery,32,rng,device),batch_from(early,32,rng,device)]
                batch = {k:torch.cat([v]+[b[k] for b in others]) for k,v in batch.items()}
            for group in opt.param_groups:
                group['lr'] = LEARNING_RATE * min(step/100.,1.)
            opt.zero_grad(set_to_none=True)
            loss = model.compute_loss(batch)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step(); ema.step(model)
            if step==1 or step%500==0:
                devices = [device.index or 0] if device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(842)
                    values = validation_metrics(ema.averaged_model,batches)
                row = dict(update=step,train_loss=float(loss.detach()),validation=values,elapsed_s=time.monotonic()-start)
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row),flush=True)
    saved = {k:ck[k] for k in ('schema_version','policy_kind','noise_scheduler_config','normalizer_stats','split_manifest')}
    config = deepcopy(ck['config']); config['output_dir']=str(args.output.resolve())
    config['dataset']['hdf5_paths']=[str(args.nominal.resolve())]+[str(p.resolve()) for p in args.collections]
    config['optim'].update(learning_rate=LEARNING_RATE,batch_size=BATCH_SIZE,epochs=0,lr_warmup_steps=100)
    saved.update(config=config,model_state_dict=model.state_dict(),ema_model_state_dict=ema.averaged_model.state_dict(),
                 epoch=0,global_step=int(ck.get('global_step',0))+UPDATES,experiment=provenance)
    torch.save(saved,args.output/'last.pt')


if __name__=='__main__': main()
