"""Prepare immutable local manifests and run one sequential GPU job at a time."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.diffusion_policy.evaluation.route_bank import generate_bank, save_route_bank

REPO = Path(__file__).resolve().parents[2]
FAMILIES = ['procedural', 'coherent_smooth', 'rounded_waypoint', 'hard_waypoint',
            'ood_arc', 'ood_s_curve', 'ood_corner']
GROUPS = [FAMILIES[:2], FAMILIES[2:4], FAMILIES[4:]]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def source_hashes():
    names = subprocess.check_output(['git', 'ls-files', 'scripts',
        'source/isaaclab_tasks/isaaclab_tasks/direct/solo12'], cwd=REPO, text=True).splitlines()
    # Include this package before its first commit as well.
    names += [str(p.relative_to(REPO)).replace('\\', '/') for p in Path(__file__).parent.glob('*.py')]
    return {n: sha(REPO / n) for n in sorted(set(names)) if (REPO / n).is_file()}


def prepare(config, output):
    if output.exists(): raise FileExistsError(f'Use a new protocol directory: {output}')
    cfg = json.loads(config.read_text())
    def path(v):
        expanded = os.path.expandvars(v)
        if '$' in expanded: raise ValueError(f'Unresolved variable: {v}')
        p = Path(expanded).expanduser()
        return p.resolve() if p.is_absolute() else (config.parent / p).resolve()
    external = {k: str(path(v)) for k, v in cfg['inputs'].items()}
    hashes = {p: sha(p) for p in external.values()}
    for key, digest in cfg.get('expected_sha256', {}).items():
        if hashes[external[key]] != digest: raise ValueError(f'Unexpected artifact identity: {key}')
    actors = []
    for a in cfg['actors']:
        a = dict(a, checkpoint=str(path(a['checkpoint'])))
        if a['kind'] not in ('hierarchy', 'policy'): raise ValueError('Unknown actor kind')
        hashes[a['checkpoint']] = sha(a['checkpoint'])
        actors.append(a)
    if len({a['name'] for a in actors}) != len(actors): raise ValueError('Duplicate actor name')
    output.mkdir(parents=True)
    banks = {}
    for name, seed in [('core', 2026111401), ('composition', 2026111402)]:
        p = output / f'{name}.npz'
        save_route_bank(p, generate_bank(FAMILIES, routes_per_family=2, base_seed=seed,
            split=f'development_comparison_20261114_{name}'))
        banks[name] = str(p)
        hashes[str(p)] = sha(p)
    jobs = []
    for seed in [42, 43, 44]:
        for init in ['bc', 'random']:
            name = f'gaussian_{init}_{seed}'
            dest = output / 'outputs' / name
            args = ['-m', 'scripts.gaussian_chunk_rl.train', '--checkpoint', external['bc_prior'],
                    '--initialization', init, '--output_dir', str(dest), '--seed', str(seed),
                    '--num_envs', '4096', '--iterations', '150', '--rollout_chunks', '32',
                    '--actor_lr', '1e-5', '--adaptive_actor_lr', '--min_actor_lr', '1e-7',
                    '--initial_std', '0.04', '--clip_ratio', '0.2', '--target_kl', '0.02',
                    '--headless', '--device', 'cuda:0']
            jobs.append(dict(id=name, group='train', args=args, output=str(dest), expected=None))
            actors.append(dict(name=name, kind='policy', checkpoint=str(dest/'last.pt'),
                               horizon=4, dependency=name))
    for actor in actors:
        for suite in ['core', 'composition']:
            for gi, families in enumerate(GROUPS):
                for variant in range(2):
                    # Stage zero is a separately labelled sensitivity, not a replacement score.
                    stages = [50, 0] if (suite == 'core' and cfg.get('startup_sensitivity', False)) else [50]
                    for settle in stages:
                        name = f'{actor["name"]}_{suite}_{gi}_{variant}_stage{settle}'
                        dest = output / 'outputs' / name
                        args = ['scripts/diffusion_policy/evaluate_policy.py', '--checkpoint',
                            external['reference'] if actor['kind']=='hierarchy' else actor['checkpoint'],
                            '--output_dir', str(dest), '--route_bank', banks[suite], '--path_shapes',
                            *families, '--speeds', *(['0.25','0.5','0.85'] if suite=='core' else ['0.35','0.65','0.85']),
                            '--repeats','2','--route_length_m','4','--duration_s','24',
                            '--warmup_steps','25','--posture_settle_steps',str(settle),
                            '--profile_transition_margin_m','0.25','--record_staging_failures',
                            '--deterministic_resets','--require_empty_output_dir','--save_timeseries',
                            '--exec_horizon',str(actor['horizon']),'--seed',
                            '2026111403' if suite=='core' else '2026111404','--headless','--device','cuda:0']
                        if suite == 'composition':
                            heights = ['0.2932','0.1705'] if variant==0 else ['0.1705','0.2932']
                            args += ['--height_profile','interleaved','--height_segment_m','0.8','--height_cycle',*heights]
                        elif variant==0: args += ['--path_heights','0.2932','0.1705']
                        else: args += ['--transition_fractions','0.5','--transition_directions','both']
                        if actor['kind']=='hierarchy':
                            args += ['--hierarchy_checkpoint',actor['checkpoint'],'--walk_checkpoint',external['walk'],
                                     '--crouch_checkpoint',external['crouch'],'--save_hierarchy_trace']
                        jobs.append(dict(id=name, group='eval', args=args, output=str(dest), actor=actor['name'],
                            suite=suite, variant=variant, stage=settle, geometry_group=gi,
                            dependency=actor.get('dependency'),
                            expected=len(families)*2*3*(2 if suite=='core' else 1)))
    if len({a['name'] for a in actors}) != len(actors): raise ValueError('Actor name collides with generated train arm')
    protocol = dict(version=1, config=cfg, inputs=hashes, sources=source_hashes(), jobs=jobs,
                    actors=actors, bank_paths=banks, inference='native checkpoint sampling',
                    training_transitions_per_arm=4096*150*32*4,
                    initialization_estimand='effect of BC weights conditional on fixed BC dataset statistics')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(f'Prepared {len(jobs)} jobs; preparation does not run training or evaluation.')


def run(root, group, job_id):
    protocol = json.loads((root/'protocol.json').read_text())
    for name, digest in protocol['sources'].items():
        if sha(REPO/name) != digest: raise ValueError(f'Source changed: {name}; prepare a new protocol')
    for name, digest in protocol['inputs'].items():
        if sha(name) != digest: raise ValueError(f'Input changed: {name}')
    state = root/'state'; state.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO),str(REPO/'source/isaaclab'),
        str(REPO/'source/isaaclab_assets'),str(REPO/'source/isaaclab_tasks'),str(REPO/'source/isaaclab_rl'),
        os.environ.get('PYTHONPATH','')]), PYTHONUNBUFFERED='1')
    chosen = [j for j in protocol['jobs'] if j['group']==group and (not job_id or j['id']==job_id)]
    if not chosen: raise ValueError('No matching jobs')
    lock = state/'RUNNING'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        for j in chosen:
            done = state/(j['id']+'.json')
            if done.exists():
                record = json.loads(done.read_text())
                if any(sha(p)!=h for p,h in record['outputs'].items()): raise ValueError('Completed output changed')
                continue
            if j.get('dependency') and not (state/(j['dependency']+'.json')).exists():
                raise ValueError(f'Train dependency missing: {j["dependency"]}')
            if j.get('dependency'):
                dependency=json.loads((state/(j['dependency']+'.json')).read_text())
                if any(sha(p)!=h for p,h in dependency['outputs'].items()):
                    raise ValueError('Training dependency changed after completion')
            dest = Path(j['output'])
            if dest.exists(): raise FileExistsError(f'Partial output preserved: {dest}; do not overwrite')
            print('RUN', j['id'], flush=True)
            with (state/(j['id']+'.log')).open('x') as log:
                subprocess.run([sys.executable,*j['args']],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if group=='eval':
                with (dest/'evaluation_summary.csv').open() as f: rows=list(csv.DictReader(f))
                if len(rows)!=j['expected']: raise ValueError('Incomplete evaluation')
                for f in ['evaluation_summary.json','timeseries.npz','timeseries_metadata.json']: assert (dest/f).is_file()
                expected = ['evaluation_summary.csv','evaluation_summary.json','episode_metrics.json','timeseries.npz','timeseries_metadata.json']
            else:
                expected=['last.pt','best.pt','metrics.jsonl','run_config.json']
                records=[json.loads(s) for s in (dest/'metrics.jsonl').read_text().splitlines()]
                if len(records)!=150: raise ValueError('Incomplete train')
                if records[-1]['total_physics_steps'] != 4096*150*32*4: raise ValueError('Interaction budget differs')
                if not any(r['Policy/actor_updates']>0 for r in records): raise ValueError('No actor updates')
            done.write_text(json.dumps(dict(job=j, outputs={str(dest/f):sha(dest/f) for f in expected}),indent=2))
    finally:
        lock.unlink()


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    q=sub.add_parser('prepare'); q.add_argument('--config',type=Path,required=True); q.add_argument('--output',type=Path,required=True)
    q=sub.add_parser('run'); q.add_argument('--protocol',type=Path,required=True); q.add_argument('--group',choices=['train','eval'],required=True); q.add_argument('--job')
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.config.resolve(),a.output.resolve())
    else: run(a.protocol.resolve(),a.group,a.job)


if __name__=='__main__': main()
