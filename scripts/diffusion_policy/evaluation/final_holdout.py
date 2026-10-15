"""Freeze, execute and summarize a once-only post-selection route bank.

Eight independent draws per family; the chosen actor and existing task
predicate are fixed before outcomes. This validates the selected WC actor,
not the historical baseline comparison or unseen robots.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile
import numpy as np
from .route_bank import generate_bank, save_route_bank, load_route_bank

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ('procedural','coherent_smooth','rounded_waypoint','hard_waypoint',
            'ood_arc','ood_s_curve','ood_corner')
from scripts.diffusion_policy.runtime.contract import WC_V3_SHA256


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def source_files():
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    other=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    sources=set(tracked)
    sources.update(p for p in other if p.startswith(('scripts/','source/')) and
                   Path(p).suffix in ('.py','.md','.json','.yaml','.yml'))
    return [p for p in sorted(sources) if p and (ROOT/p).is_file() and
            Path(p).suffix.lower() not in ('.sbs','.pt','.pth','.hdf5','.h5','.tar','.gz','.zip') and
            not p.startswith(('output/','scratch/','logs/','checkpoints','docs/context/')) and
            '/evaluations/' not in p and '__pycache__' not in p]


def freeze(args):
    out=Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    if digest(args.checkpoint)!=WC_V3_SHA256:
        raise ValueError('Selected actor differs from frozen WC-v3')
    out.mkdir(parents=True,exist_ok=True)
    routes=generate_bank(FAMILIES,routes_per_family=8,base_seed=1510202601,
                         split='final_20261015',points=101,length_m=4.)
    # Exact-coordinate duplicates against all archived versioned banks are rejected.
    old=set()
    for p in (ROOT/'scripts/dppo_diffusion_rl/experiments').rglob('*.npz'):
        try:
            for r in load_route_bank(p).values():
                old.add(hashlib.sha256(r.xy.tobytes()).hexdigest())
        except (KeyError,ValueError):
            continue
    if any(hashlib.sha256(r.xy.tobytes()).hexdigest() in old for r in routes):
        raise ValueError('Route duplicates an archived bank')
    bank=save_route_bank(out/'routes.npz',routes)
    files=source_files()
    sources={p:digest(ROOT/p) for p in files}
    with zipfile.ZipFile(out/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(ROOT/p,p)
    versions={}
    for package in ('torch','numpy','scipy','isaacsim','rsl-rl-lib','diffusers'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    commands={}
    for split,families in (('id',FAMILIES[:4]),('stress',FAMILIES[4:])):
        for profile in ('constant','transition'):
            name=f'{split}_{profile}'
            cmd=[sys.executable,'-u','scripts/diffusion_policy/evaluate_policy.py',
                 '--checkpoint',str(Path(args.checkpoint).resolve()),'--output_dir',str(out/name),
                 '--route_bank',str(out/'routes.npz'),'--path_shapes',*families,
                 '--speeds','0.25','0.5','0.85','--route_length_m','4','--duration_s','24',
                 '--repeats','8','--seed','20261015','--warmup_steps','25','--posture_settle_steps','50',
                 '--profile_transition_margin_m','0.25','--record_staging_failures','--deterministic_resets',
                 '--save_timeseries','--require_empty_output_dir','--deployment_sampler','graph',
                 '--headless','--device','cuda:0']
            cmd+=['--path_heights','0.2932','0.1705'] if profile=='constant' else [
                  '--transition_fractions','0.5','--transition_directions','both']
            commands[name]=cmd
    manifest=dict(schema=1,actor_sha256=WC_V3_SHA256,bank=bank,base_seed=1510202601,
        independent_routes=56,attempts=672,route_draws_per_family=8,inference_seed=20261015,
        families=list(FAMILIES),speeds=[.25,.5,.85],duration_s=24,dt_s=.02,
        profiles=['walk','crouch','walk_to_crouch','crouch_to_walk'],
        primary='ID at .25/.5, all profiles; 256 attempts over 32 independent routes',
        secondary='ID .85 walk and transitions; .85 constant crouch is separately labeled stress',
        geometry_stress='Legacy stress families, not a claim of disjoint support or arbitrary-path generalization',
        success='Unchanged checkpoint task predicate plus active CTE RMSE <= .10m for adherence',
        scoring='Transition height exemption .25m on each side; all failures remain in success denominators',
        stop_rule='Execute once, retain failures, no model/reward/route edits after inspection',
        inference_scope='RTX laptop FP32 K10 graph; synchronous simulation, no hardware claim',
        source_sha256=sources,source_zip_sha256=digest(out/'source.zip'),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
        python=sys.version,platform=platform.platform(),packages=versions,commands=commands)
    write(out/'protocol.json',manifest)
    print(dict(independent_routes=56,attempts=672,source_files=len(files),source_zip_bytes=(out/'source.zip').stat().st_size),flush=True)


def verify(out, *, working_tree=False):
    m=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if digest(out/'routes.npz')!=m['bank']['sha256'] or digest(out/'source.zip')!=m['source_zip_sha256']:
        raise ValueError('Frozen bank or source archive changed')
    with zipfile.ZipFile(out/'source.zip') as z:
        if set(z.namelist())!=set(m['source_sha256']):
            raise ValueError('Source archive file set changed')
        for p,h in m['source_sha256'].items():
            if hashlib.sha256(z.read(p)).hexdigest()!=h:
                raise ValueError('Frozen archived source changed: '+p)
            if working_tree and (not (ROOT/p).is_file() or digest(ROOT/p)!=h):
                raise ValueError('Working source differs from frozen executable snapshot: '+p)
    for command in m['commands'].values():
        if digest(command[command.index('--checkpoint')+1])!=m['actor_sha256']:
            raise ValueError('Actor changed')
    return m


def execute(args):
    out=Path(args.output_dir); m=verify(out,working_tree=True)
    for name,command in m['commands'].items():
        target=out/name
        if target.exists():
            raise FileExistsError('Refusing rerun or partial replacement: '+str(target))
        print('Starting '+name,flush=True)
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not (target/'evaluation_summary.csv').is_file():
            raise RuntimeError(f'{name} failed, exit {result.returncode}; preserve and inspect log')
        verify(out,working_tree=True)
        print('Completed '+name,flush=True)
    summarize(out,m)


def truth(value): return str(value).lower()=='true'


def summarize(out,m):
    rows=[]
    for name in m['commands']:
        with (out/name/'evaluation_summary.csv').open(encoding='utf-8',newline='') as f:
            rows.extend(dict(r,split=name.split('_')[0],profile_kind=name.split('_')[1]) for r in csv.DictReader(f))
    if len(rows)!=m['attempts']:
        raise ValueError(f'Expected {m["attempts"]} attempts, found {len(rows)}')
    keys=[(r['path_shape'],r['repeat'],r['requested_speed'],r['requested_height'],r['transition_direction']) for r in rows]
    if len(set(keys))!=len(keys) or any(not truth(r['task_success_applicable']) for r in rows):
        raise ValueError('Duplicate or inapplicable final conditions')
    def aggregate(selected):
        success=np.array([truth(r['task_success']) for r in selected])
        adherence=success & np.array([float(r['active_cross_track_rmse_m'])<=.1 for r in selected])
        return dict(n=len(selected),routes=len({(r['path_shape'],r['repeat']) for r in selected}),
            arrived=sum(truth(r['task_arrived']) for r in selected),task=int(success.sum()),
            adherence=int(adherence.sum()),survival=sum(truth(r['survived']) for r in selected),
            mean_active_cte_cm=float(np.nanmean([float(r['active_cross_track_rmse_m'])*100 for r in selected])),
            mean_active_height_mae_cm=float(np.nanmean([float(r['active_height_mae_m'])*100 for r in selected])),
            arrival_speed_mae_cm_s=float(np.nanmean([abs(float(r['task_success_mean_speed_error_m_s']))*100
                                                    for r in selected if truth(r['task_arrived'])])))
    result={'all':aggregate(rows),'groups':{}}
    for split in ('id','stress'):
        for speed in (.25,.5,.85):
            result['groups'][f'{split}_{speed}']=aggregate([r for r in rows if r['split']==split and float(r['requested_speed'])==speed])
            for posture in ('walk','crouch','walk_to_crouch','crouch_to_walk'):
                subset=[r for r in rows if r['split']==split and float(r['requested_speed'])==speed and
                        (r['transition_direction']==posture or (not r['transition_direction'] and
                         abs(float(r['requested_height'])-(.2932 if posture=='walk' else .1705))<1e-4
                         and posture in ('walk','crouch')))]
                result['groups'][f'{split}_{speed}_{posture}']=aggregate(subset)
    primary=[r for r in rows if r['split']=='id' and float(r['requested_speed'])<=.5]
    result['primary']=aggregate(primary)
    # Resample independent route draws within each family, preserving conditions.
    rng=np.random.default_rng(15102026); samples=[]
    grouped={(f,str(i)):[r for r in primary if r['path_shape']==f and r['repeat']==str(i)]
             for f in FAMILIES[:4] for i in range(8)}
    for _ in range(2000):
        chosen=[r for family in FAMILIES[:4] for i in rng.integers(0,8,8) for r in grouped[(family,str(i))]]
        samples.append(aggregate(chosen)['adherence']/len(chosen))
    result['primary']['route_stratified_bootstrap_95_percent']=(100*np.percentile(samples,[2.5,97.5])).tolist()
    result['scope']='One preselected actor; inference repetitions and conditions are not independent geometries. Bootstrap describes sampled generator families, not arbitrary paths.'
    result['reporting_revision']='Absolute arrival-speed error; original signed-mean draft is retained if present. No episode or task predicate changed.'
    write(out/'results_verified.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=('freeze','run','summarize','verify'))
    p.add_argument('--output-dir',required=True)
    p.add_argument('--checkpoint',default='checkpoints_iri/checkpoints_dppo/dppo_wc_v3.pt')
    a=p.parse_args()
    if a.operation=='freeze': freeze(a)
    elif a.operation=='run': execute(a)
    elif a.operation=='verify': verify(Path(a.output_dir)); print('Frozen source/model/bank verified')
    else: summarize(Path(a.output_dir),verify(Path(a.output_dir)))


if __name__=='__main__': main()
