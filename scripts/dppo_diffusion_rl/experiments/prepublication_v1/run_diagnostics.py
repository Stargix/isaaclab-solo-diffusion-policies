"""Sequential local/cluster frozen evaluation; explicit inputs, exclusive outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[4]
EXPECTED_ACTOR='2e75422000cc22ab5023cfbc820eb8d391a69b4fc008a7e0a6bbb2c1e0b54868'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--suites',nargs='+',choices=('visible_geometry','visible_transitions','pace_dynamic','pace_scalar_frozen','pace_scalar_shuffled'))
    args=parser.parse_args(); output=args.output.resolve(); checkpoint=args.checkpoint.resolve()
    actor_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actor_hash!=EXPECTED_ACTOR: raise ValueError('This protocol freezes WC seed42, not another actor')
    protocol=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    for key in ('stress','temporal'):
        path=output/Path(protocol[key]['path']).name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol[key]['sha256']:
            raise ValueError('Route bank changed after declaration')
    common=[sys.executable,str(ROOT/'scripts/diffusion_policy/evaluate_policy.py'),
        '--checkpoint',str(checkpoint),'--route_length_m','4.0',
        '--num_inference_steps','10','--exec_horizon','4',
        '--profile_transition_margin_m','0.25','--posture_settle_steps','50',
        '--deterministic_resets','--headless','--require_empty_output_dir',
        '--device','cuda:0','--save_timeseries','--duration_s','24.0']
    jobs=[]
    if args.smoke:
        jobs=[('smoke_scalar', ['--route_bank',str(output/'temporal_id.npz'),
            '--path_shapes','procedural','--repeats','1','--speeds','.35','.85',
            '--transition_fractions','.5','--transition_directions','both',
            '--pace_budget_mode','scalar_shuffled','--seed','160142'])]
    else:
        geometry=['--route_bank',str(output/'visible_stress.npz'),
            '--path_shapes','ood_s_curve','ood_arc','--repeats','12','--seed','20261001']
        jobs.append(('visible_geometry',geometry+['--speeds','.2','.4','.6','.8','1.0','--path_heights','.1705','.2932']))
        jobs.append(('visible_transitions',geometry+['--speeds','.35','.65','.85',
            '--transition_fractions','.5','--transition_directions','both']))
        for mode in protocol['scalar_controls']:
            jobs.append(('pace_'+mode,['--route_bank',str(output/'temporal_id.npz'),
                '--path_shapes','procedural','coherent_smooth','rounded_waypoint','hard_waypoint',
                '--repeats','6','--speeds','.35','.65','.85','--transition_fractions','.5',
                '--transition_directions','both','--pace_budget_mode',mode,'--seed','160142']))
    if args.suites:
        jobs=[job for job in jobs if job[0] in args.suites]
    if protocol.get('record_staging_failures',False):
        common.append('--record_staging_failures')
    env=os.environ.copy(); env['PYTHONPATH']=str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    for name, extra in jobs:
        destination=output/name
        if destination.exists(): raise FileExistsError(f'Refusing to overwrite {destination}')
        log_path=output/(name+'.log'); command_path=output/(name+'_command.json')
        if log_path.exists() or command_path.exists():
            raise FileExistsError(f'Preserve or explicitly relocate the previous attempt artifacts for {name}')
        command=common+['--output_dir',str(destination)]+extra
        command_path.write_text(json.dumps(command,indent=2),encoding='utf-8')
        sources=(Path(__file__),ROOT/'scripts/diffusion_policy/evaluate_policy.py',
                 ROOT/'scripts/diffusion_policy/evaluation/pace_intervention.py',output/'protocol.json')
        provenance=dict(actor_sha256=actor_hash,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
        (output/(name+'_provenance.json')).write_text(json.dumps(provenance,indent=2),encoding='utf-8')
        print(f'Starting {name}',flush=True)
        with log_path.open('w',encoding='utf-8') as handle:
            subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((destination/'evaluation_summary.json').read_text(encoding='utf-8'))
        if summary['checkpoint_sha256'].lower()!=actor_hash: raise ValueError('Evaluated actor hash mismatch')
        print(f'Finished {name}: {summary["num_scenarios"]} conditions',flush=True)


if __name__=='__main__': main()
