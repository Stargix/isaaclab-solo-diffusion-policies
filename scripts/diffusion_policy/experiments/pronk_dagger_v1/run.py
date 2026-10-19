"""Preflight by default; --execute runs the complete bounded isolated experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import (
    ROOT, BASE_SHA, EXPERT, EXPERT_SHA, NOMINAL, load_checkpoint, sha256)
from scripts.diffusion_policy.experiments.pronk_dagger_v1.protocol import (
    ROUNDS, UPDATES, BATCH_SIZE, NOMINAL_SHA, qualified, success_count)

COLLECT = ROOT/'scripts/diffusion_policy/experiments/pronk_intervention_v1/collect.py'
FIT = Path(__file__).with_name('fit.py')


def read_report(path):
    report = json.loads((path/'collection.json').read_text(encoding='utf-8'))
    if not report.get('attempts'):
        raise ValueError(f'Missing attempt outcomes: {path}')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=ROOT/'best.pt')
    ap.add_argument('--expert',type=Path,default=EXPERT)
    ap.add_argument('--nominal',type=Path,default=NOMINAL)
    ap.add_argument('--output',type=Path,default=ROOT/'scratch/pronk_dagger_v1')
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--execute',action='store_true')
    args = ap.parse_args()
    for name,pin in ((args.checkpoint,BASE_SHA),(args.expert,EXPERT_SHA),(args.nominal,NOMINAL_SHA)):
        if sha256(name)!=pin: raise ValueError(f'Frozen input hash mismatch: {name}')
    ck = load_checkpoint(args.checkpoint,BASE_SHA)
    if 'split_manifest' not in ck: raise ValueError('Original train/validation manifest required')
    dependencies = [COLLECT,FIT,ROOT/'scripts/diffusion_policy/runtime/controller.py',
                    ROOT/'source/isaaclab_tasks/isaaclab_tasks/direct/solo12/solo12_pronk_v8.py']
    sources = set(dependencies)
    for directory in (FIT.parent,COLLECT.parent,ROOT/'scripts/diffusion_policy/model',
                      ROOT/'scripts/diffusion_policy/train/data',
                      ROOT/'scripts/diffusion_policy/train/conditioning',
                      ROOT/'scripts/diffusion_policy/runtime',
                      ROOT/'source/isaaclab_tasks/isaaclab_tasks/direct/solo12'):
        sources.update(directory.glob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sorted(sources)}
    manifest = dict(protocol='pronk_dagger_v1',rounds=ROUNDS,updates_per_round=UPDATES,
                    batch_size=BATCH_SIZE,checkpoint_sha256=BASE_SHA,expert_sha256=EXPERT_SHA,
                    nominal_sha256=NOMINAL_SHA,source_sha256=source_hashes,
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                    git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
                    status='preflight_passed',results=[])
    if not args.execute:
        print('Input/dependency preflight only; this does not simulate or validate data windows.',flush=True)
        print(json.dumps(manifest,indent=2)); return
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    manifest['status']='running'

    def save():
        (args.output/'experiment.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    def launch(script,arguments,logname):
        for p in sources:
            if sha256(p)!=source_hashes[str(p.relative_to(ROOT))]:
                raise ValueError(f'Source changed mid-experiment: {p}')
        command=[sys.executable,str(script)]+[str(x) for x in arguments]
        with (args.output/(logname+'.log')).open('w',encoding='utf-8') as log:
            log.write(json.dumps(command)+'\n'); log.flush()
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

    def evaluate(checkpoint,name,seed):
        dest=args.output/name
        launch(COLLECT,['--checkpoint',checkpoint,'--expected_sha256',sha256(checkpoint),
                        '--expert',args.expert,'--output',dest,'--rounds',1,'--seed',seed,
                        '--evaluate','--headless','--device',args.device],name)
        report=read_report(dest)
        if len(report['attempts'])!=18: raise ValueError('Incomplete student-only screen')
        return dict(name=name,seed=seed,checkpoint_sha256=sha256(checkpoint),
                    qualified=qualified(report),successes=success_count(report),attempts=18,
                    report=str(dest/'collection.json'))

    save()
    try:
        manifest['results'].append(evaluate(args.checkpoint,'initial',543)); save()
        students={'recovery':args.checkpoint,'nominal':args.checkpoint}
        collections=[]
        for round_index in range(ROUNDS):
            dest=args.output/f'round_{round_index}_collection'
            launch(COLLECT,['--checkpoint',students['recovery'],'--expected_sha256',sha256(students['recovery']),
                            '--expert',args.expert,'--output',dest,'--rounds',4,'--seed',443+100*round_index,
                            '--dagger_round',round_index,'--headless','--device',args.device],dest.name)
            report=read_report(dest)
            if not qualified(report) or len(report['attempts'])!=72:
                raise ValueError('Expert takeover feasibility gate failed; no training on unqualified collection')
            collections.append(dest/'recovery.hdf5')
            for arm in ('recovery','nominal'):
                output=args.output/f'round_{round_index}_{arm}'
                launch(FIT,['--checkpoint',students[arm],'--expected_sha256',sha256(students[arm]),
                            '--nominal',args.nominal,'--collections',*collections,'--output',output,
                            '--arm',arm,'--device',args.device],output.name)
                students[arm]=output/'last.pt'
                loaded=load_checkpoint(students[arm])
                if loaded['experiment']['updates']!=UPDATES: raise ValueError('Incomplete fit artifact')
                manifest['results'].append(evaluate(students[arm],output.name+'_eval',543)); save()
        # Fresh seeds are consumed once, only after all updates, with no selection.
        for arm,checkpoint in dict(initial=args.checkpoint,**students).items():
            for seed in (9543,10543,11543):
                manifest['results'].append(evaluate(checkpoint,f'final_{arm}_{seed}',seed)); save()
        manifest['status']='complete'
        manifest['interpretation']='Compare fixed-budget final recovery vs nominal control, not best rollout checkpoint. Nominal physics only.'
        save()
        from scripts.diffusion_policy.experiments.pronk_dagger_v1.summarise import summarise
        summarise(args.output)
    except BaseException as error:
        manifest['status']='stopped'; manifest['reason']=str(error); save(); raise


if __name__=='__main__': main()
