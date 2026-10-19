"""All-attempt summaries and fixed-first-repeat physical traces for the pilot."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import DEFAULT_OUTPUT,SPEEDS


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=ap.parse_args()
    out=args.output/'analysis'
    out.mkdir(exist_ok=True)
    collection=json.loads((args.output/'collection.json').read_text())
    summary=dict(collection_attempts=len(collection['attempts']),
                 collection_survived=sum(r['survived'] for r in collection['attempts']),
                 collection_accepted=sum(r['saved'] for r in collection['attempts']),models={})
    traces={}
    for name in ('initial','nominal','recovery'):
        folder=args.output/('eval_'+name)
        if not (folder/'collection.json').exists():continue
        report=json.loads((folder/'collection.json').read_text())
        if report['mode']!='student_evaluation':raise ValueError('Not a student-only evaluation')
        traces[name]=np.load(folder/'cohort_0.npz')
        rows=[]
        for speed in SPEEDS:
            group=[r for r in report['attempts'] if np.isclose(r['speed'],speed)]
            rows.append(dict(speed=speed,attempts=len(group),survivors=sum(r['survived'] for r in group),
                             gait_speed_pass=sum(r['survived'] and r['cycles']>=2 and abs(r['mean_vx']-speed)<=.10 for r in group),
                             completed_cycles=sum(r['cycles'] for r in group),
                             mean_time_before_failure_capped_s=float(np.mean([6. if r['survived'] else r['failure_time_s'] for r in group])),
                             mean_first_failure_s=float(np.mean([r['failure_time_s'] for r in group if not r['survived']])) if any(not r['survived'] for r in group) else None))
        summary['models'][name]=dict(checkpoint_sha256=report['base_sha256'],rows=rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    colors=dict(initial='#2878B5',nominal='#E58B27',recovery='#29966F')
    fig,axes=plt.subplots(1,3,figsize=(12,3.3),layout='constrained')
    for j,(name,model) in enumerate(summary['models'].items()):
        x=np.arange(3)+(j-1)*.23
        axes[0].bar(x,[100*r['survivors']/r['attempts'] for r in model['rows']],width=.21,color=colors[name],label=name)
        axes[1].bar(x,[r['completed_cycles']/r['attempts'] for r in model['rows']],width=.21,color=colors[name])
        axes[2].bar(x,[r['mean_time_before_failure_capped_s'] for r in model['rows']],width=.21,color=colors[name])
    for ax in axes:
        ax.set_xticks(range(3),[f'{s:.2f}' for s in SPEEDS])
        ax.set_xlabel('Requested speed [m/s]')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
    axes[0].set(ylabel='Six-second survival [%]',ylim=(0,105))
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].set_ylabel('Completed cycles per attempt, after 2 s')
    axes[2].set(ylabel='Time before first failure [s], capped at 6 s',ylim=(0,6.3))
    fig.savefig(out/'physical_comparison.png',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(10,7),sharex=True,layout='constrained')
    for name,z in traces.items():
        time=(np.arange(len(z['height']))+1)*.02
        for j,speed in enumerate(SPEEDS):
            i=j*6  # first predeclared repeat, never select a successful example
            valid=z['valid'][:,i].astype(bool)
            for row,key in enumerate(('height','vx','clearance')):
                values=z[key][:,i]
                if key=='clearance':values=values.min(-1)*1000
                axes[row,j].plot(time,np.where(valid,values,np.nan),color=colors[name],label=name,lw=1.2)
    for j,speed in enumerate(SPEEDS):
        axes[0,j].set_title(f'{speed:.2f} m/s: first declared repeat')
        axes[1,j].axhline(speed,color='.4',ls='--',lw=.8)
        axes[2,j].axhline(3.,color='.4',ls=':',lw=.8)
        axes[2,j].set_xlabel('Time since reset [s]')
        for ax in axes[:,j]:
            ax.axvspan(0,.5,color='.8',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
            ax.grid(alpha=.15)
    axes[0,0].set_ylabel('Base height [m]')
    axes[1,0].set_ylabel('Forward speed [m/s]')
    axes[2,0].set_ylabel('Minimum four-foot clearance [mm]')
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(out/'fixed_repeat_traces.png',dpi=200)
    plt.close(fig)
    lines=['# Pronk intervention pilot results','',
           f"Collection: {summary['collection_accepted']}/{summary['collection_attempts']} accepted; {summary['collection_survived']} survived. Accepted-data filtering is not an all-attempt success rate.",'',
           '| Model | Speed | Six-second survivors | Survive + cycles + speed | Cycles after warm-up, all attempts |',
           '|---|---:|---:|---:|---:|']
    for name,model in summary['models'].items():
        for r in model['rows']:
            lines.append(f"| {name} | {r['speed']:.2f} | {r['survivors']}/{r['attempts']} | {r['gait_speed_pass']}/{r['attempts']} | {r['completed_cycles']} |")
    lines+=['','Single local pilot, seed543, nominal physics and straight commands only. No expert assists during evaluation; startup failures count. Fixed update-600 checkpoints, not physical best-of-many. Six seconds is not qualification for a new mixed policy or hardware.',
            '', 'Trace panels use the first declared repeat per speed and truncate after its first failure. The .5-s zero-command startup is shaded. Clearance >3 mm alone is not a completed cycle; the native supported-landing detector supplies cycle counts.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
