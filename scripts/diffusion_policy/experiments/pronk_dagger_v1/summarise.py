"""Summarise all attempts, including failures; no best-seed selection."""
import argparse
import json
from pathlib import Path


def summarise(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    experiment=json.loads((output/'experiment.json').read_text(encoding='utf-8'))
    groups={arm:[r for r in experiment['results'] if r['name'].startswith('final_'+arm+'_')]
            for arm in ('initial','nominal','recovery')}
    lines=['# Iterative pronk imitation results','',f"Status: {experiment['status']}",'',
           'Nominal physics, six-second attempts. Success requires survival, at least two',
           'verified cycles after two seconds, and mean speed error <=0.10 m/s.',
           'These small seed banks are diagnostic, not a publication benchmark.','',
           '| Arm | Successes / attempts | Per-seed successes / 18 |',
           '| --- | --- | --- |']
    fig,axes=plt.subplots(1,3,figsize=(10,3),layout='constrained')
    for arm,color in zip(groups,('#4c78a8','#f58518','#54a24b')):
        reports=groups[arm]
        if not reports: continue
        successes=sum(r['successes'] for r in reports)
        total=sum(r['attempts'] for r in reports)
        lines.append(f"| {arm} | {successes}/{total} | "+', '.join(str(r['successes']) for r in reports)+' |')
        rows=[]
        for report in reports:
            rows.extend(json.loads(Path(report['report']).read_text(encoding='utf-8'))['attempts'])
        speeds=(.2,.35,.5)
        for axis,metric in zip(axes,('survived','cycles','mean_vx')):
            values=[np.mean([r[metric] for r in rows if np.isclose(r['speed'],speed)]) for speed in speeds]
            axis.plot(speeds,values,'o-',label=arm,color=color)
    for axis,title in zip(axes,('Survival fraction','Verified cycles after 2 s','Mean forward velocity [m/s]')):
        axis.set(xlabel='Requested speed [m/s]',title=title)
        axis.grid(alpha=.2)
    axes[0].set_ylim(0,1.05); axes[0].legend(frameon=False)
    fig.savefig(output/'final_comparison.png',dpi=180); plt.close(fig)
    lines += ['', 'Cycle and velocity panels include every attempt. Early failures can have no',
              'post-two-second measurements (stored as zero by the collector); do not interpret',
              'these panels as steady gait estimates. Inspect the per-attempt reports and traces.',
              '', 'Do not infer arbitrary-path control, perturbation robustness or preservation',
              'after DPPO from this isolated nominal imitation experiment.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarise(parser.parse_args().output)
