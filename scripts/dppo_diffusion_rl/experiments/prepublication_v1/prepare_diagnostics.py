"""Predeclare two compact tests of a frozen actor; no training distribution edits."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.diffusion_policy.evaluation.route_bank import (
    MaterializedRoute, load_route_bank, save_route_bank, _integrate_curvature,
)
from scripts.diffusion_policy.evaluation.diagnostic_v2 import route_descriptors

ROOT = Path(__file__).resolve().parents[4]


def prepare(output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new output root, not {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(20261001)
    routes, parameters = [], []
    for family, peaks in (("ood_s_curve", (.8, 1.2, 1.6)), ("ood_arc", (.4, .8, 1.2))):
        for repeat in range(12):
            magnitude = float(peaks[repeat // 4] * rng.uniform(.95, 1.05))
            sign = -1 if repeat % 2 else 1
            arc_midpoints = (np.arange(100) + .5) * .04
            curvature = sign * magnitude * (
                np.sin(2*np.pi*arc_midpoints/4.) if family == 'ood_s_curve' else np.ones(100))
            xy, yaw = _integrate_curvature(curvature, 4.)
            route = MaterializedRoute(family, repeat, 20261001 + repeat,
                                      'diagnostic_v1', xy, yaw)
            chord = xy[-1] - xy[0]
            excursion = float(np.max(np.abs(xy[:,0]*chord[1]-xy[:,1]*chord[0])) / np.linalg.norm(chord))
            if excursion < .15 or np.linalg.norm(chord) < .5:
                raise ValueError('Not a visible, open-route diagnostic')
            routes.append(route)
            parameters.append(dict(family=family, repeat=repeat, peak_parameter_rad_m=magnitude,
                sign=sign, lateral_chord_excursion_m=excursion, **route_descriptors(xy)))
    source = ROOT / 'scripts/dppo_diffusion_rl/experiments/wct_final_evaluation_v1/route_banks/wct_id_routes_v1.npz'
    bank = load_route_bank(source)
    selected = [bank[(family, repeat)] for family in
                ('procedural','coherent_smooth','rounded_waypoint','hard_waypoint') for repeat in range(6)]
    manifest = dict(protocol='prepublication_v1', generator_seed=20261001,
        selection='All declared stress draws; first six ID draws per family; no actor outcomes inspected.',
        stress=save_route_bank(output/'visible_stress.npz', routes),
        temporal=save_route_bank(output/'temporal_id.npz', selected), parameters=parameters,
        diagnostic_families='Names reuse evaluator slots only, NOT old ood_* bank definitions.',
        speeds_constant=[.2,.4,.6,.8,1.], heights=[.1705,.2932],
        speeds_transition=[.35,.65,.85], duration_s=24.,
        record_staging_failures=True,
        initialization_policy='Count failed posture preparations as failures; never reroll, replace or drop them.',
        scalar_controls=['dynamic','scalar_frozen','scalar_shuffled'],
        acceptance='Characterization, not a guaranteed 100% gate; do not tune from this bank.')
    (output/'protocol.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,6,figsize=(10,4.0),sharex=True,sharey=True)
    for row in range(2):
        for col in range(6):
            route = routes[row*12 + 2*col]
            ax=axes[row,col]
            ax.plot(*route.xy.T,lw=1.5,color=('#285b85','#b75c37')[row])
            ax.scatter(*route.xy[0],s=14,color='#34754d')
            ax.scatter(*route.xy[-1],s=18,marker='x',color='black')
            ax.set(aspect='equal',title=f'|k|={parameters[row*12+2*col]["peak_parameter_rad_m"]:.2f}',xlabel='x [m]')
            if col==0: ax.set_ylabel(('Alternating turn','Sustained arc')[row]+'\ny [m]')
    fig.tight_layout(); fig.savefig(output/'predeclared_geometry.png',dpi=180); plt.close(fig)
    print(f'Prepared {len(routes)} stress and {len(selected)} paired ID geometries in {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)
