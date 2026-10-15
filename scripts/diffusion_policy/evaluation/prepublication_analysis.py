"""Close reporting contracts from frozen archives without modifying their files.

Historical headline CSVs permit a new *temporal* adherence gate and active
safety classification. Only saved traces permit the independent reconstructed
spatial metric and height-margin sensitivity; neither is exact train replay.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scripts.diffusion_policy.evaluation.report_contract import (
    REPORT_CONTRACT_VERSION, classify_outcome, truth, number,
    spatial_tracking, paired_route_bootstrap, project_saved_trace,
)


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as handle: return list(csv.DictReader(handle))


def write(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def condition_key(row):
    # requested_height is a redundant default on native single transitions:
    # the evaluator overwrites it with the fixed 0.1705/0.2932 profile.
    # Some archived WCT defaults differ; pair the actual task, not unused CLI values.
    paired=dict(row)
    if np.isfinite(number(row.get('transition_fraction'))) or row.get('height_schedule',''):
        paired['requested_height']='profile_defined'
    return tuple(str(paired.get(k,'')) if k in ('path_shape','repeat','transition_direction','height_profile','height_schedule')
                 or (k=='requested_height' and paired[k]=='profile_defined')
                 else round(number(row.get(k)),6) if np.isfinite(number(row.get(k))) else 'none'
                 for k in ('path_shape','repeat','requested_speed','requested_height','transition_fraction',
                           'transition_direction','height_profile','height_segment_m','height_schedule'))


def boundaries(scenario):
    arc=np.asarray(scenario['path_cumulative_m'],dtype=float)
    schedule=scenario.get('height_schedule_m',[])
    fraction=scenario.get('transition_fraction')
    if fraction is not None:
        values=(arc >= arc[-1]*float(fraction)).astype(int)
    elif schedule:
        segment=np.minimum(np.floor(arc/float(scenario['height_segment_m'])).astype(int),len(schedule)-1)
        values=np.asarray(schedule)[segment]
    else: return np.empty(0)
    return arc[np.flatnonzero(np.abs(np.diff(values))>1e-5)+1]


def analyze_traces(folder: Path):
    meta=json.loads((folder/'timeseries_metadata.json').read_text(encoding='utf-8'))
    summary=read(folder/'evaluation_summary.csv')
    # NPZ indexing decompresses a complete array each time. Cache only the four
    # arrays used here once, not once per condition (thousands of conditions).
    with np.load(folder/'timeseries.npz',allow_pickle=False) as archive:
        data={key:archive[key] for key in ('valid','positions_xy','achieved_height_m','required_height_m')}
    rows=[]
    for scenario,row in zip(meta['scenarios'],summary,strict=True):
        scenario=dict(scenario)
        if not scenario.get('height_segment_m'):
            scenario['height_segment_m']=number(row.get('height_segment_m'))
        if scenario.get('transition_fraction') is None:
            value=number(row.get('transition_fraction'))
            scenario['transition_fraction']=value if np.isfinite(value) else None
        i=int(scenario['index'])
        if i!=int(row['scenario_id']): raise ValueError('Trace order mismatch')
        steps=int(row['active_steps'])
        if steps==0: continue
        valid=data['valid'][:steps,i]
        if not valid.all(): raise ValueError('Invalid state in active trace')
        path=np.asarray(scenario['path_xy'],dtype=float)
        origin=float(np.asarray(scenario['path_cumulative_m'])[-1])-number(row['target_progress_m'])
        xy=data['positions_xy'][:steps,i]
        projection,cte=project_saved_trace(np.vstack((path[0],xy)),path)
        progress=projection[1:].astype(float)-origin
        heights=data['achieved_height_m'][:steps,i]
        required=data['required_height_m'][:steps,i]
        for margin in (0.,.1,.25):
            stats=spatial_tracking(progress,cte[1:],heights,required,boundaries(scenario)-origin,margin_m=margin)
            pose_speed_ok=all(truth(row[k]) for k in
                ('task_arrived','task_success_yaw_ok','task_success_terminal_height_ok','task_success_mean_speed_ok'))
            height_ok=np.isfinite(stats['spatial_height_mae_m']) and stats['spatial_height_mae_m']<=.04
            rows.append(dict(suite=folder.name,scenario=i,path_shape=scenario['path_shape'],
                repeat=scenario['repeat'],requested_speed=number(row['requested_speed']),
                requested_height=number(row['requested_height']),
                margin_m=margin,archived_success=truth(row['task_success']),
                reconstructed_height_success=bool(pose_speed_ok and height_ok),
                spatial_adherence_ok=bool(np.isfinite(stats['spatial_cte_rmse_m']) and stats['spatial_cte_rmse_m']<=.1),
                temporal_cte_rmse_m=number(row['active_cross_track_rmse_m']),
                projected_temporal_cte_rmse_m=float(np.sqrt(np.mean(cte[1:]**2))),
                arrival=truth(row['task_arrived']), **stats))
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trace_root',type=Path)
    args=parser.parse_args(); root=args.root.resolve(); output=args.output.resolve()
    if output.exists(): raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    metrics=root/'scratch/paper_draft_20260930/result_metrics.csv'
    evaluation=root/'scripts/dppo_diffusion_rl/evaluations'
    inputs=[]; groups={}; summaries=[]; conditions=[]
    for record in read(metrics):
        folder=evaluation/record['folder']/record['suite']
        path=folder/'evaluation_summary.csv'
        rows=read(path); transformed=[]
        if len(rows)!=int(record['n']): raise ValueError('Historical count mismatch')
        seed=43 if 'seed43' in record['folder'] else 44 if 'seed44' in record['folder'] else 42
        metadata=json.loads((folder/'evaluation_summary.json').read_text(encoding='utf-8'))
        for row in rows:
            outcome=classify_outcome(row)
            transformed.append(dict(row,**outcome))
            conditions.append(dict(suite=record['suite'],model=record['model'],seed=seed,
                scenario=row['scenario_id'],path_shape=row['path_shape'],repeat=row['repeat'],
                requested_speed=row['requested_speed'],transition_direction=row['transition_direction'],
                active_height_mae_m=number(row['active_height_mae_m']),**outcome))
        counts={k:sum(bool(r[k]) for r in transformed) for k in (
            'archived_success','adherence_success','active_survival','full_horizon_survival',
            'prearrival_physical_failure','postarrival_physical_failure','task_arrived')}
        summaries.append(dict(suite=record['suite'],model=record['model'],seed=seed,
            n=len(rows),independent_routes=len({(r['path_shape'],r['repeat']) for r in rows}),
            **{k+'_pct':100*v/len(rows) for k,v in counts.items()}))
        groups[(record['suite'],record['model'],seed)]=(transformed,metadata)
        inputs.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            checkpoint_sha256=metadata['checkpoint_sha256'],route_bank_sha256=metadata.get('route_bank_sha256')))
    write(output/'condition_metrics.csv',conditions); write(output/'suite_metrics.csv',summaries)
    paired=[]
    suites=sorted({key[0] for key in groups})
    for suite in suites:
        for a,b,seeds in (('WC DPPO','WCT DPPO',(42,43,44)),('Diff BC','Det BC',(42,))):
            if not all((suite,m,s) in groups for m in (a,b) for s in seeds): continue
            route_seed_values=defaultdict(list)
            for seed in seeds:
                left,lmeta=groups[(suite,a,seed)]; right,rmeta=groups[(suite,b,seed)]
                if lmeta.get('route_bank_sha256')!=rmeta.get('route_bank_sha256'):
                    raise ValueError('Paired comparison has different banks')
                lmap={condition_key(r):r for r in left}; rmap={condition_key(r):r for r in right}
                if len(lmap)!=len(left) or len(rmap)!=len(right) or lmap.keys()!=rmap.keys():
                    raise ValueError(f'Conditions not paired: {suite}, {a}, {b}, {seed}')
                for metric in ('archived_success','adherence_success','active_survival','full_horizon_survival'):
                    route_values=defaultdict(list)
                    for key,r in lmap.items():
                        route_values[(r['path_shape'],int(r['repeat']))].append(100*(int(r[metric])-int(rmap[key][metric])))
                    strata=defaultdict(list)
                    for route,values in route_values.items():
                        strata[route[0]].append(float(np.mean(values)))
                        route_seed_values[(metric,route)].append(float(np.mean(values)))
                    paired.append(dict(suite=suite,comparison=a+' minus '+b,seed=str(seed),metric=metric,
                        **paired_route_bootstrap(strata)))
            if len(seeds)>1:
                for metric in ('archived_success','adherence_success','active_survival','full_horizon_survival'):
                    strata=defaultdict(list)
                    for (m,route),values in route_seed_values.items():
                        if m==metric: strata[route[0]].append(float(np.mean(values)))
                    paired.append(dict(suite=suite,comparison=a+' minus '+b,seed='mean_fixed_three',metric=metric,
                        **paired_route_bootstrap(strata)))
    write(output/'paired_route_intervals.csv',paired)
    trace_rows=[]
    if args.trace_root:
        for folder in sorted(args.trace_root.resolve().iterdir()):
            if (folder/'timeseries.npz').is_file():
                print(f'Analyzing saved traces: {folder.name}',flush=True)
                trace_rows.extend(analyze_traces(folder))
                for name in ('timeseries.npz','timeseries_metadata.json','evaluation_summary.csv'):
                    path=folder/name; inputs.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write(output/'trace_margin_sensitivity.csv',trace_rows)
        sensitivity=[]
        for suite in sorted({r['suite'] for r in trace_rows}):
            for margin in (0.,.1,.25):
                rows=[r for r in trace_rows if r['suite']==suite and r['margin_m']==margin]
                sensitivity.append(dict(suite=suite,margin_m=margin,n_with_valid_trace=len(rows),
                    scored_fraction_median=float(np.nanmedian([r['height_scored_fraction'] for r in rows])),
                    spatial_height_mae_cm=float(100*np.nanmean([r['spatial_height_mae_m'] for r in rows])),
                    reconstructed_height_success_pct=float(100*np.mean([r['reconstructed_height_success'] for r in rows])),
                    spatial_cte_rmse_cm=float(100*np.nanmean([r['spatial_cte_rmse_m'] for r in rows]))))
        write(output/'margin_summary.csv',sensitivity)
    for source in (Path(__file__), Path(__file__).with_name('report_contract.py')):
        inputs.append(dict(path=str(source.relative_to(root)),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    manifest=dict(contract=REPORT_CONTRACT_VERSION,inputs=inputs,
        adherence='archived task_success AND active temporal CTE RMSE <=0.10 m',
        spatial='Independent continuous-projection first-passage reconstruction, NOT exact training accumulator replay',
        safety='Until first positional arrival; nonarrivals use full evaluation horizon; postarrival failures separate',
        intervals='Stratified paired geometry bootstrap; observed actor seeds fixed, no claim of training-population CI',
        unchanged_archives=True,trace_missing='Only selected saved-trace suites are reconstructed; CSV-only records never imply exact spatial success')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Closed {len(summaries)} headline records and {len(trace_rows)} trace/margin records in {output}',flush=True)


if __name__=='__main__': main()
