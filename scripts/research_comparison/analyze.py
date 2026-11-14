"""Validate complete paired suite outputs and separate startup from tracking."""
import argparse
import json
from pathlib import Path
import numpy as np
from .suite import sha


def distance(points, route):
    a, d = route[:-1], np.diff(route, axis=0)
    norm = (d*d).sum(1)
    keep = norm > 1e-12
    a,d,norm = a[keep],d[keep],norm[keep]
    if not len(a): raise ValueError('Degenerate route')
    t = np.clip(((points[:,None]-a)*d).sum(2)/norm,0,1)
    return np.linalg.norm(points[:,None]-a-t[:,:,None]*d,axis=2).min(1)


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text()); data=[]; keys={}
    for job in protocol['jobs']:
        if job['group']!='eval': continue
        done=root/'state'/(job['id']+'.json')
        if not done.exists(): raise ValueError(f'Pending job: {job["id"]}; refusing incomplete comparison')
        record=json.loads(done.read_text())
        if any(sha(p)!=h for p,h in record['outputs'].items()): raise ValueError('Output checksum changed')
        folder=Path(job['output'])
        # Read per-episode records and validate trace metric rather than aggregate across resets.
        rows=json.loads((folder/'episode_metrics.json').read_text())
        meta=json.loads((folder/'timeseries_metadata.json').read_text())
        z=np.load(folder/'timeseries.npz')
        if len(rows)!=job['expected']: raise ValueError('Wrong condition count')
        for i,r in enumerate(rows):
            n=int(r['active_steps']); valid=z['valid'][:n,i].astype(bool)
            xy=z['positions_xy'][:n,i][valid]; progress=z['progress_m'][:n,i][valid]
            path=np.array(meta['scenarios'][i]['path_xy']); err=distance(xy,path)
            rmse=float(np.sqrt(np.mean(err**2))) if len(err) else None
            if rmse is not None and abs(rmse-r['active_cross_track_rmse_m'])>1e-5:
                raise ValueError('CTE reconstruction differs from official metric')
            after=progress>=.5
            key=(job['suite'],job['stage'],job['variant'],r['path_shape'],r['repeat'],
                 r['requested_speed'],r['requested_height'],r['transition_direction'])
            bucket=keys.setdefault((job['actor'],job['suite'],job['stage']),set())
            if key in bucket: raise ValueError('Duplicate condition')
            bucket.add(key)
            if job['suite']=='composition' and int(r['target_height_changes'])!=4:
                raise ValueError('Repeated profile did not contain four changes')
            height=np.abs(z['achieved_height_m'][:n,i][valid]-z['required_height_m'][:n,i][valid])
            length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
            boundaries=(np.arange(.8,length-1e-4,.8) if job['suite']=='composition'
                        else ([float(r['transition_fraction'])*length] if r['transition_direction'] else []))
            plateau=np.ones(len(progress),dtype=bool)
            for boundary in boundaries: plateau &= np.abs(progress-boundary)>.25
            data.append(dict(actor=job['actor'],suite=job['suite'],stage=job['stage'],key=key,
                success=r['task_success'],adherence=bool(r['task_success'] and rmse is not None and rmse<=.1),
                survived=r['survived'],arrived=r['route_arrived'],initialization_failure=r['initialization_failure'],
                active_steps=n,total_cte_m=rmse,after_half_m_samples=int(after.sum()),
                after_half_m_cte_m=float(np.sqrt(np.mean(err[after]**2))) if after.any() else None,
                initial_distance_m=float(err[0]) if len(err) else None,
                full_active_height_mae_m=float(height.mean()) if len(height) else None,
                plateau_height_mae_m=float(height[plateau].mean()) if plateau.any() else None,
                arrival_speed_error_mps=abs(r['route_arrival_mean_speed_m_s']-r['requested_speed']) if r['route_arrived'] else None))
    for suite in ['core','composition']:
        for stage in sorted({k[2] for k in keys if k[1]==suite}):
            sets=[v for k,v in keys.items() if k[1:]==(suite,stage)]
            if len(sets)!=len(protocol['actors']) or any(s!=sets[0] for s in sets):
                raise ValueError('Condition sets do not match across all actors')
    summary={}
    for actor,suite,stage in keys:
        rows=[r for r in data if (r['actor'],r['suite'],r['stage'])==(actor,suite,stage)]
        group=dict(n=len(rows))
        for k in ['success','adherence','survived','arrived','initialization_failure']: group[k]=sum(bool(r[k]) for r in rows)
        for k in ['total_cte_m','after_half_m_cte_m','full_active_height_mae_m','plateau_height_mae_m','arrival_speed_error_mps']:
            values=[r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
            group[k]=dict(n=len(values),mean=float(np.mean(values)) if values else None)
        summary[f'{actor}/{suite}/stage{stage}']=group
    curves={}
    for job in protocol['jobs']:
        if job['group']=='train':
            done=root/'state'/(job['id']+'.json')
            if not done.exists(): raise ValueError('Missing training result')
            record=json.loads(done.read_text())
            if any(sha(p)!=h for p,h in record['outputs'].items()): raise ValueError('Training output checksum changed')
            rows=[json.loads(line) for line in (Path(job['output'])/'metrics.jsonl').read_text().splitlines()]
            curves[job['id']]=[dict(iteration=r['iteration'],transitions=r['total_physics_steps'],
                task_success=r.get('Episode/success_rate'),base_contact=r.get('Episode/base_contact_rate'),
                reward=r.get('Episode/return_mean')) for r in rows]
    result=dict(summary=summary,episodes=data,training_curves=curves,
        caveat='Post-0.5m window is diagnostic and conditions on reaching that progress; official full scores retained. Geometry draws, not all crossed episodes, are independent.')
    (root/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--protocol',type=Path,required=True)
    analyze(p.parse_args().protocol.resolve())
