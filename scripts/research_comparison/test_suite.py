import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts.gaussian_chunk_rl.config import GaussianPPOConfig
from scripts.gaussian_chunk_rl.policy import load_bc_policy
from scripts.gaussian_chunk_rl.tests.test_gaussian_ppo import make_policy, inputs
from .suite import prepare, sha
from .analyze import distance, analyze


def test_initialization_only_changes_weights(monkeypatch):
    prior=make_policy().policy
    cfg=asdict(prior.cfg)
    source=dict(algorithm='deterministic_chunk_bc',config=dict(model=cfg,
        dataset=dict(history=8,prediction_horizon=16,execution_offset=8,
            goal_representation='hindsight_geom_profile16',include_padded_starts=True,dt=.02)),
        normalizer_stats=prior.normalizer_stats,ema_model_state_dict=prior.state_dict())
    monkeypatch.setattr('scripts.gaussian_chunk_rl.policy.load_training_checkpoint', lambda *a,**k:source)
    torch.manual_seed(123); _,bc=load_bc_policy('unused','cpu',GaussianPPOConfig())
    rng_bc=torch.get_rng_state()
    torch.manual_seed(123); _,random=load_bc_policy('unused','cpu',GaussianPPOConfig(),initialization='random')
    assert torch.equal(rng_bc,torch.get_rng_state())  # subsequent critic initialization remains paired
    assert any(not torch.equal(a,b) for a,b in zip(bc.parameters(),random.parameters()))
    assert torch.equal(bc.log_std,random.log_std)
    assert bc.policy.normalizer_stats is random.policy.normalizer_stats
    torch.manual_seed(123); _,repeat=load_bc_policy('unused','cpu',GaussianPPOConfig(),initialization='random')
    for a,b in zip(random.parameters(),repeat.parameters()): torch.testing.assert_close(a,b,rtol=0,atol=0)
    x=inputs()
    for policy in [bc,random]:
        sample=policy.sample(*x)
        assert torch.isfinite(sample.denormalized_trajectory).all()


def test_protocol_is_frozen_bounded_and_matched(tmp_path):
    artifact=tmp_path/'artifact.pt'; artifact.write_bytes(b'fixture')
    cfg=dict(inputs={k:str(artifact) for k in ['bc_prior','reference','walk','crouch']},
        actors=[dict(name='h42',kind='hierarchy',horizon=10,checkpoint=str(artifact)),
                dict(name='wc42',kind='policy',horizon=4,checkpoint=str(artifact))])
    config=tmp_path/'config.json'; config.write_text(json.dumps(cfg))
    out=tmp_path/'protocol'; prepare(config,out)
    p=json.loads((out/'protocol.json').read_text())
    assert len([j for j in p['jobs'] if j['group']=='train'])==6
    for a in p['actors']:
        jobs=[j for j in p['jobs'] if j.get('actor')==a['name']]
        assert sum(j['expected'] for j in jobs if j['suite']=='core')==168
        assert sum(j['expected'] for j in jobs if j['suite']=='composition')==84
        assert max(j['expected'] for j in jobs)<=36
        assert all(j['stage']==50 for j in jobs)
    with pytest.raises(FileExistsError): prepare(config,out)


def test_segment_distance_decomposition():
    route=np.array([[0.,0.],[1.,0.]])
    np.testing.assert_allclose(distance(np.array([[.5,0.],[.5,.2],[-.3,.4]]),route),[0,.2,.5])


def test_analysis_requires_complete_paired_outputs(tmp_path):
    jobs=[]
    (tmp_path/'state').mkdir()
    for actor in ['h','wc']:
        for suite in ['core','composition']:
            name=f'{actor}_{suite}'; out=tmp_path/name; out.mkdir()
            job=dict(id=name,group='eval',actor=actor,suite=suite,stage=50,variant=0,expected=1,output=str(out))
            jobs.append(job)
            row=dict(active_steps=8,active_cross_track_rmse_m=.02,target_height_changes=4,
                     path_shape='straight',repeat=0,requested_speed=.5,requested_height=.2932,
                     transition_direction=None,transition_fraction=None,task_success=True,survived=True,
                     route_arrived=True,initialization_failure=False,route_arrival_mean_speed_m_s=.5)
            (out/'episode_metrics.json').write_text(json.dumps([row]))
            (out/'timeseries_metadata.json').write_text(json.dumps(dict(scenarios=[dict(path_xy=[[0,0],[4,0]])])))
            np.savez(out/'timeseries.npz',valid=np.ones((8,1),bool),
                positions_xy=np.array([[[x,.02]] for x in np.linspace(0,3.8,8)]),
                progress_m=np.linspace(0,3.8,8)[:,None],achieved_height_m=np.full((8,1),.29),
                required_height_m=np.full((8,1),.2932))
            outputs={str(p):sha(p) for p in out.iterdir()}
            (tmp_path/'state'/(name+'.json')).write_text(json.dumps(dict(outputs=outputs)))
    protocol=dict(jobs=jobs,actors=[dict(name='h'),dict(name='wc')])
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    analyze(tmp_path)
    assert json.loads((tmp_path/'analysis.json').read_text())['summary']['h/core/stage50']['success']==1
    (tmp_path/'state'/'wc_core.json').unlink()
    with pytest.raises(ValueError,match='Pending'): analyze(tmp_path)
