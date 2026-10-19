"""Simulator-free regression checks for the iterative experiment."""
from types import SimpleNamespace

import numpy as np
import torch

from scripts.diffusion_policy.experiments.pronk_dagger_v1.protocol import early_anchor, success_count
from scripts.diffusion_policy.experiments.pronk_dagger_v1.fit import batch_from, recovery_pools
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import recovery_anchors


def test_early_pool_excludes_expert_tail_and_student_targets():
    assert not early_anchor(39,40)
    assert early_anchor(40,40)
    assert early_anchor(55,40)
    assert not early_anchor(56,40)


def test_startup_and_settled_prefixes_have_valid_recovery_windows():
    for start in (0,25):
        for round_index in range(3):
            takeover=start+40+10*round_index
            mask=np.arange(300)<start
            mask |= np.arange(300)>=takeover
            anchors=recovery_anchors(mask)
            assert anchors[0]==takeover
            assert len([t for t in anchors if early_anchor(t,takeover)])==16
            assert all(mask[t:t+100].all() for t in anchors)


def test_success_is_not_survival_alone():
    report={'attempts':[dict(survived=True,cycles=0,mean_vx=.2,speed=.2),
                        dict(survived=True,cycles=3,mean_vx=.2,speed=.2)]}
    assert success_count(report)==1


def test_sampler_uses_only_explicit_split_and_fresh_windows():
    class Dataset:
        def __getitem__(self,index):
            return {'actions':torch.tensor([index],dtype=torch.float32)}
    batch=batch_from([(Dataset(),[1,3,7])],128,np.random.default_rng(42),'cpu')
    assert set(batch['actions'].flatten().tolist())=={1.,3.,7.}


def test_recovery_split_and_early_indices_preserve_augmentation(monkeypatch,tmp_path):
    import h5py
    import scripts.diffusion_policy.experiments.pronk_dagger_v1.fit as fit
    path=tmp_path/'recovery.hdf5'
    with h5py.File(path,'w') as f:
        for i in range(40):
            demo=f.create_group(f'data/demo_{i}')
            demo.attrs['cohort']=0 if i<30 else 3
            demo.attrs['takeover_step']=40
    samples=[SimpleNamespace(demo_idx=i,anchor_step=t) for i in range(40) for t in (40,55,56)]
    ds=SimpleNamespace(demos=[SimpleNamespace(demo_name=f'demo_{i}') for i in range(40)],
                       samples=samples,_symmetry_count=4)
    ds.sample_indices_for_demos=lambda ids:[j*4+k for j,s in enumerate(samples) if s.demo_idx in ids for k in range(4)]
    monkeypatch.setattr(fit,'make_dataset',lambda *args:ds)
    monkeypatch.setattr(fit,'audit_action_support',lambda *args:{})
    tr,early,val,_=recovery_pools([path],{},None)
    assert len(tr[0][1])==30*3*4
    assert len(early[0][1])==30*2*4
    assert set(tr[0][1]).isdisjoint(val[0][1])


def test_dependency_closure_finds_nested_missing_module(monkeypatch,tmp_path):
    from scripts.diffusion_policy.experiments.pronk_dagger_v1 import dependencies as dep
    package=tmp_path/'fixture'
    package.mkdir()
    (package/'__init__.py').write_text('',encoding='utf-8')
    (package/'v8.py').write_text('from .v7 import Config\n',encoding='utf-8')
    (package/'v7.py').write_text('from .v6 import Config\n',encoding='utf-8')
    monkeypatch.setattr(dep,'MODULE_ROOTS',{'fixture':package})
    monkeypatch.setattr(dep,'SEEDS',('fixture.v8',))
    import pytest
    with pytest.raises(FileNotFoundError,match='fixture.v6'):
        dep.source_closure()


def test_collection_registers_v8_without_global_initializer():
    import ast
    from pathlib import Path
    collector=Path(__file__).resolve().parents[1]/'pronk_intervention_v1/collect.py'
    tree=ast.parse(collector.read_text(encoding='utf-8'))
    registrations=[node for node in ast.walk(tree) if isinstance(node,ast.Call)
                   and isinstance(node.func,ast.Attribute) and node.func.attr=='register']
    assert any(any(k.arg=='id' and isinstance(k.value,ast.Constant)
                   and k.value.value=='solo12-pronk-v8-v0' for k in node.keywords)
               for node in registrations)


def test_action_support_audit_keeps_extreme_history_and_teacher_corrections(tmp_path):
    import h5py
    from scripts.diffusion_policy.experiments.pronk_dagger_v1.support import audit_action_support
    actions=np.zeros((220,12),dtype=np.float32)
    actions[49,0]=2.78  # learner context, not an expert correction
    actions[54,1]=-2.5  # valid future teacher correction, outside old range
    before=actions.copy()
    samples=[SimpleNamespace(demo_idx=0,anchor_step=t) for t in (50,60,80)]
    ds=SimpleNamespace(demos=[SimpleNamespace(actions=actions,demo_name='demo_0')],
                       samples=samples,execution_offset=8,future_horizon=8)
    path=tmp_path/'support.hdf5'
    with h5py.File(path,'w') as f:
        group=f.create_group('data/demo_0')
        group.attrs['takeover_step']=50
        group.create_dataset('diagnostics/expert_control',data=np.arange(220)>=50)
    stats=SimpleNamespace(action=SimpleNamespace(min=np.full(12,-1.),max=np.ones(12)))
    audit=audit_action_support(ds,stats,path)
    assert audit['past_and_future_windows']==1
    assert audit['past_only_windows']==1
    assert audit['future_only_windows']==0
    assert audit['independent_outside_frames']==2
    assert audit['outside_components']==2
    assert audit['expert_outside_frames']==1
    assert audit['student_outside_frames']==1
    assert audit['early_windows']==2
    assert audit['early_future_outside_windows']==1
    assert audit['excluded_windows']==0
    assert audit['outside_window_fraction']>.05
    assert ds.samples is samples
    np.testing.assert_array_equal(actions,before)


def test_sampler_revision_preserves_weights_normalization_and_original_config():
    from scripts.diffusion_policy.experiments.pronk_dagger_v1.support import unclipped_checkpoint
    original=dict(config={'diffusion':{'clip_sample':True}},
                  noise_scheduler_config={'clip_sample':True},
                  model_state_dict={'weight':torch.ones(2)},normalizer_stats={'fixed':'original'})
    revised=unclipped_checkpoint(original)
    assert original['config']['diffusion']['clip_sample'] is True
    assert original['noise_scheduler_config']['clip_sample'] is True
    assert revised['config']['diffusion']['clip_sample'] is False
    assert revised['noise_scheduler_config']['clip_sample'] is False
    assert revised['model_state_dict'] is original['model_state_dict']
    assert revised['normalizer_stats'] is original['normalizer_stats']


def test_unclipped_ddpm_can_represent_corrective_labels_outside_original_range():
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    label=torch.full((1,16,12),2.78)
    sample=torch.zeros_like(label)
    for clip,expected in ((True,1.),(False,2.78)):
        scheduler=DDPMScheduler(num_train_timesteps=10,prediction_type='sample',clip_sample=clip)
        output=scheduler.step(label,0,sample).prev_sample
        torch.testing.assert_close(output,torch.full_like(label,expected))


def test_reuse_rejects_old_pilot_instead_of_silently_mixing_protocols(tmp_path):
    import json
    import pytest
    from scripts.diffusion_policy.experiments.pronk_dagger_v1.support import validate_reused_collection
    (tmp_path/'collection.json').write_text(json.dumps(dict(base_sha256='base',expert_sha256='expert',
        seed=443,mode='takeover_collection',dagger_round=None)),encoding='utf-8')
    with pytest.raises(ValueError,match='round-zero'):
        validate_reused_collection(tmp_path,'base','expert')


def test_reuse_complete_collection_is_read_only_and_validates_masks(tmp_path):
    import h5py
    import json
    import pytest
    from scripts.diffusion_policy.experiments.pronk_dagger_v1.support import validate_reused_collection
    rows=[]
    path=tmp_path/'recovery.hdf5'
    with h5py.File(path,'w') as f:
        for cohort in range(4):
            for env in range(18):
                speed=(.2,.35,.5)[env//6]
                start=0 if env%2==0 else 25
                takeover=start+(20,30,40)[(env%6)//2]
                name=f'demo_{len(rows)}'
                rows.append(dict(cohort=cohort,env=env,speed=speed,mean_vx=speed,survived=True,
                                 cycles=3,saved=True,demo=name,student_start_step=start,takeover_step=takeover))
                d=f.create_group('data/'+name)
                d.attrs['cohort']=cohort; d.attrs['takeover_step']=takeover
                d.create_dataset('actions',data=np.zeros((300,12)))
                d.create_dataset('diagnostics/expert_control',data=(np.arange(300)<start)|(np.arange(300)>=takeover))
                d.create_dataset('training_anchor_steps',data=np.arange(takeover,201))
    report=dict(base_sha256='base',expert_sha256='expert',seed=443,mode='takeover_collection',
                dagger_round=0,gate_passed=True,attempts=rows)
    (tmp_path/'collection.json').write_text(json.dumps(report),encoding='utf-8')
    before=path.read_bytes()
    assert validate_reused_collection(tmp_path,'base','expert')==path
    assert path.read_bytes()==before
    with h5py.File(path,'r+') as f:
        f['data/demo_0/diagnostics/expert_control'][100]=False
    with pytest.raises(ValueError,match='control mask'):
        validate_reused_collection(tmp_path,'base','expert')
