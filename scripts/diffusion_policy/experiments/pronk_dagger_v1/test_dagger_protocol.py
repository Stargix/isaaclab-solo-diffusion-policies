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
    monkeypatch.setattr(fit,'retain_representable_targets',lambda *args:{})
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
