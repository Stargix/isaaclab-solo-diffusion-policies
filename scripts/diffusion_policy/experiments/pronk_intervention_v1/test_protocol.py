import numpy as np
from scripts.diffusion_policy.experiments.pronk_intervention_v1.protocol import recovery_anchors, gate_passes, ROOT


def test_root():
    assert (ROOT/'isaaclab.bat').is_file()


def test_takeover_future_mask():
    mask = np.ones(300, dtype=bool)
    mask[25:65] = False
    a = recovery_anchors(mask)
    assert np.array_equal(a, np.arange(65,201))
    # The first recovery anchor still contains actual learner past context.
    assert not mask[a[0]-8:a[0]].all()
    for t in a:
        assert mask[t:t+100].all()


def test_later_student_interval_cannot_leak_into_hindsight():
    mask=np.ones(300,dtype=bool)
    mask[170]=False
    a=recovery_anchors(mask)
    assert 70 in a and 71 not in a and 170 not in a and 171 in a


def test_gate_does_not_hide_a_failed_speed():
    rows=[dict(speed=v,survived=True,cycles=3,mean_vx=v) for v in (.2,.35,.5) for _ in range(6)]
    assert gate_passes(rows)
    rows[-1]['survived']=False
    rows[-2]['survived']=False
    assert not gate_passes(rows)
    assert not gate_passes([])


def test_dagger_student_phase_is_per_env_not_global_tick():
    """Late student starters must replan at phase 0, not reuse global tick%4."""
    import torch
    starts=torch.tensor([0,25,0,25])
    tick=25
    phases=(tick-starts).clamp(min=0)
    assert phases.tolist()==[25,0,25,0]
    assert (phases%4).tolist()==[1,0,1,0]
    need=((phases%4)==0)
    assert need.tolist()==[False,True,False,True]


def test_range_mask_checks_past_and_future_without_editing_labels():
    from types import SimpleNamespace
    from scripts.diffusion_policy.experiments.pronk_intervention_v1.finetune import retain_representable_targets
    actions=np.zeros((2020,12),dtype=np.float32)
    actions[1000,0]=2.
    ds=SimpleNamespace(demos=[SimpleNamespace(actions=actions,demo_name='demo_0')],
        samples=[SimpleNamespace(demo_idx=0,anchor_step=t) for t in range(8,2008)],
        execution_offset=8,future_horizon=8)
    stats=SimpleNamespace(action=SimpleNamespace(min=np.full(12,-1.),max=np.ones(12)))
    audit=retain_representable_targets(ds,stats)
    assert [r['anchor'] for r in audit['excluded_anchors']]==list(range(993,1009))
    assert actions[1000,0]==2.
