"""Frozen engineering budget for isolated iterative pronk imitation."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ROUNDS = 3
UPDATES = 4000
BATCH_SIZE = 128
LEARNING_RATE = 1e-5
NOMINAL_SHA = '2d4c3947e2403841883e8c734d1a44fec41908e855e7df9f829e8b887e489ca4'


def early_anchor(anchor_step, takeover_step):
    """First 16 valid correction anchors, not the settled expert tail."""
    return takeover_step <= anchor_step < takeover_step + 16


def success_count(report):
    return sum(r['survived'] and r['cycles'] >= 2
               and abs(r['mean_vx'] - r['speed']) <= .10 for r in report['attempts'])


def qualified(report):
    # Small-screen qualification only, not a paper success-rate estimate.
    return bool(report['gate_passed'])


def enforce_collection_gate(*, evaluate, fixed_exposure):
    """A performance diagnostic must not veto valid iterative recovery data."""
    return evaluate or not fixed_exposure


def exposure_increment(round_index, *, fixed_exposure):
    if round_index not in range(ROUNDS):
        raise ValueError('Invalid DAgger round')
    return 0 if fixed_exposure else 10*round_index


def usable_collection_counts(path):
    """Keep episode separation and the existing minimums, not a success-rate gate."""
    import h5py
    with h5py.File(path,'r') as f:
        cohorts=[int(d.attrs['cohort']) for d in f['data'].values()]
    if any(c not in range(4) for c in cohorts):
        raise ValueError('Unexpected recovery cohort')
    train=sum(c<3 for c in cohorts)
    validation=sum(c==3 for c in cohorts)
    return dict(train=train,validation=validation,usable=train>=30 and validation>=10)
