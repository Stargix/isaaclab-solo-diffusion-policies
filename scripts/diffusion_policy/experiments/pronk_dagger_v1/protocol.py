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
