"""Isolated, bounded expert-takeover imitation pilot; no changes to WC/WCT."""
from pathlib import Path
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
BASE_SHA = '2098f85b5326c6562e1ddf51c3a4687fb2079bca6849dfdcfd1e45b830fba07b'
EXPERT = ROOT / 'logs/rsl_rl/solo12_rsl_rl_pronk_runs/2026-10-11_16-18-25_pronk_v8_signed_bounded_exposure_seed42/model_2999.pt'
EXPERT_SHA = '73c572ffa444576e6afad230dac1a6d0fd91db59050469af60737d845679807d'
NOMINAL = ROOT / 'scripts/diffusion_policy/data/datasets/walk_crouch_pronk_v1_aligned/pronk_full.hdf5'
DEFAULT_OUTPUT = ROOT / 'scratch/pronk_intervention_v1_20261019'
STEPS = 300
START = 25
PREFIXES = (20, 30, 40)
SPEEDS = (.20, .35, .50)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path, expected_sha=None):
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    allowed = [np.ndarray, np.dtype,
               (np.core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct'),
               type(np.dtype(np.float32)), type(np.dtype(np.float64)), type(np.dtype(np.int64))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location='cpu', weights_only=True)


def recovery_anchors(expert_control, history=8, horizon=16, goal_steps=100):
    """Future actions and future task outcome must be expert-controlled.

    Past states/actions may belong to the learner: these are genuine context,
    not hypothetical expert labels. Action targets before the execution offset
    describe the actual observed past; executable targets are expert actions.
    """
    mask = np.asarray(expert_control, dtype=bool)
    future = horizon-history
    anchors = []
    for t in range(max(0, len(mask)-goal_steps+1)):
        end = max(t+future, t+goal_steps)
        if end <= len(mask) and mask[t:end].all():
            anchors.append(t)
    return np.asarray(anchors, dtype=np.int64)


def gate_passes(rows):
    if not rows:
        return False
    good = [r['survived'] and r['cycles'] >= 2 and abs(r['mean_vx']-r['speed']) <= .10 for r in rows]
    # Require coverage in every speed group, not merely a good overall average.
    return all(np.mean([g for g,r in zip(good,rows) if np.isclose(r['speed'],v)]) >= .80 for v in SPEEDS)
