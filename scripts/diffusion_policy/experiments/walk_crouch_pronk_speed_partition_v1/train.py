"""Launch the corrected speed-partition ablation, not the overlapping v1 data."""
import argparse
from pathlib import Path
import sys

from prepare import ROOT, INDEX, verify

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'walk_crouch_pronk_v1'))
from run_phase_a import launch_verified

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    output = ROOT / 'scripts/diffusion_policy/runs/walk_crouch_pronk_speed_partition_v1_k10'
    raise SystemExit(launch_verified(verify(), INDEX, output, args.device,
                                    preflight_only=args.preflight_only))
