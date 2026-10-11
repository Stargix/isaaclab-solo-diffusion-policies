"""No-simulator regressions for the speed-partition ablation."""
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/diffusion_policy'))
from train.data.dataset import SpatialHindsightDataset
from selection import eligible_anchors


def motion(speeds):
    root = np.zeros((len(speeds) + 1, 3), dtype=np.float32)
    root[1:, 0] = np.cumsum(np.asarray(speeds) * .02)
    command = np.zeros_like(root)
    command[:, 0] = .8
    return root, command


def fixture(path, anchors=None):
    with h5py.File(path, 'w') as file:
        data = file.create_group('data')
        data.attrs.update(skill_names=np.asarray(['walk'], dtype='S32'),
                          control_rate_hz=50.,
                          convention='aligned: obs[t] is the proprioceptive state BEFORE executing actions[t]',
                          condition_schema='velocity_xyyaw_plus_desired_base_height_v1')
        demo = data.create_group('demo_0')
        obs = demo.create_group('obs')
        for key, dim in {'joint_pos':12, 'joint_vel':12, 'base_ang_vel':3,
                         'projected_gravity':3, 'last_action':12, 'root_pos_w':3,
                         'root_quat_w':4, 'command_speed':3, 'desired_base_height':1}.items():
            obs[key] = np.zeros((299, dim), dtype=np.float32)
        obs['root_quat_w'][:, 0] = 1
        obs['root_pos_w'][:, 0] = np.arange(299) * .02 * .8
        obs['desired_base_height'][:] = .2932
        demo['actions'] = np.zeros((299, 12), dtype=np.float32)
        demo['dones'] = np.r_[np.zeros(298, dtype=bool), True]
        demo['skill_idx'] = np.zeros(299, dtype=np.int8)
        if anchors is not None:
            demo['training_anchor_steps'] = np.asarray(anchors, dtype=np.int16)


class SelectionTests(unittest.TestCase):
    def test_fast_only_and_exact_threshold_not_lowered(self):
        root, command = motion(np.full(298, .8))
        self.assertTrue(len(eligible_anchors(root, command, 'walk')))
        root, command = motion(np.full(298, .599))
        self.assertEqual(len(eligible_anchors(root, command, 'walk')), 0)

    def test_zero_then_fast_is_not_a_walk_start_example(self):
        root, command = motion(np.r_[np.zeros(50), np.linspace(0, .8, 25), np.full(223, .8)])
        command[:50] = 0
        anchors = eligible_anchors(root, command, 'walk')
        self.assertTrue(len(anchors))
        self.assertTrue(np.all(anchors - 9 > 50))
        self.assertFalse(any(a < 75 for a in anchors))

    def test_fast_average_does_not_hide_slow_future(self):
        # Mean over 2 s remains >.6 but there is a .2 m/s segment in the future.
        root, command = motion(np.r_[np.full(75, 1.), np.full(25, .2), np.full(198, 1.)])
        self.assertNotIn(40, eligible_anchors(root, command, 'walk'))

    def test_delayed_action_previous_tick_is_checked(self):
        root, command = motion(np.full(298, .8))
        command[70] = 0
        # anchor79 starts proprio history71 but delayed action history70.
        self.assertNotIn(79, eligible_anchors(root, command, 'walk'))

    def test_pronk_cap_and_native_crouch_starts(self):
        root, command = motion(np.full(298, .3))
        self.assertIn(0, eligible_anchors(root, command, 'pronk'))
        self.assertIn(0, eligible_anchors(root, command, 'crouch'))
        root, command = motion(np.full(298, .7))
        self.assertEqual(len(eligible_anchors(root, command, 'pronk')), 0)

    def test_loader_selection_multiplicity_and_normalizer(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            path = Path(directory) / 'selected.hdf5'
            fixture(path, [80, 80, 81])
            with h5py.File(path, 'r+') as file:
                # A deliberately huge unused action must not set normalization.
                file['data/demo_0/actions'][200] = 100
                file['data/demo_0/obs/last_action'][201] = 100
            loader = SpatialHindsightDataset([str(path)], include_padded_starts=True,
                                             symmetry_mode='quadruped', goal_representation='hindsight_geom_profile16')
            self.assertEqual([int(s.anchor_step) for s in loader.samples], [80, 80, 81])
            self.assertEqual(len(loader), 12)
            low, high = loader._exact_action_range(loader.demos)
            self.assertTrue(np.all(high == 0))
            self.assertNotIn(200, loader._normalization_rows(loader.demos[0]))
            self.assertEqual(tuple(loader[0]['goal_hist'].shape), (8, 16))
            self.assertEqual(tuple(loader[0]['actions'].shape), (16, 12))
            self.assertGreaterEqual(float(loader[0]['goal_hist'][:, -1].min()), .6)
            self.assertEqual(loader.sample_indices_for_demos([0]), list(range(12)))

    def test_legacy_without_selection_keeps_original_200_anchors(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            path = Path(directory) / 'legacy.hdf5'
            fixture(path)
            loader = SpatialHindsightDataset([str(path)], include_padded_starts=True,
                                             goal_representation='hindsight_geom_profile16')
            self.assertEqual(len(loader.samples), 200)
            np.testing.assert_array_equal(loader._normalization_rows(loader.demos[0]), np.arange(299))

    def test_invalid_or_unsorted_anchor_metadata_aborts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'scratch') as directory:
            for index, anchors in enumerate(([10, 9], [200], [-1])):
                path = Path(directory) / f'bad_{index}.hdf5'
                fixture(path, anchors)
                with self.assertRaises(ValueError):
                    SpatialHindsightDataset([str(path)], include_padded_starts=True,
                                             goal_representation='hindsight_geom_profile16')


if __name__ == '__main__':
    unittest.main()
