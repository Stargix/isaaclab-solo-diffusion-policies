import unittest
import numpy as np
from scripts.diffusion_policy.runtime.safety import CommandSupervisor, SafetyFault
from scripts.diffusion_policy.sim2sim.perturbations import PROTOCOLS


class SafetyTests(unittest.TestCase):
    def args(self):
        return dict(q=np.zeros(12), dq=np.zeros(12), target=np.zeros(12), now=1., state_stamp=.99, plan_stamp=.98, sequence=0)

    def guard(self):
        return CommandSupervisor(np.full(12, -1.), np.ones(12))

    def test_valid_target_is_copied_not_clamped(self):
        a = self.args(); a["target"][:] = .5
        out = self.guard().admit(**a); a["target"][:] = .7
        np.testing.assert_array_equal(out, np.full(12, .5))

    def test_faults_latch_and_never_send(self):
        faults = {"stale": {"state_stamp": .9}, "delay": {"plan_stamp": .8},
                  "nan_state": {"q": np.full(12, np.nan)}, "nan_target": {"target": np.full(12, np.nan)},
                  "inf_velocity": {"dq": np.full(12, np.inf)}, "shape": {"target": np.zeros(11)},
                  "limit": {"target": np.full(12, 1.01)}, "velocity": {"dq": np.full(12, 101.)},
                  "future": {"state_stamp": 1.01}, "nan_clock": {"now": float("nan")}}
        faults.update({"none_clock": {"now": None}, "string_target": {"target": ["bad"]*12},
                       "boolean_sequence": {"sequence": True}})
        for name, patch in faults.items():
            with self.subTest(name=name):
                guard = self.guard(); a = dict(self.args(), **patch); sent = []
                with self.assertRaises(SafetyFault):
                    sent.append(guard.admit(**a))
                self.assertEqual(sent, [])
                with self.assertRaisesRegex(SafetyFault, "Latched"):
                    guard.admit(**self.args())

    def test_old_sequence_and_reversed_clock_rejected(self):
        for patch in ({}, {"now": .9, "state_stamp": .89, "plan_stamp": .88, "sequence": 1}):
            guard = self.guard(); guard.admit(**self.args())
            with self.assertRaises(SafetyFault):
                guard.admit(**dict(self.args(), **patch))

    def test_impulses_are_exact_and_not_periodic_replay(self):
        p = PROTOCOLS["lateral_impulse"]
        self.assertEqual(p.impulse(250), .2); self.assertEqual(p.impulse(600), -.2)
        self.assertEqual(p.impulse(249), 0); self.assertEqual(p.impulse(601), 0)
        self.assertEqual(PROTOCOLS["nominal"].impulse(250), 0)


if __name__ == "__main__":
    unittest.main()
