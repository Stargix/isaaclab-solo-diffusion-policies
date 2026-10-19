"""Explicit opt-in CUDA regression; never require local checkpoint assets in CI."""
import os
import unittest
import torch
from scripts.diffusion_policy.runtime.contract import load_runtime_contract
from scripts.diffusion_policy.runtime.cuda_graph import CudaGraphDPPOSampler
from scripts.diffusion_policy.runtime.fixtures import load_fixture_bundle


@unittest.skipUnless(os.environ.get("SOLO_GRAPH_CHECKPOINT") and os.environ.get("SOLO_GRAPH_FIXTURE")
                     and torch.cuda.is_available(), "Requires explicit checkpoint/fixture and CUDA")
class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_runtime_contract(os.environ["SOLO_GRAPH_CHECKPOINT"], "cuda:0")
        cls.candidate = load_runtime_contract(os.environ["SOLO_GRAPH_CHECKPOINT"], "cuda:0")
        cls.graph = CudaGraphDPPOSampler(cls.candidate.policy)
        _, rows = load_fixture_bundle(os.environ["SOLO_GRAPH_FIXTURE"])
        cls.inputs = tuple(torch.as_tensor(rows[0][key], device="cuda:0", dtype=torch.float32).unsqueeze(0)
                           for key in ("proprio_history", "action_history", "goal_history"))

    def test_explicit_generator_preserves_outputs_and_rng(self):
        ref_rng = torch.Generator(device="cuda:0").manual_seed(123)
        graph_rng = torch.Generator(device="cuda:0").manual_seed(123)
        reference = self.reference.sampler.predict_action_denormalized(*self.inputs, generator=ref_rng)
        candidate = self.graph.predict_action_denormalized(*self.inputs, generator=graph_rng)
        self.assertTrue(torch.equal(reference, candidate))
        self.assertTrue(torch.equal(ref_rng.get_state(), graph_rng.get_state()))

    def test_returned_chunk_survives_later_replay(self):
        first = self.graph.predict_action_denormalized(*self.inputs)
        saved = first.clone()
        second = self.graph.predict_action_denormalized(*self.inputs)
        self.assertNotEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.equal(first, saved))

    def test_wrong_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.graph.predict_action_denormalized(self.inputs[0][:, :7], *self.inputs[1:])

    def test_bank_batch_preserves_outputs_and_rng(self):
        fresh = load_runtime_contract(os.environ["SOLO_GRAPH_CHECKPOINT"], "cuda:0")
        graph = CudaGraphDPPOSampler(fresh.policy, batch_size=32)
        inputs = tuple(x.repeat(32, 1, 1) for x in self.inputs)
        r1 = torch.Generator(device="cuda:0").manual_seed(123)
        r2 = torch.Generator(device="cuda:0").manual_seed(123)
        expected = self.reference.sampler.predict_action_denormalized(*inputs, generator=r1)
        actual = graph.predict_action_denormalized(*inputs, generator=r2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(r1.get_state(), r2.get_state()))


if __name__ == "__main__":
    unittest.main()
