import unittest
import torch
from scripts.diffusion_policy.runtime.host_buffers import HostBufferSampler


class FakeSampler:
    device = torch.device("cpu")

    def predict_action_denormalized(self, p, a, g, **kwargs):
        return a[:, :4].clone()

    def executable_chunk(self, trajectory, actions):
        return trajectory[:, :actions]


class HostTests(unittest.TestCase):
    def test_shape_and_distinct_chunk_slots_are_preserved(self):
        adapter = HostBufferSampler(FakeSampler())
        p = torch.zeros(1,8,30); a = torch.arange(96,dtype=torch.float32).reshape(1,8,12); g = torch.zeros(1,8,16)
        chunk = adapter.executable_chunk(adapter.predict_action_denormalized(p,a,g), 4)
        self.assertEqual(chunk.device.type, "cpu")
        torch.testing.assert_close(chunk, a[:, :4], rtol=0, atol=0)
        self.assertNotEqual(float(chunk[0,0,0]),float(chunk[0,3,0]))

    def test_unexpected_dtype_is_not_silently_converted(self):
        with self.assertRaises(ValueError):
            HostBufferSampler(FakeSampler()).predict_action_denormalized(torch.zeros(1,8,30,dtype=torch.float64),
                torch.zeros(1,8,12),torch.zeros(1,8,16))


if __name__ == "__main__":
    unittest.main()
