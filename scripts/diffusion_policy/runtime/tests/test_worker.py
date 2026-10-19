import time
import unittest
import torch
import numpy as np
from scripts.diffusion_policy.runtime.worker import InferenceWorker


class Sampler:
    def predict_action_denormalized(self,p,a,g):
        return a
    def executable_chunk(self,trajectory,n):
        return trajectory[:,:n]


class WorkerTests(unittest.TestCase):
    def test_copied_inputs_and_distinct_ordered_slots(self):
        worker=InferenceWorker(Sampler(),injected_delay_s=.02)
        p=torch.zeros(1,8,30);a=torch.arange(96,dtype=torch.float32).reshape(1,8,12);g=torch.zeros(1,8,16)
        expected=a[0,:4].numpy().copy()
        try:
            worker.submit(0,time.perf_counter(),(p,a,g)); a.zero_()
            result=worker.receive(0,time.perf_counter()+1.)
            np.testing.assert_array_equal(result.actions,expected)
            self.assertFalse(result.actions.flags.writeable)
        finally: self.assertTrue(worker.close())

    def test_late_plan_is_not_dispatched(self):
        worker=InferenceWorker(Sampler(),injected_delay_s=.05)
        try:
            worker.submit(0,time.perf_counter(),(torch.zeros(1,8,30),torch.zeros(1,8,12),torch.zeros(1,8,16)))
            with self.assertRaises(TimeoutError): worker.receive(0,time.perf_counter()+.005)
        finally: self.assertTrue(worker.close())

    def test_wrong_sequence_is_rejected(self):
        worker=InferenceWorker(Sampler())
        try:
            worker.submit(4,time.perf_counter(),(torch.zeros(1,8,30),torch.zeros(1,8,12),torch.zeros(1,8,16)))
            with self.assertRaises(ValueError): worker.receive(0,time.perf_counter()+1.)
        finally: self.assertTrue(worker.close())


if __name__=='__main__':unittest.main()
