import copy
import unittest
import torch
from .parallel_report import metric,first_update
P=dict(same_state_rmse_absolute=1e-8,same_state_rmse_relative=1e-5,write_relative_l2=.01,write_cosine=.9999)


def audit():
    return dict(raw_gradients={'large':torch.ones(1000),'small':torch.ones(2)},actual_deltas={'w':torch.ones(2)},
        optimizer={'state':{'w':{'momentum_buffer':torch.ones(2),'step':torch.tensor(1)}},
                   'model_statistics':{'layer':{'input_cov':torch.eye(2),'_total_forwards':torch.tensor(128)}}})


class ReportContracts(unittest.TestCase):
    def test_small_bad_tensor_cannot_hide_in_global_norm(self):
        a=audit();b=copy.deepcopy(a);b['raw_gradients']['small'][0]+=.001
        self.assertFalse(first_update(a,b,P)['passed'])
    def test_counter_changes_fail_exactly(self):
        a=audit();b=copy.deepcopy(a);b['optimizer']['model_statistics']['layer']['_total_forwards']+=1
        self.assertFalse(first_update(a,b,P)['passed'])
    def test_write_gate_distinct_from_linear_gate(self):
        a=torch.tensor([1.,1.]);b=a*1.005
        m=metric(a,b,P);self.assertTrue(m['write_pass']);self.assertFalse(m['linear_pass'])
    def test_zero_and_nonfinite_cases(self):
        self.assertTrue(metric(torch.zeros(2),torch.zeros(2),P)['write_pass'])
        self.assertFalse(metric(torch.zeros(2),torch.ones(2),P)['write_pass'])
        with self.assertRaises(ValueError):metric(torch.ones(2),torch.tensor([float('nan'),1]),P)


if __name__=='__main__':unittest.main()
