import json
import os
import torch
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .multi_gpu import submission_command
from .parallel_train import main,configure_execution


class LaunchContracts(unittest.TestCase):
    def test_exact_slurm_gpu_requests(self):
        for count in [1,2,4]:
            p=dict(gpus=count,gpu_type='nvidia_rtx_a4000',cpus=2*count,memory_gib=8*count,walltime='03:00:00',node=None)
            command=submission_command(Path('/study/run'),p)
            self.assertIn('--nodes=1',command);self.assertIn('--ntasks=1',command)
            self.assertIn(f'--gres=gpu:nvidia_rtx_a4000:{count}',command)
            self.assertIn('--no-requeue',command)
            self.assertFalse(any(x.startswith('--array') for x in command))

    def test_deterministic_profile_requires_pre_cuda_environment(self):
        old=torch.are_deterministic_algorithms_enabled()
        old_cudnn=torch.backends.cudnn.deterministic
        try:
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaises(ValueError):configure_execution(dict(device='cuda'))
            with patch.dict(os.environ,{'CUBLAS_WORKSPACE_CONFIG':':4096:8'}):
                self.assertEqual(configure_execution(dict(device='cuda')),'deterministic')
                self.assertTrue(torch.are_deterministic_algorithms_enabled())
                self.assertTrue(torch.backends.cudnn.deterministic)
                self.assertEqual(configure_execution(dict(device='cuda',execution_profile='native')),'native')
                self.assertFalse(torch.are_deterministic_algorithms_enabled())
        finally:
            torch.use_deterministic_algorithms(old)
            torch.backends.cudnn.deterministic=old_cudnn

    def test_existing_result_is_never_relabelled_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);status=out/'status.json';status.write_text('{"status":"complete"}')
            with patch('sys.argv',['parallel_train','--config','unused','--out',str(out),'--expected-world-size','2']):
                with self.assertRaises(FileExistsError):main()
            self.assertEqual(status.read_text(),'{"status":"complete"}')
            self.assertFalse((out/'failure_rank0.json').exists())


if __name__=='__main__':unittest.main()
