"""Synthetic contracts for all-target story evaluation; no real test scores."""
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from .model import GPT, ModelConfig
from .stories import TokenCorpus
from .train import evaluate


class StoryEvaluationContracts(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        # A full window, a two-target tail, and a two-target short story.
        self.corpus = TokenCorpus(
            {"train": np.arange(41) % 10, "val": np.array([1, 2, 3, 4, 5, 6, 0, 7, 8, 0])},
            tuple(map(str, range(10))),
            dict(seq_len=4, default_stream_seed=99, eot_id=0, evaluation_policy_version=2),
            np.array([0, 4, 7]), dev_documents=({"start": 0, "end": 7}, {"start": 7, "end": 10}),
            dev_lengths=np.array([4, 2, 2]), dev_document_indices=np.array([0, 0, 1]))
        self.cfg = dict(microbatch_sequences=2, seq_len=4, precision="fp32")
        torch.manual_seed(173)
        self.model = GPT(ModelConfig(vocab_size=10, n_layer=1, n_embd=8, n_head=2, seq_len=4,
                                     track_input_stats=True, track_input_cov=True))

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_padding_is_ignored_and_all_real_targets_are_weighted(self):
        starts = self.corpus.evaluation_starts("val", 4)
        aggregate, per_window = evaluate(self.model, self.corpus, "val", starts, self.cfg, torch.device("cpu"))
        expected = []
        self.model.eval()
        with torch.no_grad():
            for start, length in zip(starts.tolist(), (4, 2, 2)):
                tokens = self.corpus.tokens("val")[start:start + length + 1]
                logits = self.model(tokens[:-1][None])
                expected.append(F.cross_entropy(logits[0], tokens[1:], reduction="sum").double())
        self.assertAlmostEqual(aggregate, float(sum(expected) / 8), places=6)
        torch.testing.assert_close(per_window, torch.stack(expected) / torch.tensor([4, 2, 2]),
                                   rtol=1e-6, atol=1e-6)
        # EOT=0 is a real target in both stories, not an implicit ignored pad ID.
        _, y = self.corpus.evaluation_batch("val", starts, 4)
        self.assertEqual(int((y != -100).sum()), 8)
        self.assertEqual(int((y == 0).sum()), 2)

    def test_evaluation_partition_and_statistics_are_unchanged(self):
        self.model.train()
        before = {name: value.clone() for name, value in self.model.named_buffers()}
        starts = self.corpus.evaluation_starts("val", 4)
        a, _ = evaluate(self.model, self.corpus, "val", starts, self.cfg, torch.device("cpu"))
        b, _ = evaluate(self.model, self.corpus, "val", starts,
                        dict(self.cfg, microbatch_sequences=1), torch.device("cpu"))
        self.assertAlmostEqual(a, b, places=6)
        self.assertTrue(self.model.training)
        for name, value in self.model.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_evaluation_batch_override_preserves_full_target_mean(self):
        starts = self.corpus.evaluation_starts("val", 4)
        expected, per_window = evaluate(self.model, self.corpus, "val", starts, self.cfg, torch.device("cpu"))
        batches = []
        hook = self.model.register_forward_pre_hook(lambda module, inputs: batches.append(inputs[0].shape[0]))
        try:
            actual, observed = evaluate(self.model, self.corpus, "val", starts,
                                       dict(self.cfg, evaluation_microbatch_sequences=1), torch.device("cpu"))
        finally:
            hook.remove()
        self.assertEqual(batches, [1, 1, 1])
        self.assertEqual(self.cfg["microbatch_sequences"], 2)
        self.assertAlmostEqual(actual, expected, places=6)
        torch.testing.assert_close(observed, per_window, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
