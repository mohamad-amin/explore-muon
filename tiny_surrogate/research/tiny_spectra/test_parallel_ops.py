import copy
import unittest
from unittest.mock import patch
import torch
from .model import StepStatLinear,GPT,ModelConfig
from .parallel_ops import StatisticsSpan,microbatch_span
from .optim import make_optimizer


class Collector(torch.nn.Module):
    def __init__(self,clock):
        super().__init__();self.layer=StepStatLinear(3,2,cov=True,decay=.83,cov_decay=.97,cov_stride=2,stats_clock=clock).double()
    def begin_step_stats(self):self.layer.begin_step()
    def finish_step_stats(self,ema=None):self.layer.finish_step(ema)
    def forward(self,x):return self.layer(x)


class ParallelContracts(unittest.TestCase):
    def test_partitions_preserve_microbatches_and_empty_ranks(self):
        for count in [1,2,3,18,128]:
            for world in [1,2,4]:
                spans=[microbatch_span(count,world,r) for r in range(world)]
                self.assertEqual([i for a,b in spans for i in range(a,b)],list(range(count)))
        chunks=list(torch.arange(69).split(4))
        self.assertEqual(len(chunks),18)
        self.assertEqual([len(x) for x in chunks],[4]*17+[1])
        self.assertEqual(microbatch_span(1,4,0),(0,0))

    def test_ordered_affine_ema_float64_oracle(self):
        for clock in ['microforward','step']:
            oracle=Collector(clock)
            with torch.no_grad():
                for name,b in oracle.layer.named_buffers():
                    if name.startswith('input_'):b.fill_(.4)
                if clock=='microforward':oracle.layer._total_forwards.fill_(7)
            for world in [1,2,4]:
                reference=copy.deepcopy(oracle);root=copy.deepcopy(oracle)
                for seqs in [9,1,69]:
                    data=torch.arange(seqs*5*3,dtype=torch.float64).reshape(seqs,5,3)/100
                    chunks=list(data.split(4));cfg=dict(stats_clock=clock,cov_ema=.91)
                    reference.begin_step_stats()
                    for chunk in chunks:reference(chunk)
                    reference.finish_step_stats(None if clock=='microforward' else .91)
                    replicas=[copy.deepcopy(root) for _ in range(world)];contexts=[]
                    for rank,model in enumerate(replicas):
                        span=microbatch_span(len(chunks),world,rank);ctx=StatisticsSpan(model,world,rank,len(chunks),span,cfg)
                        ctx.begin()
                        for chunk in chunks[span[0]:span[1]]:model(chunk)
                        contexts.append(ctx)
                    if world==1:contexts[0].finish()
                    else:
                        packets=[ctx.packet() for ctx in contexts]
                        contexts[0].install(sum(x[0] for x in packets),sum(x[1] for x in packets))
                    root=replicas[0]
                    a=dict(reference.layer.named_buffers());b=dict(root.layer.named_buffers())
                    for key in a:
                        # Pooled local accumulators are irrelevant in microforward mode.
                        if key.startswith('input_') or key in ['_step_count','_step_cov_count','_step_forwards','_total_forwards']:
                            torch.testing.assert_close(a[key],b[key],atol=1e-12,rtol=1e-12,msg=f'{clock}/{world}/{seqs}/{key}')

    def test_global_gradient_weighting_precedes_clipping(self):
        for n in [1,5,69]:
            x=torch.arange(n*3,dtype=torch.float64).reshape(n,3)/10
            w=torch.ones(3,2,dtype=torch.float64,requires_grad=True)
            reference=(x@w).square().mean();reference.backward();expected=w.grad.clone()
            chunks=list(x.split(4))
            for world in [1,2,4]:
                gradients=[]
                for rank in range(world):
                    v=w.detach().clone().requires_grad_();a,b=microbatch_span(len(chunks),world,rank)
                    for chunk in chunks[a:b]:((chunk@v).square().mean()*len(chunk)/n).backward()
                    gradients.append(torch.zeros_like(v) if v.grad is None else v.grad)
                actual=sum(gradients)
                torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
                clipped=actual/max(1.,float(actual.norm()))
                torch.testing.assert_close(clipped,expected/max(1.,float(expected.norm())),atol=1e-12,rtol=1e-12)

    def test_rank_zero_optimizer_cannot_silently_shard_state(self):
        torch.manual_seed(1)
        model=GPT(ModelConfig(vocab_size=17,n_layer=1,n_embd=8,n_head=1,seq_len=4,
            track_input_stats=True,track_input_cov=True,stats_clock='microforward'))
        cfg=dict(method='spd',lr=.01,aux_lr=.002,root_refresh=1)
        optimizer=make_optimizer(model,cfg,'cpu',distributed_optimizer=False)
        x=torch.arange(8).reshape(2,4);model.begin_step_stats();model(x,(x+1)%17).backward();model.finish_step_stats()
        with patch('research.adamw_spectra.muon.dist.is_initialized',return_value=True), \
             patch('research.adamw_spectra.muon.dist.get_world_size',side_effect=AssertionError('Hidden owner sharding')), \
             patch('research.adamw_spectra.muon.dist.all_reduce',side_effect=AssertionError('Hidden collective')):
            optimizer.step()
        self.assertEqual(len(optimizer.soap['states']),6)
        self.assertEqual(len(optimizer.data_norm['roots']),6)

    def test_original_optimizer_default_unchanged(self):
        model=GPT(ModelConfig(vocab_size=17,n_layer=1,n_embd=8,n_head=1,seq_len=4))
        optimizer=make_optimizer(model,dict(method='muon',lr=.01,aux_lr=.002),'cpu')
        self.assertTrue(optimizer.distributed_optimizer)


if __name__=='__main__':unittest.main()
