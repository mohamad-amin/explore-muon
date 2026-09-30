"""Synchronous data parallelism with one complete optimizer on rank zero.

Microbatches retain their original boundaries and global order. Per-forward
EMAs are composed as ordered affine maps; FP32 reassociation is recorded rather
than claimed bitwise identical to serial accumulation.
"""
import math
import torch
import torch.distributed as dist
from .model import StepStatLinear


def microbatch_span(count,world,rank):
    if type(count) is not int or count<1 or type(world) is not int or world<1 or not 0<=rank<world:
        raise ValueError('Positive microbatch count/world and valid rank required')
    return count*rank//world,count*(rank+1)//world


def flat_copy(tensors):
    return torch.cat([x.detach().reshape(-1) for x in tensors])


@torch.no_grad()
def unpack(flat,tensors):
    offset=0
    for t in tensors:
        t.copy_(flat[offset:offset+t.numel()].view_as(t));offset+=t.numel()
    if offset!=flat.numel():raise ValueError('Collective packet shape mismatch')


@torch.no_grad()
def reduce_gradients(model,world,rank):
    if world==1:return
    parameters=list(model.parameters())
    packet=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    dist.reduce(packet,dst=0,op=dist.ReduceOp.SUM)
    if rank==0:
        offset=0
        for p in parameters:
            p.grad=packet[offset:offset+p.numel()].view_as(p);offset+=p.numel()


@torch.no_grad()
def broadcast_parameters(model,world):
    if world==1:return
    parameters=list(model.parameters());packet=flat_copy(parameters)
    dist.broadcast(packet,src=0);unpack(packet,parameters)


class StatisticsSpan:
    """Export local sufficient statistics, then install the ordered global EMA."""
    def __init__(self,model,world,rank,total_microbatches,span,cfg):
        self.model,self.world,self.rank,self.total,self.span,self.cfg=model,world,rank,total_microbatches,span,cfg
        if span!=microbatch_span(total_microbatches,world,rank):raise ValueError('Noncontiguous or changed microbatch assignment')
        self.modules=[m for m in model.modules() if isinstance(m,StepStatLinear)]
        self.entries=[];self.counts=[];self.previous_totals=[]

    @torch.no_grad()
    def begin(self):
        self.model.begin_step_stats()
        if self.world==1:return
        for m in self.modules:
            if m.stats_clock=='microforward':
                names=[('input_mean',m.decay),('input_sq',m.decay),('input_weight',m.decay)]
                if m.cov:names += [('input_cov',m.cov_decay),('input_cov_mean',m.cov_decay),('input_cov_weight',m.cov_decay)]
                for name,decay in names:
                    tensor=getattr(m,name)
                    self.entries.append((tensor,tensor.clone() if self.rank==0 else None,decay))
                    tensor.zero_()
                self.previous_totals.append((m._total_forwards,m._total_forwards.clone()))
                m._total_forwards.zero_()
                self.counts += [m._step_forwards,m._total_forwards]
            else:
                names=['_step_mean_sum','_step_sq_sum']
                if m.cov:names+=['_step_cov_sum','_step_cov_mean_sum']
                self.entries += [(getattr(m,n),None,None) for n in names]
            self.counts.append(m._step_count)
            if m.cov:self.counts.append(m._step_cov_count)

    @torch.no_grad()
    def packet(self):
        if not self.entries:return None,None
        suffix=self.total-self.span[1]
        values=torch.cat([(t*(decay**suffix) if decay is not None else t).reshape(-1) for t,_,decay in self.entries])
        counts=flat_copy(self.counts)
        return values,counts

    @torch.no_grad()
    def install(self,values,counts):
        if self.rank!=0:raise ValueError('Only rank zero owns global statistics')
        offset=0
        for tensor,old,decay in self.entries:
            block=values[offset:offset+tensor.numel()].view_as(tensor);offset+=tensor.numel()
            if decay is None:tensor.copy_(block)
            else:tensor.copy_(old).mul_(decay**self.total).add_(block)
        if offset!=values.numel():raise ValueError('Bad statistic packet shape')
        unpack(counts,self.counts)
        for counter,previous in self.previous_totals:counter.add_(previous)
        for m in self.modules:
            if m.stats_clock=='microforward' and int(m._step_forwards)!=self.total:
                raise ValueError('Global per-forward statistic clock changed')
        self.model.finish_step_stats(ema=self.cfg['cov_ema'] if self.cfg.get('stats_clock','step')=='step' else None)

    @torch.no_grad()
    def finish(self):
        if self.world==1:
            self.model.finish_step_stats(ema=self.cfg['cov_ema'] if self.cfg.get('stats_clock','step')=='step' else None)
            return
        values,counts=self.packet()
        if values is not None:
            dist.reduce(values,dst=0,op=dist.ReduceOp.SUM)
            dist.reduce(counts,dst=0,op=dist.ReduceOp.SUM)
            if self.rank==0:self.install(values,counts)
        if self.rank!=0:
            for m in self.modules:m.collect_stats=False
