"""Unit test: output-gradient second moments from the ordinary backward via a gradient probe, under model.compile.

OutProbe is an identity on a Linear's output whose backward also returns e^T e (sampled positions) as the gradient
of a dummy (d_out, d_out) leaf tensor, so autograd accumulates the statistic into probe.grad with no side effects
and no extra pass. Compared with an eager reference that captures grad_output with full backward hooks.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 16


class OutProbe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, probe):
        return y.view_as(y)

    @staticmethod
    def backward(ctx, grad):
        e = grad[:, 1::STRIDE].flatten(0, 1).float()
        return grad, e.T @ e


class Linear(nn.Linear):
    def __init__(self, i, o):
        super().__init__(i, o, bias=True)
        self.probe = None

    def forward(self, x):
        y = F.linear(x, self.weight.type_as(x), self.bias.type_as(x))
        if self.probe is not None and self.training:
            y = OutProbe.apply(y, self.probe)
        return y


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc, self.proj = Linear(d, 4 * d), Linear(4 * d, d)

    def forward(self, x):
        return x + self.proj(self.fc(F.rms_norm(x, (x.size(-1),))).relu().square())


class Net(nn.Module):
    def __init__(self, v=512, d=128, n=2):
        super().__init__()
        self.embed = nn.Embedding(v, d).bfloat16()
        self.blocks = nn.ModuleList([Block(d) for _ in range(n)])
        self.head = Linear(d, v)

    def forward(self, idx, tgt):
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        logits = self.head(F.rms_norm(x, (x.size(-1),))).float()
        return F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="sum")


def main():
    torch.manual_seed(0)
    dev = "cuda"
    ref = Net().to(dev)
    test = copy.deepcopy(ref)
    lins = [m for m in test.blocks.modules() if isinstance(m, Linear)]
    for m in lins:
        m.probe = torch.zeros(m.out_features, m.out_features, device=dev, requires_grad=True)
    test.compile(dynamic=False)
    idx = torch.randint(0, 512, (4, 256), device=dev, dtype=torch.int32)
    tgt = torch.randint(0, 512, (4, 256), device=dev)
    # reference: eager, full backward hooks capture grad_output
    captured = {}
    rlins = [m for m in ref.blocks.modules() if isinstance(m, Linear)]
    hooks = [m.register_full_backward_hook(lambda mod, gi, go, k=k: captured.__setitem__(k, go[0])) for k, m in enumerate(rlins)]
    for micro in range(2):  # two micro-batches accumulate
        ref(idx, tgt).backward()
        e = [captured[k][:, 1::STRIDE].flatten(0, 1).float() for k in range(len(rlins))]
        if micro == 0:
            want = [x.T @ x for x in e]
        else:
            want = [w + x.T @ x for w, x in zip(want, e)]
        test(idx, tgt).backward()
    for h in hooks:
        h.remove()
    for k, m in enumerate(lins):
        got = m.probe.grad
        rel = float((got - want[k]).norm() / want[k].norm())
        print(f"linear {k}: probe.grad shape {tuple(got.shape)}, rel err vs eager {rel:.2e}")
    # parameter gradients unchanged by the probes
    for (n1, p1), (n2, p2) in zip(ref.named_parameters(), test.named_parameters()):
        rel = float((p1.grad.float() - p2.grad.float()).norm() / p1.grad.float().norm().clamp_min(1e-30))
        if rel > 1e-2:
            print("param grad mismatch", n1, rel)
    print("probes are not parameters:", all(m.probe is not p for m in lins for p in test.parameters()))
    print("ok")


if __name__ == "__main__":
    main()
