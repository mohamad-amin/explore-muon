"""Small algebra checks only; no torch, model, datasets, or accelerators."""
import json
import math
from pathlib import Path
import random


def add(*vectors):
    return [sum(entries) for entries in zip(*vectors)]


def ce(logits, label):
    maximum = max(logits)
    return maximum + math.log(sum(math.exp(x-maximum) for x in logits)) - logits[label]


def matvec(matrix, vector):
    return [sum(x*y for x, y in zip(row, vector)) for row in matrix]


rng = random.Random(20260928)
errors = []
label_spreads = []
bilinear_errors = []
for _ in range(100):
    z, a, b, c = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(4)]
    overlaps = []
    for label in range(5):
        total = ce(add(z,a,b,c), label)-ce(add(z,a), label)-ce(add(z,b), label)+ce(z,label)
        overlap = ce(add(z,a,b),label)-ce(add(z,a),label)-ce(add(z,b),label)+ce(z,label)
        transport = ce(add(z,a,b,c),label)-ce(add(z,a,b),label)
        errors.append(abs(total-overlap-transport))
        overlaps.append(overlap)
    label_spreads.append(max(overlaps)-min(overlaps))
    W, dW = [[[rng.uniform(-1,1) for _ in range(3)] for _ in range(5)] for _ in range(2)]
    h, dh = [[rng.uniform(-1,1) for _ in range(3)] for _ in range(2)]
    Wnext = [add(row, drow) for row, drow in zip(W,dW)]
    z00, z10, z01, z11 = matvec(W,h), matvec(W,add(h,dh)), matvec(Wnext,h), matvec(Wnext,add(h,dh))
    mixed = [v11-v10-v01+v00 for v11,v10,v01,v00 in zip(z11,z10,z01,z00)]
    predicted = matvec(dW,dh)
    bilinear_errors.extend(abs(x-y) for x,y in zip(mixed,predicted))

quadratic = lambda body, aux: .5*(body+aux-1)**2
baseline = quadratic(0,0)
toy = {
    'body_change': quadratic(.9,0)-baseline,
    'aux_change': quadratic(0,.4)-baseline,
    'full_change': quadratic(.9,.4)-baseline,
    'interaction': quadratic(.9,.4)-quadratic(.9,0)-quadratic(0,.4)+baseline,
    'full_minus_body': quadratic(.9,.4)-quadratic(.9,0),
}
report = {
    'trials':100,
    'max_loss_split_error':max(errors),
    'max_overlap_label_spread':max(label_spreads),
    'max_head_bilinear_error':max(bilinear_errors),
    'quadratic_counterexample':toy,
    'scope':'Synthetic algebra only; no empirical mechanism claim.',
}
assert max(max(errors),max(label_spreads),max(bilinear_errors)) < 1e-13
assert toy['aux_change'] < 0 and toy['full_minus_body'] > 0
Path(__file__).with_name('identity_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
