"""Exact finite enumeration for the peer review; no model or training imports."""
import itertools
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(260928)
p = np.array([[0.15, 0.25, 0.60], [0.70, 0.20, 0.10]])
# A coupled Jacobian: output-token rows by two hidden positions of width two.
j = rng.normal(size=(6, 4))
w = np.array([[1.2, -0.3], [-0.3, -0.4]])
q = np.kron(np.eye(2), w) / 2
h = np.zeros((6, 6))
l = np.zeros_like(h)
for t in range(2):
    sl = slice(3*t, 3*t+3)
    h[sl, sl] = np.diag(p[t]) - np.outer(p[t], p[t])
    l[sl, sl] = np.diag(np.sqrt(p[t])) - np.outer(p[t], np.sqrt(p[t]))
k = j.T @ h @ j
tmap = j.T @ l
a = tmap.T @ q @ tmap
gaussian_variance = 2 * np.trace(q @ k @ q @ k)
rademacher_variance = 2 * np.square(a).sum() - 2 * np.square(a.diagonal()).sum()
expected_b = (k[:2, :2] + k[2:, 2:]) / 2


def enumeration(draws):
    mean_b = np.zeros((2, 2))
    first = second = 0.
    for probability, v in draws:
        error = j.T @ v
        rows = error.reshape(2, 2)
        mean_b += probability * (rows.T @ rows / 2)
        scalar = float(error @ q @ error)
        first += probability * scalar
        second += probability * scalar**2
    return mean_b, second - first**2


cat_draws = []
for labels in itertools.product(range(3), repeat=2):
    v = p.copy()
    probability = 1.
    for t, y in enumerate(labels):
        v[t, y] -= 1.
        probability *= p[t, y]
    cat_draws.append((probability, v.flatten()))
cat_b, cat_var = enumeration(cat_draws)
rad_b, rad_var = enumeration(
    (1/64, l @ np.array(signs))
    for signs in itertools.product([-1., 1.], repeat=6)
)
cat_cumulant = 0.
for t in range(2):
    sl = slice(3*t, 3*t+3)
    jt = j[sl, :]
    kt = jt.T @ h[sl, sl] @ jt
    fourth = 0.
    for y in range(3):
        score = p[t].copy()
        score[y] -= 1.
        contribution = jt.T @ score
        fourth += p[t, y] * float(contribution @ q @ contribution)**2
    cat_cumulant += fourth - np.trace(q @ kt)**2 - 2*np.trace(q @ kt @ q @ kt)

errors = {
    'half_factor_max_abs': float(np.max(np.abs(l @ l.T - h))),
    'categorical_expected_B_max_abs': float(np.max(np.abs(cat_b - expected_b))),
    'rademacher_expected_B_max_abs': float(np.max(np.abs(rad_b - expected_b))),
    'categorical_variance_max_abs': float(abs(cat_var - gaussian_variance - cat_cumulant)),
    'rademacher_variance_max_abs': float(abs(rad_var - rademacher_variance)),
}
assert max(errors.values()) < 1e-11, errors
assert rad_var <= gaussian_variance + 1e-11
binary = []
for probability in [0.01, 0.10, 0.25, 0.50, 0.90, 0.99]:
    scale = probability * (1-probability)
    binary.append({'p': probability,
                   'categorical_squared_score_variance': scale*(1-2*probability)**2,
                   'gaussian_squared_score_variance': 2*scale**2})
result = {
    'status': 'passed', 'scope': 'exact finite algebra checks only; no trained model',
    'seed': 260928, 'errors': errors,
    'scalar_quadratic_variance': {'categorical': float(cat_var),
                                'gaussian': float(gaussian_variance),
                                'rademacher': float(rad_var)},
    'binary_counterexamples': binary,
}
Path(__file__).with_name('algebra_checks.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
