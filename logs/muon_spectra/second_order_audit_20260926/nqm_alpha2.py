"""NQM check (v2, divergence-safe). Per input eigendirection j: h_j = lam_j (gamma = 1); per-step gradient-noise
variance n_j/b with n_j = sigma2 * lam_j^kappa (kappa = 1: Fisher ~ GN); preconditioner p_j = lam_j^(-2 alpha)
(linear analog of PD's sandwich); EMA momentum beta; WSD schedule with cooldown fraction cd; LR tuned per alpha."""
import numpy as np, sys
np.seterr(all="ignore")
d = 768
lam = 1.0 / np.arange(1, d + 1); lam = lam / lam.mean()
h = lam
alphas = np.array([0, 1/16, 1/8, 3/16, 1/4, 5/16, 3/8, 7/16, 1/2])
etas = np.logspace(-3.5, 2.5, 49)

def run(T, b, cd, beta, prior, kappa=1.0):
    d0 = 1.0 / h if prior == "equal_loss" else np.ones(d)
    d0 = d0 / (0.5 * (h * d0).sum())
    n = lam ** kappa; n = n / n.mean()
    q = n / b
    p = lam[None, None, :] ** (-2 * alphas[:, None, None])
    p = p / np.exp(np.log(p).mean(axis=-1, keepdims=True))
    E = etas[None, :, None]
    Sdd = np.broadcast_to(d0, (len(alphas), len(etas), d)).copy(); Sdm = np.zeros_like(Sdd); Smm = np.zeros_like(Sdd)
    dead = np.zeros((len(alphas), len(etas)), bool)
    stable = T - int(cd * T)
    for t in range(T):
        s = 1.0 if t < stable else (T - t) / (T - stable)
        eta = E * s
        A11 = 1 - eta * p * (1 - beta) * h; A12 = -eta * p * beta; A21 = (1 - beta) * h; A22 = beta
        B1 = -eta * p * (1 - beta); B2 = 1 - beta
        Sdd, Sdm, Smm = (A11**2 * Sdd + 2 * A11 * A12 * Sdm + A12**2 * Smm + B1 * B1 * q,
                         A11 * A21 * Sdd + (A11 * A22 + A12 * A21) * Sdm + A12 * A22 * Smm + B1 * B2 * q,
                         A21**2 * Sdd + 2 * A21 * A22 * Sdm + A22**2 * Smm + B2 * B2 * q)
        if t % 50 == 0 or t == T - 1:
            bad = ~np.isfinite(Sdd).all(-1) | (Sdd.max(-1) > 1e12)
            dead |= bad
            Sdd[bad] = 0; Sdm[bad] = 0; Smm[bad] = 0
    L = np.where(dead, np.inf, 0.5 * (h * Sdd).sum(-1))
    best = L.min(1); k = best.argmin()
    edge = L[k].argmin() in (0, len(etas) - 1)
    return alphas[k], best, edge

def wiener(b, prior, kappa=1.0):
    d0 = 1.0 / h if prior == "equal_loss" else np.ones(d)
    d0 = d0 / (0.5 * (h * d0).sum())
    n = lam ** kappa; n = n / n.mean()
    s = h**2 * d0
    step = (1 / h) * s / (s + n / b)
    return -np.polyfit(np.log(lam), np.log(step), 1)[0] / 2

cases = [("isotropic", 1.0), ("equal_loss", 1.0), ("isotropic", 0.5), ("isotropic", 0.0)]
for prior, kappa in cases:
    for cd, beta in ((0.1, 0.0), (0.7, 0.0), (0.7, 0.95)):
        out = []
        for b in (1e1, 1e2, 1e3, 1e4):
            a, best, edge = run(T=1500, b=b, cd=cd, beta=beta, prior=prior, kappa=kappa)
            out.append(f"b=1e{int(np.log10(b))}: a*={a:.3f}{'!' if edge else ''} [L(0)/L*={best[0]/best.min():.2f}, L(1/4)/L*={best[4]/best.min():.2f}, L(1/2)/L*={best[-1]/best.min():.2f}] W={wiener(b, prior, kappa):.2f}")
        print(f"{prior:<10} noise~lam^{kappa} cd={cd} beta={beta}: " + " | ".join(out)); sys.stdout.flush()
