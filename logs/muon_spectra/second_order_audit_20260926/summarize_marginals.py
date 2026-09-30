"""Summarize check_tools.py output: tool checks, then the one-sided GN marginals per matrix kind.

usage: summarize_marginals.py CHECK_JSON
Ratios are curvatures relative to K-FAC along eigenvectors of C (input side) or B (output side):
  exact/kfac   exact GN marginal (per-sequence sampled-label gradients) vs tr(B) C (or tr(C) B)
  token/kfac   per-token GN (error-weighted C) vs K-FAC: the activation-error dependence
  exact/token  exact vs per-token: cross-position terms
Medians over the model's depths; "top1" is the leading eigenvector, "top16" the median of the first 16.
"""
import json
import statistics
import sys

KINDS = ("q", "k", "v", "o", "up", "down")


def med(values):
    return statistics.median(values) if values else float("nan")


def main(path):
    data = json.load(open(path))
    for run in data["runs"]:
        print(f"\n=== {run['checkpoint']} (step {run['step']}, {run.get('sequences')} sequences, "
              f"{run.get('seconds', 0):.0f} s)")
        print(f"a) per-sequence sum vs batch gradient: max rel err {run['a_per_sequence_sum']['max_rel_err']:.2e}")
        for key, r in run["b_sampled_vs_exact"].items():
            print(f"b) {key:10s} sampled/exact {r['ratio']:.3f} (MC rel. s.e. {r['mc_rel_se']:.3f})")
        for key, r in run["c_symmetries"].items():
            print(f"c) {key:34s} q/q_random {r['ratio']:.2e}")
        print(f"d) marginal trace consistency: max rel err {run['d_trace_consistency_max_rel_err']:.2e}")
        m = run["marginals"]
        print("\nInput side (eigenvectors of C); medians over depths")
        print(f"{'kind':5s} {'C top/med':>10s} {'C PR':>7s} | {'exact/kfac':>21s} | {'token/kfac':>21s} | "
              f"{'exact/token':>21s} | {'mean dir: shareC exact/kfac exact/token':>40s} | {'slope exact':>11s} {'ovl8':>5s}")
        for kind in KINDS:
            rows = [v["in"] for n, v in m.items() if n.endswith("." + kind)]
            if not rows:
                continue
            f = lambda key, i=None: med([(r[key][i] if i is not None else statistics.median(r[key][:16])) for r in rows])
            mean_dir = [r["mean_direction"] for r in rows]
            print(f"{kind:5s} {med([r['factor_top_over_median'] for r in rows]):10.0f} "
                  f"{med([r['factor_participation_ratio'] for r in rows]):7.1f} | "
                  f"top1 {f('top_ratio_exact_kfac', 0):5.2f} top16 {f('top_ratio_exact_kfac'):5.2f} | "
                  f"top1 {f('top_ratio_token_kfac', 0):5.2f} top16 {f('top_ratio_token_kfac'):5.2f} | "
                  f"top1 {f('top_ratio_exact_token', 0):5.2f} top16 {f('top_ratio_exact_token'):5.2f} | "
                  f"{med([d['share_of_C'] for d in mean_dir]):8.2f} {med([d['exact_over_kfac'] for d in mean_dir]):10.2f} "
                  f"{med([d['exact_over_token'] for d in mean_dir]):11.2f}        | "
                  f"{med([r['slope_exact_vs_factor'] for r in rows]):11.2f} "
                  f"{med([r['top8_overlap_with_factor']['exact'] for r in rows]):5.2f}")
        print("\nOutput side (eigenvectors of B); medians over depths")
        print(f"{'kind':5s} {'B top/med':>10s} {'B PR':>7s} | {'exact/kfac':>21s} | {'token/kfac':>21s} | "
              f"{'slope exact':>11s} {'ovl8 exact':>10s} {'ovl8 ef':>8s}")
        for kind in KINDS:
            rows = [v["out"] for n, v in m.items() if n.endswith("." + kind)]
            if not rows:
                continue
            f = lambda key, i=None: med([(r[key][i] if i is not None else statistics.median(r[key][:16])) for r in rows])
            print(f"{kind:5s} {med([r['factor_top_over_median'] for r in rows]):10.0f} "
                  f"{med([r['factor_participation_ratio'] for r in rows]):7.1f} | "
                  f"top1 {f('top_ratio_exact_kfac', 0):5.2f} top16 {f('top_ratio_exact_kfac'):5.2f} | "
                  f"top1 {f('top_ratio_token_kfac', 0):5.2f} top16 {f('top_ratio_token_kfac'):5.2f} | "
                  f"{med([r['slope_exact_vs_factor'] for r in rows]):11.2f} "
                  f"{med([r['top8_overlap_with_factor']['exact'] for r in rows]):10.2f} "
                  f"{med([r['top8_overlap_with_factor']['ef'] for r in rows]):8.2f}")
        print("\nGradient noise scale (sequences), medians over depths")
        print("  " + "  ".join(f"{kind}: {med([v['gradient']['noise_scale_sequences'] for n, v in m.items() if n.endswith('.' + kind)]):.0f}"
                              for kind in KINDS))


if __name__ == "__main__":
    main(sys.argv[1])
