#!/usr/bin/env python3
"""Tester correctness checks (CPU only).

1. The vectorised Clopper-Pearson decision thresholds reproduce the scalar
   ``beta.ppf`` rule exactly on a dense (k, n) grid.
2. The new ``certify(tester="cp")`` returns bit-identical verdicts and sample
   counts to the earlier scalar implementation on identical Bernoulli
   streams (the old file is imported from ``--old``; skipped if unset).
3. Empirical error rates of cp / cs / forced variants on a p-grid stay below
   delta (sanity, not a proof).
"""
import argparse, importlib.util, random, sys, time
sys.path.insert(0, ".")
import numpy as np
from src.verification.ProCerDCL import demo_alg as new
from src.verification.ProCerDCL.demo_alg import clopper_pearson, cp_decision_thresholds, certify

ap = argparse.ArgumentParser()
ap.add_argument("--old", default=None,
                help="path to an earlier demo_alg.py to compare against (check 2 is skipped if unset)")
ap.add_argument("--runs", type=int, default=40)
args = ap.parse_args()

# 1. threshold equivalence -------------------------------------------------
bad = 0
for theta1, theta2, d in [(0.01, 0.015, 0.0036 / 50), (0.0, 0.01, 0.002), (0.015, 0.0227, 1e-4),
                          (0.1, 0.2, 0.01), (0.3, 0.5, 0.05)]:
    ns = np.arange(1, 400)
    ky, kn = cp_decision_thresholds(ns, theta1, theta2, d)
    for n, a, b in zip(ns, ky, kn):
        for k in range(0, int(n) + 1):
            L, U = clopper_pearson(k, int(n), d)
            yes_ref, no_ref = (U <= theta1), (L > theta2)
            yes_new, no_new = (k <= a), (k >= b)
            if yes_ref != yes_new or no_ref != no_new:
                bad += 1
                if bad < 10:
                    print("MISMATCH", theta1, theta2, d, n, k, (L, U), (a, b), yes_ref, yes_new, no_ref, no_new)
print(f"[1] threshold equivalence: mismatches = {bad}")

class Stream:
    def __init__(self, p, seed): self.rng = random.Random(seed); self.p = p
    def draw(self): return 1 if self.rng.random() < self.p else 0


# 2. stream equivalence vs old implementation --------------------------------
if args.old:
    spec = importlib.util.spec_from_file_location("old_alg", args.old)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)

    diff = 0; tot = 0; t_old = t_new = 0.0
    for p in [0.0, 0.002, 0.005, 0.009, 0.0125, 0.016, 0.02, 0.05, 0.2]:
        for seed in range(args.runs):
            t0 = time.perf_counter(); ro = old.certify(0.01, 0.005, 0.01, 1, Stream(p, seed).draw); t_old += time.perf_counter() - t0
            t0 = time.perf_counter(); rn = certify(0.01, 0.005, 0.01, 1, Stream(p, seed).draw); t_new += time.perf_counter() - t0
            tot += 1
            if (ro.answer, ro.n_samples) != (rn.answer, rn.n_samples):
                diff += 1
                if diff < 10: print("DIFF", p, seed, (ro.answer, ro.n_samples), (rn.answer, rn.n_samples))
    print(f"[2] stream equivalence: {tot - diff}/{tot} identical; CPU time old={t_old:.1f}s new={t_new:.1f}s")
else:
    print("[2] stream equivalence: skipped (no --old)")

# 3. error-rate sanity for cp / cs / forced ----------------------------------
TH, ET, DE = 0.01, 0.005, 0.01
for cfg in [dict(tester="cp"), dict(tester="cs"), dict(tester="cp", forced=True), dict(tester="cs", forced=True)]:
    line = []
    for p in [0.002, 0.005, 0.009, 0.01, 0.0125, 0.015, 0.016, 0.02, 0.05]:
        wrong = none = 0; ns = []
        for seed in range(args.runs):
            r = certify(TH, ET, DE, 1, Stream(p, 1000 + seed).draw, **cfg)
            ns.append(r.n_samples)
            # error-side soundness: wrong Yes only if p > theta+eta, wrong No only if p <= theta
            if r.answer == "Yes" and p > TH + ET: wrong += 1
            elif r.answer == "No" and p <= TH: wrong += 1
            elif r.answer == "None": none += 1
        line.append(f"p={p}:w{wrong}/n{none}/med{int(np.median(ns))}")
    print(f"[3] {cfg}: " + "  ".join(line))
