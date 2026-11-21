#!/usr/bin/env python3
"""Synthetic Bernoulli soundness check for all verifiers (no GPU).

For a grid of true p values around (theta, theta+eta), runs each verifier on
i.i.d. Bernoulli(p) streams and reports the UNSOUND-verdict rate, the None
rate, and median/mean sample counts. The criterion is the
error-side soundness definition actually claimed by every method:

    unsound Yes  <=>  answer == Yes and p >  theta + eta
    unsound No   <=>  answer == No  and p <= theta
    (inside the indifference region (theta, theta+eta] any verdict is allowed)

and additionally reports, for the forced-decision methods, how often they
answer inside the indifference region (``gray_decisions``). Writes
rq2/synthetic_check.csv.
"""
import random, statistics, csv, sys
sys.path.insert(0, ".")
from verifier import VERIFIERS


class FakeSampler:
    def __init__(self, p, seed): self.p = p; self.rng = random.Random(seed)
    def draw(self): return 1 if self.rng.random() < self.p else 0
    def sample(self, n): return sum(self.draw() for _ in range(n))


THETA, ETA, DELTA = 0.01, 0.005, 0.01
GRID = [0.0, 0.001, 0.005, 0.009, 0.0125, 0.016, 0.02, 0.05, 0.2]
RUNS = 100
rows = []
for p_true in GRID:
    gray = THETA < p_true <= THETA + ETA
    for vid in sorted(VERIFIERS):
        name, runner = VERIFIERS[vid]
        wrong = none = gray_dec = 0; counts = []
        for seed in range(RUNS):
            r = runner(FakeSampler(p_true, seed), THETA, ETA, DELTA, 1)
            counts.append(r.n_samples)
            if r.answer == "Yes" and p_true > THETA + ETA: wrong += 1
            elif r.answer == "No" and p_true <= THETA: wrong += 1
            elif r.answer not in ("Yes", "No"): none += 1
            elif gray: gray_dec += 1
        rows.append(dict(p=p_true, method=name, runs=RUNS, unsound=wrong, none=none,
                         gray_decisions=gray_dec if gray else "",
                         median_n=int(statistics.median(counts)), mean_n=int(statistics.mean(counts))))
        print(rows[-1], flush=True)
with open("rq2/synthetic_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("WROTE rq2/synthetic_check.csv")
