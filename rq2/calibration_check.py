#!/usr/bin/env python3
"""E3 -- false-certification / calibration study (no GPU).

Measures how often the algorithm says Yes when p is actually large.

Unlike :mod:`rq2.synthetic_check`, which sweeps every verifier with a
coarse wrong-verdict count, this script isolates the *guarantee* and measures
it against its nominal level on Bernoulli streams whose true p is known by
construction.

The contract we certify is one-sided on each side of the indifference band:

* ``p > theta + eta``  -> answering **Yes** is a false certification; the
  guarantee caps this at ``delta``.
* ``p <= theta``       -> answering **No** is a false rejection; also capped
  at ``delta``.
* ``theta < p <= theta + eta`` -> the band. Either verdict is permitted by the
  contract, so nothing is counted as an error there; we report the verdict mix
  instead, because that is what a reader actually wants to see.

Note that ``synthetic_check.py`` scores a Yes anywhere above ``theta`` as
wrong, which is *stricter* than the contract and conflates band behaviour with
genuine unsoundness. We report both columns so the two scripts reconcile.

Clopper-Pearson upper bounds make the comparison honest: with RUNS repetitions
an observed 0 errors still only certifies an error rate below roughly
``3/RUNS``, so we say that rather than claiming zero.

Usage:
    python rq2/calibration_check.py                 # full study, RUNS=1000
    python rq2/calibration_check.py --runs 100      # quick smoke
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from verifier import VERIFIERS  # noqa: E402

THETA, ETA, DELTA = 0.01, 0.005, 0.01

# True p values spanning: well below theta, just below theta, inside the
# (theta, theta+eta] band, just above the band, and far above it.
GRID = [
    0.0, 0.002, 0.005, 0.008, 0.010,      # p <= theta          -> No is an error
    0.012, 0.015,                          # band                -> no error defined
    0.018, 0.020, 0.030, 0.050, 0.100,     # p > theta+eta       -> Yes is an error
]


class FakeSampler:
    """i.i.d. Bernoulli(p) stream standing in for the disagreement oracle."""

    def __init__(self, p, seed):
        self.p = p
        self.rng = random.Random(seed)

    def draw(self):
        return 1 if self.rng.random() < self.p else 0

    def sample(self, n):
        return sum(self.draw() for _ in range(n))


def _one_run(job):
    """Worker: one verification on a Bernoulli(p) stream. Returns (answer, n)."""
    p_true, seed, verifier_id = job
    _, runner = VERIFIERS[verifier_id]
    r = runner(FakeSampler(p_true, seed), THETA, ETA, DELTA, 1)
    return r.answer, r.n_samples


def region(p: float) -> str:
    if p <= THETA:
        return "below"          # p <= theta
    if p <= THETA + ETA:
        return "band"           # theta < p <= theta+eta
    return "above"              # p > theta+eta


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper limit of the one-sided (1-alpha) CI for a binomial rate."""
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    try:
        from scipy.stats import beta
    except ImportError:  # pragma: no cover - scipy is a listed dependency
        return float("nan")
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--verifier-id", type=int, default=0, help="0 = ProCerDCL")
    ap.add_argument(
        "--out", default=str(REPO / "rq2" / "calibration_check.csv")
    )
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 8))
    args = ap.parse_args()

    name, _ = VERIFIERS[args.verifier_id]
    print(
        f"method={name}  theta={THETA} eta={ETA} delta={DELTA} "
        f"runs={args.runs} workers={args.workers}"
    )
    print(
        f"{'p_true':>7} {'region':>6} {'Yes':>5} {'No':>5} {'None':>5} "
        f"{'err':>5} {'err_rate':>9} {'CP95_up':>8} {'med_n':>8}"
    )

    # One flat job list so every core stays busy: the cost per run varies by
    # an order of magnitude across the grid (cheapest far from the band).
    jobs = [
        (p, seed, args.verifier_id) for p in GRID for seed in range(args.runs)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(_one_run, jobs, chunksize=4))

    by_p = {p: [] for p in GRID}
    for (p, _seed, _vid), res in zip(jobs, results):
        by_p[p].append(res)

    rows = []
    for p_true in GRID:
        reg = region(p_true)
        yes = no = none = 0
        counts = []
        for answer, n_samples in by_p[p_true]:
            counts.append(n_samples)
            if answer == "Yes":
                yes += 1
            elif answer == "No":
                no += 1
            else:
                none += 1

        # Errors under the actual contract.
        if reg == "above":
            err = yes            # false certification
            err_kind = "false_Yes"
        elif reg == "below":
            err = no             # false rejection
            err_kind = "false_No"
        else:
            err = 0              # band: both verdicts permitted
            err_kind = "none_defined"

        # Stricter bookkeeping used by synthetic_check.py, for reconciliation.
        strict_err = (yes if p_true > THETA else 0) + (
            no if p_true <= THETA + ETA else 0
        )

        rate = err / args.runs
        upper = clopper_pearson_upper(err, args.runs)
        row = dict(
            method=name,
            p=p_true,
            region=reg,
            runs=args.runs,
            yes=yes,
            no=no,
            none=none,
            err_kind=err_kind,
            errors=err,
            err_rate=round(rate, 5),
            cp95_upper=round(upper, 5) if upper == upper else "",
            strict_errors=strict_err,
            delta=DELTA,
            within_guarantee=(reg == "band") or (rate <= DELTA),
            median_n=int(statistics.median(counts)),
            mean_n=int(statistics.mean(counts)),
        )
        rows.append(row)
        print(
            f"{p_true:>7.3f} {reg:>6} {yes:>5} {no:>5} {none:>5} {err:>5} "
            f"{rate:>9.4f} {row['cp95_upper']:>8} {row['median_n']:>8,}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    violations = [r for r in rows if not r["within_guarantee"]]
    worst = max((r for r in rows if r["region"] != "band"), key=lambda r: r["err_rate"])
    print(f"\nnominal delta          : {DELTA}")
    print(
        f"worst empirical error  : {worst['err_rate']:.4f} at p={worst['p']} "
        f"({worst['err_kind']}, CP95 upper {worst['cp95_upper']})"
    )
    print(f"grid points violating  : {len(violations)} / {len(rows)}")
    if violations:
        for v in violations:
            print(f"  VIOLATION p={v['p']} {v['err_kind']} rate={v['err_rate']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
