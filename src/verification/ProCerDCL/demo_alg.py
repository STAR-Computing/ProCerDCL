"""ProCerDCL: adaptive probabilistic certification with sample reuse.

Direct, function-for-function implementation of the five algorithms in
``Algorithm.pdf``. Decides whether the unknown success probability ``p`` of a
binary sampler satisfies ``p <= theta`` (Yes) or ``p > theta + eta`` (No) with
global error budget ``delta``.

    Algorithm 1  create_interval     -- one proving/refuting step
    Algorithm 2  allocate_deltas     -- interval list I and budgets {delta_i}
    Algorithm 3  certify             -- main entry point
    Algorithm 4  clopper_pearson     -- exact binomial confidence interval
    Algorithm 5  sequential_ci_tester-- inner tester with shared sample pool

Tester options (all opt-in through ``certify(..., tester=, forced=)``):

* ``tester="cp"`` (default) is the original sequential Clopper-Pearson tester,
  now evaluated through vectorised integer decision thresholds (one batch of
  ``scipy.stats.binom`` calls per tester instead of one ``beta.ppf`` per
  sample) and running success counts (O(N) instead of O(N^2)). Verdicts are
  bit-identical to the scalar implementation.
* ``tester="cs"`` replaces the Bonferroni-over-checkpoints split by an
  anytime-valid confidence sequence: the Beta-mixture test martingale
  (Ville's inequality). Both wrong-Yes and wrong-No are contained in the single
  event ``sup_t M_t(p) >= 1/delta_i``, so one budget ``delta_i`` covers the
  tester with no per-checkpoint deflation.
* ``forced=True`` removes ``None``: the sequential part runs at ``delta_i/2``
  and, if it exhausts its truncation budget, a PROVERO-style fixed-sample
  midpoint decision at ``delta_i/2`` is taken on the same prefix of the pool.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import betaln
from scipy.stats import beta, binom


# ---------------------------------------------------------------------------
# Result labels, types, and the demo random source.
# ---------------------------------------------------------------------------
YES, NO, UNKNOWN = "Yes", "No", "None"
Interval = Tuple[float, float]

TRUE_P = 0.50   # the "unknown" p the demo sampler simulates
SEED = 0
EPS = 1e-12     # float tolerance for the strict ">" tests in Alg. 2
_rng = random.Random(SEED)

# Beta(a, b) mixing prior of the confidence-sequence tester (Jeffreys).
CS_PRIOR_A = 0.5
CS_PRIOR_B = 0.5


def draw_one() -> int:
    """One Bernoulli(TRUE_P) draw. Replace with a real property sampler."""
    return 1 if _rng.random() < TRUE_P else 0


# ---------------------------------------------------------------------------
# Algorithm 1: CreateInterval
# ---------------------------------------------------------------------------
def create_interval(
    theta: float, prev: Interval, eta: float, *, left: bool
) -> Interval:
    """One proving (``left=True``) or refuting (``left=False``) step."""
    theta1, theta2 = prev

    if theta == 0 and left:
        return theta, theta + eta
    if theta1 == 0 and theta2 == 0 and left:
        return 0.0, theta
    if theta1 == 0 and theta2 == 0 and not left:
        return theta + eta, 1.0

    alpha = theta2 - theta1
    step = max(eta, alpha / 2)
    if left:
        lo, hi = theta2 - step, theta2
    else:
        lo, hi = theta1, theta1 + step
    return max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))


# ---------------------------------------------------------------------------
# Algorithm 2: AllocateDeltasByIntervalLength
# ---------------------------------------------------------------------------
def allocate_deltas(
    theta: float, eta: float, delta: float
) -> Tuple[List[Tuple[Interval, str]], List[float]]:
    """Enumerate all proving + refuting intervals (then the final fallback
    ``(theta, theta + eta)``) and split ``delta`` proportional to ``1/length``.
    """
    intervals: List[Tuple[Interval, str]] = []
    left, right = (0.0, 0.0), (0.0, 0.0)

    while True:
        left = create_interval(theta, left, eta, left=True)
        if left[1] - left[0] > eta + EPS:
            intervals.append((left, "left"))

        right = create_interval(theta, right, eta, left=False)
        if right[1] - right[0] > eta + EPS:
            intervals.append((right, "right"))

        if (left[1] - left[0] <= eta + EPS
                and right[1] - right[0] <= eta + EPS):
            break

    intervals.append(((theta, theta + eta), "final"))

    weights = [1.0 / (iv[1] - iv[0]) for iv, _ in intervals]
    total = sum(weights)
    deltas = [delta * w / total for w in weights]
    return intervals, deltas


# ---------------------------------------------------------------------------
# Algorithm 4: ClopperPearson  (declared before Alg. 5 because Alg. 5 calls it)
# ---------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, delta: float) -> Interval:
    """Exact two-sided ``1 - delta`` confidence interval for a Binomial p."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("require 0 <= k <= n")
    L = 0.0 if k == 0 else float(beta.ppf(delta / 2, k, n - k + 1))
    U = 1.0 if k == n else float(beta.ppf(1 - delta / 2, k + 1, n - k))
    return L, U


def tester_budget(theta1: float, theta2: float, delta_i: float) -> int:
    """Truncation budget N_i (PROVERO Algorithm 2, line 1)."""
    return math.ceil(
        (math.sqrt(3 * theta1) + math.sqrt(2 * theta2)) ** 2
        / (theta2 - theta1) ** 2
        * math.log(1 / delta_i)
    )


def _checkpoints(Ni: int, batch_size: int, geometric: float = 0.0) -> List[int]:
    if geometric > 1.0:
        # group-sequential looks at n_j = ceil(GEOM_N0 * geometric**j)
        cps, n = [], float(GEOM_N0)
        while math.ceil(n) < Ni:
            c = math.ceil(n)
            if not cps or c > cps[-1]:
                cps.append(c)
            n *= geometric
        cps.append(Ni)
        return cps
    cps = list(range(batch_size, Ni + 1, batch_size))
    if not cps or cps[-1] != Ni:
        cps.append(Ni)
    return cps


def cp_decision_thresholds(
    ns: Sequence[int], theta1: float, theta2: float, delta_per_check: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Integer thresholds equivalent to the Clopper-Pearson rule at level
    ``delta_per_check`` for every checkpoint ``n`` in ``ns``:

        U(k, n) <= theta1   <=>   k <= k_yes[n]
        L(k, n) >  theta2   <=>   k >= k_no[n]

    using the binomial-tail characterisation of the CP bounds
    (``U <= theta1`` iff ``P[Bin(n, theta1) <= k] <= delta/2``;
    ``L > theta2`` iff ``P[Bin(n, theta2) >= k] < delta/2``). ``k_yes = -1``
    means Yes is impossible at that n; ``k_no = n + 1`` means No is impossible.
    """
    ns = np.asarray(ns, dtype=np.int64)
    half = delta_per_check / 2.0
    # --- Yes side -----------------------------------------------------------
    if theta1 <= 0.0:
        k_yes = np.full(ns.shape, -1, dtype=np.int64)
    else:
        # smallest k with cdf >= half, then step back while cdf > half
        k = binom.ppf(half, ns, theta1).astype(np.int64)
        k = np.clip(k, 0, ns)
        over = binom.cdf(k, ns, theta1) > half
        k_yes = np.where(over, k - 1, k)
        # ppf may under-shoot by one in rare float cases: move up if allowed
        up_ok = (k_yes + 1 <= ns) & (binom.cdf(k_yes + 1, ns, theta1) <= half)
        k_yes = np.where(up_ok, k_yes + 1, k_yes)
    # --- No side ------------------------------------------------------------
    if theta2 >= 1.0:
        k_no = ns + 1
    else:
        # smallest k' with sf(k') <= half  ->  k = k' + 1 satisfies
        # sf(k-1) <= half; then enforce the strict "<" of L > theta2.
        kp = binom.isf(half, ns, theta2).astype(np.int64)
        k_no = np.clip(kp + 1, 1, ns + 1)
        strict_fail = (k_no <= ns) & ~(binom.sf(k_no - 1, ns, theta2) < half)
        k_no = np.where(strict_fail, k_no + 1, k_no)
        down_ok = (k_no - 1 >= 1) & (binom.sf(k_no - 2, ns, theta2) < half)
        k_no = np.where(down_ok, k_no - 1, k_no)
    return k_yes, k_no


def _log_mixture_martingale(k: int, t: int, p0: float) -> float:
    """log M_t(p0) of the Beta(a,b)-mixture test martingale for Bernoulli
    observations (k successes in t trials) against the point null p0."""
    a, b = CS_PRIOR_A, CS_PRIOR_B
    num = betaln(a + k, b + t - k) - betaln(a, b)
    if p0 <= 0.0:
        den = 0.0 if k == 0 else -math.inf
    elif p0 >= 1.0:
        den = 0.0 if k == t else -math.inf
    else:
        den = k * math.log(p0) + (t - k) * math.log1p(-p0)
    return float(num - den)


# ---------------------------------------------------------------------------
# Algorithm 5: SequentialBinomialCITesterWithReuse (+ CS / forced variants)
# ---------------------------------------------------------------------------
def _grow_pool(pool: List[int], n: int, draw: Callable[[], int]) -> None:
    while len(pool) < n:
        pool.append(draw())


def sequential_ci_tester(
    theta1: float,
    theta2: float,
    delta_i: float,
    batch_size: int,
    pool: List[int],
    draw: Callable[[], int],
    *,
    tester: str = "cp",
    forced: bool = False,
    stats: Optional[dict] = None,
    geometric: float = 0.0,
) -> str:
    """Grow the shared ``pool`` by a fixed batch size, evaluate the stopping
    rule after each batch, and early-stop as soon as ``theta1`` is separated
    from ``theta2``. Returns ``UNKNOWN`` if the budget runs out (unless
    ``forced``). ``tester`` selects the rule: ``"cp"`` = Bonferroni-split
    Clopper-Pearson (original), ``"cs"`` = Beta-mixture confidence sequence.
    """
    if theta1 >= theta2:
        raise ValueError("require theta1 < theta2")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if tester not in ("cp", "cs"):
        raise ValueError("tester must be 'cp' or 'cs'")

    d_seq = delta_i / 2.0 if forced else delta_i
    Ni = tester_budget(theta1, theta2, d_seq)
    checkpoints = _checkpoints(Ni, batch_size, geometric)

    # CP splits the error budget across checkpoints. One CS threshold covers them all.
    if tester == "cp":
        k_yes, k_no = cp_decision_thresholds(
            checkpoints, theta1, theta2, d_seq / len(checkpoints))
    else:
        log_thr = math.log(1.0 / d_seq)

    k = 0
    prev = 0
    verdict = UNKNOWN
    for j, nt in enumerate(checkpoints):
        _grow_pool(pool, nt, draw)
        k += sum(pool[prev:nt])
        prev = nt
        if tester == "cp":
            if k <= k_yes[j]:
                verdict = YES
                break
            if k >= k_no[j]:
                verdict = NO
                break
        else:
            phat = k / nt
            if phat <= theta1 and _log_mixture_martingale(k, nt, theta1) >= log_thr:
                verdict = YES
                break
            if phat > theta2 and _log_mixture_martingale(k, nt, theta2) >= log_thr:
                verdict = NO
                break

    if verdict == UNKNOWN and forced:
        # PROVERO fixed-sample decision on the first Ni draws at level d_seq.
        gap = theta2 - theta1
        eta1 = 0.0 if theta1 <= 0.0 else gap / (1.0 + math.sqrt(2.0 * theta2 / (3.0 * theta1)))
        verdict = YES if (k / Ni) <= theta1 + eta1 else NO
        if stats is not None:
            stats["forced"] = True

    if stats is not None:
        stats.update(k=int(k), n=int(prev), budget=int(Ni))
    return verdict


# ---------------------------------------------------------------------------
# Algorithm 3: AdaptiveBinPCertifyWithReuse  (public entry point)
# ---------------------------------------------------------------------------
@dataclass
class CertifyResult:
    answer: str                                       # YES / NO / UNKNOWN
    n_samples: int                                    # total draws used
    intervals: List[Tuple[Interval, str]] = field(default_factory=list)
    deltas: List[float] = field(default_factory=list)
    last_stats: dict = field(default_factory=dict)    # (k, n, budget) of the deciding tester


def certify(
    theta: float,
    eta: float,
    delta: float,
    batch_size: int = 1,
    draw: Callable[[], int] = draw_one,
    *,
    tester: str = "cp",
    forced: bool = False,
    reuse: bool = True,
    uniform_delta: bool = False,
    geometric: float = 0.0,
) -> CertifyResult:
    """Sound (1 - delta)-confidence decision between ``p <= theta`` and
    ``p > theta + eta``. Walks the planned intervals in order; every
    tester call reuses the same sample pool. ``batch_size`` is a fixed
    checking interval with default value 1.

    ``reuse=False`` / ``uniform_delta=True`` are the ablation switches
    (Insight 3 / Insight 2 off); soundness is unaffected.
    """
    if not 0 <= theta <= 1:
        raise ValueError("theta must be in [0, 1]")
    if not 0 < eta <= 1:
        raise ValueError("eta must be in (0, 1]")
    if theta + eta > 1:
        raise ValueError("theta + eta must be <= 1")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    intervals, deltas = allocate_deltas(theta, eta, delta)
    if uniform_delta:
        deltas = [delta / len(intervals)] * len(intervals)
    pool: List[int] = []
    total = 0

    for ((theta1, theta2), side), delta_i in zip(intervals, deltas):
        if not reuse:
            pool = []
        stats: dict = {}
        ans = sequential_ci_tester(
            theta1, theta2, delta_i, batch_size, pool, draw,
            tester=tester, forced=forced, stats=stats, geometric=geometric,
        )
        # The shared pool counts each oracle draw once across all intervals.
        total = total + len(pool) if not reuse else len(pool)
        # Do not use the boundary interval (0, theta) as a Yes witness:
        # it would require proving p <= 0 instead of p being small.
        if side == "left" and theta1 > 0.0 and ans == YES:
            return CertifyResult(YES, total, intervals, deltas, stats)
        if side == "right" and ans == NO:
            return CertifyResult(NO, total, intervals, deltas, stats)
        if side == "final":
            return CertifyResult(ans, total, intervals, deltas, stats)

    return CertifyResult(UNKNOWN, total, intervals, deltas, {})


GEOM_N0 = 50


def certify_single(
    theta: float,
    eta: float,
    delta: float,
    batch_size: int = 1,
    draw: Callable[[], int] = draw_one,
    *,
    tester: str = "cp",
    geometric: float = 0.0,
) -> CertifyResult:
    """Main algorithm: one sequential test on the indifference band (theta,
    theta + eta) at the full delta, with no interval family, no reuse and no
    allocation. Same Yes/No/None contract as :func:`certify`, which is the
    interval-family (PROVERO-compatible) mode."""
    pool: List[int] = []
    stats: dict = {}
    ans = sequential_ci_tester(theta, theta + eta, delta, batch_size, pool, draw,
                               tester=tester, stats=stats, geometric=geometric)
    return CertifyResult(ans, len(pool), [((theta, theta + eta), "final")], [delta], stats)


def certify_variant(
    theta: float,
    eta: float,
    delta: float,
    batch_size: int = 1,
    draw: Callable[[], int] = draw_one,
    reuse: bool = True,
    uniform_delta: bool = False,
    **kw,
) -> CertifyResult:
    """Ablation harness kept for backward compatibility; see :func:`certify`."""
    return certify(theta, eta, delta, batch_size, draw,
                   reuse=reuse, uniform_delta=uniform_delta, **kw)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    theta, eta, delta = 0.1, 0.01, 0.01
    r = certify(theta, eta, delta)

    print(f"theta       = {theta}")
    print(f"eta         = {eta}")
    print(f"delta       = {delta}")
    print("batch_size  = 1")
    print(f"true p      = {TRUE_P}")
    print(f"seed        = {SEED}")
    print(f"intervals   = {len(r.intervals)}")
    print(f"sum(deltas) = {sum(r.deltas):.12f}")
    print(f"answer      = {r.answer}")
    print(f"samples     = {r.n_samples}")
