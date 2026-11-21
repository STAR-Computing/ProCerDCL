"""Search sampled clean-label radii for a reference and a compiled classifier.

The routine brackets and bisects radii under the assumption that the violation
probability is nondecreasing with radius. Its result records each statistical
verdict and the largest radius tested with a Yes verdict.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional

from src.semantic_checker import predict_label
from src.verification.PROVERO import YES, NO, UNKNOWN
from src.verification.PROVERO.samplers import AdversarialRobustnessSampler
from verifier import run_verifier

# Default search cap, as a multiple of ||x0||_2, when max_epsilon is omitted.
MAX_EPSILON_NORM_FACTOR = 4.0
VERIFIER_NAMES = {
    0: "ProCerDCL",
    1: "PROVERO",
    2: "DirectCertify",
    3: "SPRT",
    4: "ProCerDCL-NR",
    5: "ProCerDCL-UD",
    6: "ProCerDCL-CS",
    7: "ProCerDCL-F",
    8: "ProCerDCL-CS-F",
    9: "Single-CP",
    10: "Single-CS",
    11: "Single-CP-G",
    12: "ProCerDCL-G",
}


@dataclass
class EpsilonSearchResult:
    method: str
    theta: float
    eta: float
    start_epsilon: float
    epsilon_step: float
    best_epsilon: Optional[float]
    stop_epsilon: Optional[float]
    stop_answer: Optional[str]
    n_probes: int
    n_samples: int
    probe_log: Optional[List[dict]] = None
    elapsed_s: Optional[float] = None

    @property
    def found(self) -> bool:
        return self.best_epsilon is not None


def choose_answer(answers: List[str]) -> str:
    """Choose the majority answer for repeated probes at one epsilon."""
    yes = answers.count(YES)
    no = answers.count(NO)
    if yes > no:
        return YES
    if no > yes:
        return NO
    return UNKNOWN




# A positive patch size restricts the sampling support to the top-left square.
PATCH = 0


def _patch_mask(x0):
    if PATCH <= 0:
        return None
    import torch as _t
    m = _t.zeros_like(x0.squeeze(0))
    m[..., :PATCH, :PATCH] = 1.0
    return m


def _make_sampler(cl_model, x0, y0, *, epsilon, device, seed, sampler_batch_size, norm="2"):
    """Build the paper's adversarial-density sampler at radius ``epsilon``.

    The property (PROVERO Section II-A) is ``psi = 1`` iff a perturbation drawn
    uniformly from the L2 ``epsilon``-ball changes the *compiled* model's
    prediction away from the original model's clean label ``y0``. Its success
    probability is the adversarial density that the verifier certifies against
    ``theta``. Forward passes are batched (``sampler_batch_size``) so the large
    sample counts the verifier requests stay tractable.
    """
    return AdversarialRobustnessSampler(
        cl_model,
        x0.squeeze(0),  # collate yields (1, ...); sampler wants no batch dim
        y=y0,
        epsilon=epsilon,
        norm=norm,
        batch_size=sampler_batch_size,
        device=device,
        seed=seed,
        mask=_patch_mask(x0),
    )

def find_max_yes_epsilon(
    *,
    verifier_id: int,
    model,
    cl_model,
    x0,
    tokenizer,
    device,
    theta: float,
    eta: float,
    delta: float,
    batch_size: int,
    start_epsilon: float,
    epsilon_step: float,
    trials_per_epsilon: int,
    seed: int,
    max_epsilon: Optional[float] = None,
    sampler_batch_size: int = 256,
    norm: str = "2",
    probe_delta_mode: str = "fixed",
) -> EpsilonSearchResult:
    """Return the largest tested radius with a Yes verdict, if one was found.

    The search assumes that the violation rate is nondecreasing with radius.
    Under this assumption it searches for the boundary in two phases:

    1. **Exponential bracketing** -- probe ``start_epsilon`` (which must be Yes),
       then grow the offset geometrically (x2) until a probe is no longer Yes or
       ``max_epsilon`` is reached, giving a ``[yes, not-yes]`` bracket.
    2. **Binary search** -- bisect the bracket until its width is below
       ``epsilon_step`` (the resolution), keeping the largest Yes epsilon.

    Each probe runs the verifier ``trials_per_epsilon`` times and takes the
    majority answer (see :func:`choose_answer`). ``max_epsilon`` defaults to
    ``MAX_EPSILON_NORM_FACTOR * ||x0||_2`` (for an L2 ball). For the infinity norm,
    the cap defaults to the maximum absolute coordinate plus one.
    """
    if verifier_id not in VERIFIER_NAMES:
        raise ValueError(
            f"unknown verifier-id {verifier_id}; "
            f"choose one of {sorted(VERIFIER_NAMES)}"
        )
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must be in [0, 1]")
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must be in (0, 1)")
    if theta + eta > 1.0:
        raise ValueError("theta + eta must be <= 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if start_epsilon < 0.0:
        raise ValueError("start_epsilon must be non-negative")
    if epsilon_step <= 0.0:
        raise ValueError("epsilon_step must be positive")
    if trials_per_epsilon < 1:
        raise ValueError("trials_per_epsilon must be >= 1")

    method_name = VERIFIER_NAMES[verifier_id]
    if max_epsilon is None:
        # The L2 noise must grow somewhat past the input's own norm before it
        # dominates and flips the label, so cap a few multiples above ||x0||.
        if norm == "inf":
            max_epsilon = float(x0.abs().max().item()) + 1.0
        else:
            max_epsilon = MAX_EPSILON_NORM_FACTOR * float(x0.flatten().norm(p=2).item())

    # Reference label: the original model's clean prediction. The sampler then
    # measures how often the compiled model's perturbed prediction deviates.
    y0 = int(predict_label(model, x0).reshape(-1)[0].item())

    # Per-probe error budget: "fixed" spends the full delta on every probe
    # (the historical protocol; the composed search then carries a union-bound
    # confidence of 1 - K*delta); "split" divides delta by a precomputed upper
    # bound K_max on the number of probes so the *reported radius* carries the
    # full 1 - delta guarantee.
    if probe_delta_mode == "split":
        k_max = 2 + 2 * max(1, math.ceil(math.log2(max(max_epsilon, epsilon_step)
                                                   / epsilon_step)))
        probe_delta = delta / k_max
    else:
        k_max = None
        probe_delta = delta

    t_start = time.perf_counter()
    probe_log: List[dict] = []
    n_probes = 0
    n_samples = 0

    def _finish(res: "EpsilonSearchResult") -> "EpsilonSearchResult":
        res.probe_log = probe_log
        res.elapsed_s = round(time.perf_counter() - t_start, 3)
        return res

    def probe(epsilon: float) -> str:
        """Majority verifier answer at ``epsilon`` over ``trials_per_epsilon``."""
        nonlocal n_probes, n_samples
        answers: List[str] = []
        for trial in range(trials_per_epsilon):
            sampler = _make_sampler(
                cl_model, x0, y0,
                epsilon=epsilon, device=device, seed=seed + trial,
                sampler_batch_size=sampler_batch_size, norm=norm,
            )
            result = run_verifier(
                verifier_id, sampler, theta, eta, probe_delta, batch_size
            )
            answers.append(result.answer)
            n_probes += 1
            n_samples += result.n_samples
            probe_log.append({
                "epsilon": round(float(epsilon), 6),
                "answer": result.answer,
                "n_samples": int(result.n_samples),
                "probe_delta": probe_delta,
            })
        return choose_answer(answers)

    # Phase 0: the lower bound must itself be certified Yes, else there is no
    # Yes region to bound.
    ans = probe(start_epsilon)
    if ans != YES:
        return _finish(EpsilonSearchResult(
            method=method_name, theta=theta, eta=eta,
            start_epsilon=start_epsilon, epsilon_step=epsilon_step,
            best_epsilon=None, stop_epsilon=start_epsilon, stop_answer=ans,
            n_probes=n_probes, n_samples=n_samples,
        ))
    best_epsilon = start_epsilon

    # Phase 1: grow the offset geometrically until a probe is no longer Yes.
    offset = epsilon_step
    hi = start_epsilon + offset
    while hi < max_epsilon:
        ans = probe(hi)
        if ans != YES:
            break
        best_epsilon = hi
        offset *= 2.0
        hi = start_epsilon + offset
    else:
        # Never left the Yes region below the cap: clamp and test the cap once.
        hi = max_epsilon
        ans = probe(hi)
        if ans == YES:
            return _finish(EpsilonSearchResult(
                method=method_name, theta=theta, eta=eta,
                start_epsilon=start_epsilon, epsilon_step=epsilon_step,
                best_epsilon=hi, stop_epsilon=hi, stop_answer=ans,
                n_probes=n_probes, n_samples=n_samples,
            ))

    # Phase 2: bisect the [best_epsilon (Yes), hi (not Yes)] bracket.
    yes_eps, no_eps = best_epsilon, hi
    stop_epsilon, stop_answer = no_eps, ans
    while no_eps - yes_eps > epsilon_step:
        mid = 0.5 * (yes_eps + no_eps)
        ans = probe(mid)
        if ans == YES:
            yes_eps = best_epsilon = mid
        else:
            no_eps = mid
            stop_epsilon, stop_answer = no_eps, ans

    return _finish(EpsilonSearchResult(
        method=method_name,
        theta=theta,
        eta=eta,
        start_epsilon=start_epsilon,
        epsilon_step=epsilon_step,
        best_epsilon=best_epsilon,
        stop_epsilon=stop_epsilon,
        stop_answer=stop_answer,
        n_probes=n_probes,
        n_samples=n_samples,
    ))
