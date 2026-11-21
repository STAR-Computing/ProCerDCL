"""PROVERO: scalable quantitative verification for deep neural networks.

Implementation of the core algorithms from

    Baluta, Chua, Meel, Saxena.
    "Scalable Quantitative Verification For Deep Neural Networks."
    arXiv:2002.06864 (ICSE 2021).

Problem (Section III). Given a black-box 0/1 property ``psi`` defined over a
sampleable input space, let ``p = E[psi]`` be the (unknown) probability that the
property holds. Given a threshold ``theta``, tolerance ``eta`` and confidence
``delta``, decide ``Yes`` (``p <= theta``) or ``No`` (``p > theta``) such that

    Pr[ answer = 'Yes' | p <= theta       ] >= 1 - delta
    Pr[ answer = 'No'  | p >  theta + eta  ] >= 1 - delta

using as few samples of ``psi`` as possible.

The class :class:`Provero` implements:

    * :meth:`tester`        -- TESTER primitive            (Algorithm 2)
    * :meth:`bin_certify`   -- BINPCERTIFY binary search    (Algorithm 3)
    * ``_create_interval``  -- CREATEINTERVAL               (Algorithm 4)
    * :meth:`direct_certify`-- naive single-TESTER baseline (METAPROVERO with
                               theta1 = theta, theta2 = theta + eta)

The verifier only needs black-box access to ``psi`` through a *binary sampler*
(see :mod:`src.verification.PROVERO.samplers`): any object exposing ``sample(n) -> int``
that returns the number of successes over ``n`` i.i.d. draws, or a plain
callable with the same contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Tuple

# Decision labels used throughout (match the paper's 'Yes' / 'No' / None).
YES = "Yes"
NO = "No"
UNKNOWN = "None"

# A binary sampler: an object with ``sample(n) -> successes`` or a callable.
BinarySampler = Any

# Tolerance for interval-width comparisons against ``eta``. CREATEINTERVAL floors
# the width at ``eta`` via ``theta1 + max(eta, alpha/2)``, but that sum rounds to
# slightly above ``eta`` in floating point (e.g. 0.15 - 0.10 == 0.05000000000002).
# Without slack the width stays nominally "> eta" forever and BINPCERTIFY never
# reaches its final test -- an infinite loop. The paper assumes exact reals.
_WIDTH_EPS = 1e-9


@dataclass
class ProveroResult:
    """Outcome of a certification call.

    Attributes:
        answer: one of ``"Yes"`` (p <= theta) or ``"No"`` (p > theta + eta).
        n_samples: total number of ``psi`` samples drawn during the call.
        n_tester_calls: number of TESTER invocations during the call.
        theta, eta, delta: the parameters the call was made with.
    """

    answer: str
    n_samples: int
    n_tester_calls: int
    theta: float
    eta: float
    delta: float

    def __bool__(self) -> bool:  # truthy iff a sound 'Yes' was returned
        return self.answer == YES


class Provero:
    """Sound quantitative verifier (PROVERO).

    Args:
        sampler: black-box access to the property. Either an object with a
            ``sample(n: int) -> int`` method returning the number of successes
            among ``n`` i.i.d. Bernoulli(p) draws, or a callable ``f(n) -> int``
            with the same contract.

    Example:
        >>> from src.verification.PROVERO import Provero, BernoulliSampler
        >>> verifier = Provero(BernoulliSampler(p=0.05, seed=0))
        >>> res = verifier.bin_certify(theta=0.10, eta=1e-2, delta=1e-2)
        >>> res.answer
        'Yes'
    """

    def __init__(self, sampler: BinarySampler) -> None:
        if hasattr(sampler, "sample"):
            self._draw_fn: Callable[[int], int] = sampler.sample
        elif callable(sampler):
            self._draw_fn = sampler
        else:
            raise TypeError(
                "sampler must expose .sample(n) -> int or be callable f(n) -> int"
            )
        self.sampler = sampler
        # Per-call counters (reset at the start of each top-level certify call).
        self._n_samples = 0
        self._n_tester_calls = 0

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _draw(self, n: int) -> int:
        """Draw ``n`` i.i.d. samples; return the number of successes."""
        successes = int(self._draw_fn(n))
        if not 0 <= successes <= n:
            raise ValueError(
                f"sampler returned {successes} successes for n={n} draws"
            )
        self._n_samples += n
        return successes

    # ------------------------------------------------------------------ #
    # Algorithm 2: TESTER
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tester_num_samples(theta1: float, theta2: float, delta: float) -> int:
        """N from Algorithm 2, line 1."""
        gap = theta2 - theta1
        coeff = (math.sqrt(3.0 * theta1) + math.sqrt(2.0 * theta2)) ** 2
        return int(math.ceil(coeff / (gap * gap) * math.log(1.0 / delta)))

    @staticmethod
    def _tester_etas(theta1: float, theta2: float) -> Tuple[float, float]:
        """eta1, eta2 from Algorithm 2, lines 2-3."""
        gap = theta2 - theta1
        if theta1 <= 0.0:
            eta1 = 0.0
        else:
            eta1 = gap / (1.0 + math.sqrt(2.0 * theta2 / (3.0 * theta1)))
        eta2 = gap - eta1
        return eta1, eta2

    def tester(self, theta1: float, theta2: float, delta: float) -> str:
        """TESTER (Algorithm 2).

        Soundly decides between ``p <= theta1`` and ``p > theta2`` (with
        confidence ``1 - delta`` each); returns ``"None"`` only when no decision
        is reached. Requires ``0 <= theta1 < theta2 <= 1`` and ``0 < delta < 1``.
        """
        if not 0.0 <= theta1 < theta2 <= 1.0:
            raise ValueError("require 0 <= theta1 < theta2 <= 1")
        if not 0.0 < delta < 1.0:
            raise ValueError("require 0 < delta < 1")

        n = self._tester_num_samples(theta1, theta2, delta)
        eta1, eta2 = self._tester_etas(theta1, theta2)

        successes = self._draw(n)
        p_hat = successes / n
        self._n_tester_calls += 1

        cutoff = theta1 + eta1
        if p_hat <= cutoff:
            return YES
        if p_hat > cutoff:
            return NO
        return UNKNOWN

    # ------------------------------------------------------------------ #
    # Algorithm 4: CREATEINTERVAL
    # ------------------------------------------------------------------ #
    @staticmethod
    def _create_interval(
        theta: float, theta1: float, theta2: float, eta: float, left: bool
    ) -> Tuple[float, float]:
        """CREATEINTERVAL (Algorithm 4).

        Proving (``left=True``) intervals keep the right edge fixed at ``theta``
        and grow the left edge toward it; refuting (``left=False``) intervals
        keep the left edge fixed at ``theta + eta`` and shrink the right edge
        toward it. Each step at least halves the interval width.
        """
        if theta == 0.0 and left:
            return theta, theta + eta
        if theta1 == 0.0 and theta2 == 0.0 and left:
            return 0.0, theta
        if theta1 == 0.0 and theta2 == 0.0 and not left:
            return theta + eta, 1.0

        alpha = theta2 - theta1
        if left:
            return theta2 - max(eta, alpha / 2.0), theta2
        return theta1, theta1 + max(eta, alpha / 2.0)

    # ------------------------------------------------------------------ #
    # Algorithm 3: BINPCERTIFY
    # ------------------------------------------------------------------ #
    @staticmethod
    def _bin_certify_budget(theta: float, eta: float) -> int:
        """n from Algorithm 3, line 3 -- an upper bound on TESTER calls.

        Used to split the confidence budget (``delta_min = delta / n``) so the
        union bound over all TESTER calls keeps total error within ``delta``.
        """

        def steps(num: float, den: float) -> int:
            if num <= 0.0 or den <= 0.0:
                return 0
            return max(0, int(math.ceil(math.log2(num / den))))

        return 3 + steps(theta, eta) + steps(1.0 - theta - eta, eta)

    def bin_certify(
        self, theta: float, eta: float, delta: float
    ) -> ProveroResult:
        """BINPCERTIFY (Algorithm 3): the adaptive binary-search certifier.

        Returns a :class:`ProveroResult`. Soundness (Theorem 5.1):

            Pr[answer = 'Yes' | p <= theta]      >= 1 - delta
            Pr[answer = 'No'  | p >  theta + eta] >= 1 - delta
        """
        self._validate(theta, eta, delta)
        self._reset_counters()

        n = self._bin_certify_budget(theta, eta)
        delta_min = delta / n  # confidence per TESTER call (union bound)

        theta1_l = theta2_l = 0.0  # proving interval (left of theta)
        theta1_r = theta2_r = 0.0  # refuting interval (right of theta + eta)

        while True:
            # Proving side: try to show p <= theta.
            theta1_l, theta2_l = self._create_interval(
                theta, theta1_l, theta2_l, eta, left=True
            )
            if theta2_l - theta1_l > eta + _WIDTH_EPS:
                if self.tester(theta1_l, theta2_l, delta_min) == YES:
                    return self._result(YES, theta, eta, delta)

            # Refuting side: try to show p > theta + eta.
            theta1_r, theta2_r = self._create_interval(
                theta, theta1_r, theta2_r, eta, left=False
            )
            if theta2_r - theta1_r > eta + _WIDTH_EPS:
                if self.tester(theta1_r, theta2_r, delta_min) == NO:
                    return self._result(NO, theta, eta, delta)

            # Both intervals have shrunk to <= eta: do the final direct test.
            if (theta2_l - theta1_l <= eta + _WIDTH_EPS) and (
                theta2_r - theta1_r <= eta + _WIDTH_EPS
            ):
                ans = self.tester(theta, theta + eta, delta_min)
                return self._result(ans, theta, eta, delta)

    # ------------------------------------------------------------------ #
    # Naive baseline (METAPROVERO with theta1 = theta, theta2 = theta + eta)
    # ------------------------------------------------------------------ #
    def direct_certify(
        self, theta: float, eta: float, delta: float
    ) -> ProveroResult:
        """Single TESTER call on ``(theta, theta + eta)``.

        The simplest sound certifier (no early stopping); useful as a baseline
        against :meth:`bin_certify`.
        """
        self._validate(theta, eta, delta)
        self._reset_counters()
        ans = self.tester(theta, theta + eta, delta)
        return self._result(ans, theta, eta, delta)

    # Default entry point.
    def certify(self, theta: float, eta: float, delta: float) -> ProveroResult:
        return self.bin_certify(theta, eta, delta)

    __call__ = certify

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate(theta: float, eta: float, delta: float) -> None:
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be in [0, 1]")
        if not 0.0 < eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        if theta + eta > 1.0:
            raise ValueError("theta + eta must be <= 1")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")

    def _reset_counters(self) -> None:
        self._n_samples = 0
        self._n_tester_calls = 0

    def _result(
        self, answer: str, theta: float, eta: float, delta: float
    ) -> ProveroResult:
        return ProveroResult(
            answer=answer,
            n_samples=self._n_samples,
            n_tester_calls=self._n_tester_calls,
            theta=theta,
            eta=eta,
            delta=delta,
        )
