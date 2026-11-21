"""Trace-once / replay-many support.

A :class:`DrawCache` wraps a black-box binary sampler at one fixed radius and
memoises its i.i.d. draws in order. Several verifiers can then be run on the
*same* oracle draws: the first verifier to request draw ``j`` triggers the
oracle, later ones read it back. Under assumption A3 (statelessness) the
replayed prefix is an i.i.d. Bernoulli(p) sequence for every verifier, so each
verifier's marginal guarantee is untouched, and the comparison between
verifiers becomes paired (identical draws) instead of independent.

The cache also exposes the two sampler APIs used in the artifact:
``draw() -> int`` (ProCerDCL / SPRT) and ``sample(n) -> int`` (PROVERO).
A ``view()`` returns a fresh cursor over the shared prefix.
"""
from __future__ import annotations

from typing import Callable, List


class DrawCache:
    def __init__(self, draw_fn: Callable[[], int]):
        self._draw_fn = draw_fn
        self.buf: List[int] = []
        self.oracle_calls = 0

    def get(self, j: int) -> int:
        while len(self.buf) <= j:
            self.buf.append(int(self._draw_fn()))
            self.oracle_calls += 1
        return self.buf[j]

    def view(self) -> "CacheView":
        return CacheView(self)

    def prefix(self, n: int) -> List[int]:
        self.get(n - 1)
        return self.buf[:n]


class CacheView:
    """Sequential cursor over a :class:`DrawCache` (one per verifier)."""

    def __init__(self, cache: DrawCache):
        self.cache = cache
        self.pos = 0

    def draw(self) -> int:
        v = self.cache.get(self.pos)
        self.pos += 1
        return v

    def sample(self, n: int) -> int:
        # Each verifier advances its own cursor through the shared draws.
        self.cache.get(self.pos + n - 1)
        s = sum(self.cache.buf[self.pos:self.pos + n])
        self.pos += n
        return s

    @property
    def consumed(self) -> int:
        return self.pos
