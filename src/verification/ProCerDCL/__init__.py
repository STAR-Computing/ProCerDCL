"""ProCerDCL: adaptive probabilistic certification with sample reuse.

Decides whether the unknown success probability ``p`` of a binary property
satisfies ``p <= theta`` (Yes) or ``p > theta + eta`` (No) with global error
budget ``delta``, reusing a shared sample pool across the planned intervals.
"""

from .demo_alg import (
    YES,
    NO,
    UNKNOWN,
    CertifyResult,
    certify,
    clopper_pearson,
)

__all__ = [
    "certify",
    "CertifyResult",
    "clopper_pearson",
    "YES",
    "NO",
    "UNKNOWN",
]