"""Probabilistic verification methods for compiled-model equivalence.

Each subpackage soundly decides whether the disagreement density ``p`` between a
model and its compiled version satisfies ``p <= theta`` (Yes) or ``p > theta +
eta`` (No) with confidence ``1 - delta``:

    * :mod:`src.verification.PROVERO`   -- closed-form TESTER (arXiv:2002.06864).
    * :mod:`src.verification.ProCerDCL` -- Clopper-Pearson CIs with sample reuse.
"""
