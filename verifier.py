"""Certify that a ``torch.compile``d model is semantically equivalent to the
original, using a selectable probabilistic verification method.

Pipeline
--------
1. Load a (model, tokenizer, dataset) for a task id via :func:`utils.load_task`.
2. Compile the model with :func:`torch.compile`.
3. Define the property psi(x) = "the compiled model prediction disagrees with
   the original model prediction" on one input at a time.
4. Run the selected verifier to soundly decide whether the disagreement density
   ``p`` satisfies ``p <= theta`` (property satisfied) or ``p > theta + eta``
   (property violated), with confidence ``1 - delta``.

Verification methods (``--verifier-id``)
----------------------------------------
* ``0`` -- ProCerDCL (:mod:`src.verification.ProCerDCL`): adaptive binary-search
  certification with Clopper-Pearson CIs and a shared, reused sample pool.
* ``1`` -- PROVERO (:mod:`src.verification.PROVERO`): adaptive binary-search
  certification with the closed-form TESTER primitive (arXiv:2002.06864).

Both methods share the same decision contract (``Yes`` iff ``p <= theta``), so
the two are drop-in alternatives over the very same equivalence sampler.

Example
-------
    python verifier.py --task-id 1 --verifier-id 1 --theta 0.05 --eta 0.01
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.verification.PROVERO import Provero, YES, NO
from src.verification import ProCerDCL
from src.semantic_checker import agreement_mask, predict_label
from utils import TASKS, load_task, RANDOM_SEED, result_dir
from utils import COMPILER, model_cache_path

# ``predict_label`` is re-exported here for callers that import it from this
# module (e.g. verify_bound); the canonical definition lives in semantic_checker.
__all__ = ["collate", "predict_label", "run_verifier", "VerificationResult"]


def collate(dataset, indices, tokenizer, device):
    """Build a model-ready batch from ``dataset`` rows at ``indices``.

    Returns a ``Tensor`` for vision tasks (stacked ``pixel_values``) or a dict
    of tokenized tensors for text-classification tasks.
    """
    rows = [dataset[int(i)] for i in indices]

    if "pixel_values" in rows[0]:  # vision task
        return torch.stack([r["pixel_values"] for r in rows]).to(device)

    # text-classification task: find the sentence column and tokenize.
    if tokenizer is None:
        raise ValueError("a tokenizer is required for text datasets")
    text_col = next(
        (c for c in ("sentence", "text", "sentence1") if c in rows[0]), None
    )
    if text_col is None:
        raise ValueError(f"no text column found in row keys {list(rows[0])}")
    texts = [r[text_col] for r in rows]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in enc.items()}


# --------------------------------------------------------------------------- #
# Equivalence sampler
# --------------------------------------------------------------------------- #
class CompiledEquivalenceSampler:
    """Binary sampler for the original-vs-compiled disagreement property.

    Each draw picks a random dataset input and reports ``1`` iff the compiled
    model's prediction disagrees with the original model's (via the canonical
    :func:`src.semantic_checker.agreement_mask`). Its success probability is the
    disagreement density ``p`` the verifier certifies against ``theta``.
    """

    def __init__(
        self, model, cl_model, dataset, tokenizer, *, device, batch_size, seed
    ) -> None:
        self.model = model
        self.cl_model = cl_model
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.n_data = len(dataset)
        self._buffer: list[int] = []

    def _disagreement_flags(self, b: int) -> list[int]:
        idx = self.rng.integers(0, self.n_data, size=b)
        x = collate(self.dataset, idx, self.tokenizer, self.device)
        return (~agreement_mask(self.model, self.cl_model, x)).int().tolist()

    def sample(self, n: int) -> int:
        """Number of disagreements over ``n`` fresh inputs (PROVERO)."""
        disagreements = 0
        remaining = n
        while remaining > 0:
            b = min(self.batch_size, remaining)
            disagreements += sum(self._disagreement_flags(b))
            remaining -= b
        return disagreements

    def draw(self) -> int:
        """Single disagreement flag for one fresh input (ProCerDCL)."""
        if not self._buffer:
            self._buffer = self._disagreement_flags(self.batch_size)
        return self._buffer.pop()


# --------------------------------------------------------------------------- #
# Verifier dispatch
# --------------------------------------------------------------------------- #
@dataclass
class VerificationResult:
    """Method-agnostic certificate produced by :func:`run_verifier`."""

    method: str
    answer: str
    n_samples: int
    extra: Dict[str, Any] = field(default_factory=dict)


def _run_procerdcl(sampler, theta, eta, delta, batch_size) -> VerificationResult:
    res = ProCerDCL.certify(
        theta, eta, delta, batch_size=batch_size, draw=sampler.draw
    )
    return VerificationResult(
        method="ProCerDCL",
        answer=res.answer,
        n_samples=res.n_samples,
        extra={"intervals": len(res.intervals)},
    )


def _run_provero(sampler, theta, eta, delta, batch_size=None) -> VerificationResult:
    # batch_size is accepted for a uniform runner signature but unused: PROVERO
    # draws its own sample sizes internally.
    res = Provero(sampler).bin_certify(theta, eta, delta)
    return VerificationResult(
        method="PROVERO",
        answer=res.answer,
        n_samples=res.n_samples,
        extra={"tester_calls": res.n_tester_calls},
    )


def _run_direct(sampler, theta, eta, delta, batch_size=None) -> VerificationResult:
    """PROVERO's own single fixed-sample tester on (theta, theta+eta)."""
    res = Provero(sampler).direct_certify(theta, eta, delta)
    return VerificationResult(
        method="DirectCertify",
        answer=res.answer,
        n_samples=res.n_samples,
        extra={},
    )


def _run_sprt(sampler, theta, eta, delta, batch_size=None) -> VerificationResult:
    """Truncated SPRT with indifference region [theta, theta+eta].

    Wald's sequential probability-ratio test of H0: p = theta vs
    H1: p = theta + eta with alpha = beta = delta; truncated at PROVERO's
    fixed-sample tester budget with a forced midpoint decision, so a
    Yes/No verdict is always produced (classical truncated-SPRT design).
    """
    import math as _math
    p0, p1 = theta, theta + eta
    # log-likelihood-ratio increments
    l1 = _math.log(p1 / p0) if p0 > 0 else float("inf")
    l0 = _math.log((1.0 - p1) / (1.0 - p0))
    hi = _math.log((1.0 - delta) / delta)   # accept H1 (No)
    lo = _math.log(delta / (1.0 - delta))   # accept H0 (Yes)
    n_max = Provero._tester_num_samples(p0, p1, delta)
    llr = 0.0
    n = k = 0
    while n < n_max:
        y = sampler.draw()
        n += 1
        k += y
        llr += l1 if y else l0
        if llr >= hi:
            return VerificationResult("SPRT", NO, n, {"llr": llr, "k": k})
        if llr <= lo:
            return VerificationResult("SPRT", YES, n, {"llr": llr, "k": k})
    cutoff = p0 + (p1 - p0) / 2.0  # forced midpoint decision at truncation
    ans = YES if (k / n) <= cutoff else NO
    return VerificationResult("SPRT", ans, n, {"llr": llr, "k": k,
                                               "truncated": True})


def _run_procerdcl_noreuse(sampler, theta, eta, delta, batch_size) -> VerificationResult:
    from src.verification.ProCerDCL.demo_alg import certify_variant
    res = certify_variant(theta, eta, delta, batch_size=batch_size,
                          draw=sampler.draw, reuse=False)
    return VerificationResult("ProCerDCL-NR", res.answer, res.n_samples,
                              {"intervals": len(res.intervals)})


def _run_procerdcl_uniform(sampler, theta, eta, delta, batch_size) -> VerificationResult:
    from src.verification.ProCerDCL.demo_alg import certify_variant
    res = certify_variant(theta, eta, delta, batch_size=batch_size,
                          draw=sampler.draw, uniform_delta=True)
    return VerificationResult("ProCerDCL-UD", res.answer, res.n_samples,
                              {"intervals": len(res.intervals)})


def _procerdcl_factory(name, **kw):
    def _run(sampler, theta, eta, delta, batch_size):
        res = ProCerDCL.certify(theta, eta, delta, batch_size=batch_size,
                                draw=sampler.draw, **kw)
        return VerificationResult(name, res.answer, res.n_samples,
                                  {"intervals": len(res.intervals),
                                   **{f"last_{k}": v for k, v in res.last_stats.items()}})
    return _run


def _single_factory(name, **kw):
    def _run(sampler, theta, eta, delta, batch_size):
        from src.verification.ProCerDCL.demo_alg import certify_single
        res = certify_single(theta, eta, delta, batch_size=batch_size,
                             draw=sampler.draw, **kw)
        return VerificationResult(name, res.answer, res.n_samples, {})
    return _run


# verifier-id -> (display name, runner). Add new methods here.
VERIFIERS = {
    0: ("ProCerDCL", _run_procerdcl),
    1: ("PROVERO", _run_provero),
    2: ("DirectCertify", _run_direct),
    3: ("SPRT", _run_sprt),
    4: ("ProCerDCL-NR", _run_procerdcl_noreuse),
    5: ("ProCerDCL-UD", _run_procerdcl_uniform),
    # anytime-valid confidence-sequence tester and None-free modes
    6: ("ProCerDCL-CS", _procerdcl_factory("ProCerDCL-CS", tester="cs")),
    7: ("ProCerDCL-F", _procerdcl_factory("ProCerDCL-F", tester="cp", forced=True)),
    8: ("ProCerDCL-CS-F", _procerdcl_factory("ProCerDCL-CS-F", tester="cs", forced=True)),
    # main algorithm: a single sequential test on (theta, theta+eta) at the full delta
    9: ("Single-CP", _single_factory("Single-CP", tester="cp")),
    10: ("Single-CS", _single_factory("Single-CS", tester="cs")),
    11: ("Single-CP-G", _single_factory("Single-CP-G", tester="cp", geometric=1.1)),
    12: ("ProCerDCL-G", _procerdcl_factory("ProCerDCL-G", tester="cp", geometric=1.1)),
}


def run_verifier(
    verifier_id, sampler, theta, eta, delta, batch_size
) -> VerificationResult:
    """Dispatch to the verification method selected by ``verifier_id``."""
    if verifier_id not in VERIFIERS:
        raise ValueError(
            f"unknown verifier-id {verifier_id}; "
            f"choose one of {sorted(VERIFIERS)}"
        )
    _, runner = VERIFIERS[verifier_id]
    return runner(sampler, theta, eta, delta, batch_size)


def save_results(task, result, args, backend, satisfied) -> Path:
    """Save the certification outcome under :data:`utils.RESULT_DIR`.

    Layout: ``<RESULT_DIR>/<task_slug>/<backend>/<method>/cert.json`` describing
    the run.
    """
    out = result_dir(task, backend, result.method)
    meta = {
        "kind": "verification",
        "task_id": args.task_id,
        "task": f"{task.model_name} on {task.dataset_name}",
        "method": result.method,
        "compile_backend": backend,
        "model_cache": str(model_cache_path(args.task_id)),
        "theta": args.theta,
        "eta": args.eta,
        "delta": args.delta,
        "answer": result.answer,
        "satisfied": satisfied,
        "n_samples": result.n_samples,
        "seed": args.seed,
        **result.extra,
    }
    (out / "cert.json").write_text(json.dumps(meta, indent=2))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verifier-id", type=int, default=0,
                   choices=sorted(VERIFIERS),
                   help="verification method id: 0 is ProCerDCL, 1 is PROVERO")
    p.add_argument("--task-id", type=int, default=1,
                   help="task id 1..10 (see utils.list_tasks)")
    p.add_argument("--theta", type=float, default=0.95,
                   help="disagreement-density threshold")
    p.add_argument("--eta", type=float, default=0.01, help="error tolerance")
    p.add_argument("--delta", type=float, default=0.01, help="1 - confidence")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--compile-id", type=int, default=0,
                   help="torch.compile backend id")
    p.add_argument("--seed", type=int, default=RANDOM_SEED,
                   help="random seed for the sampler")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    torch.manual_seed(RANDOM_SEED)

    method_name, _ = VERIFIERS[args.verifier_id]
    task = TASKS[args.task_id]
    print(f"Task {args.task_id}: {task.model_name} on {task.dataset_name}")

    # 1. Load model + dataset.
    model, tokenizer, dataset = load_task(args.task_id)
    model = model.to(device).eval()
    print(f"Dataset size: {len(dataset)}")

    # 2. Compile the backend (fall back to eager if a backend is unavailable).
    backend = COMPILER[args.compile_id]
    cl_model = torch.compile(model, backend=backend)
    print(f"Compiled model with torch.compile(backend={backend!r}).")

    # 3. Build the sampler over the property "models disagree".
    sampler = CompiledEquivalenceSampler(
        model, cl_model, dataset, tokenizer,
        device=device, batch_size=args.batch_size, seed=args.seed,
    )

    # 4. Certify with the selected verifier.
    print(f"\nRunning {method_name}: theta={args.theta}, eta={args.eta}, "
          f"delta={args.delta} ...")
    result = run_verifier(
        args.verifier_id, sampler,
        args.theta, args.eta, args.delta, args.batch_size,
    )

    satisfied = result.answer == YES
    print(f"\n--- {result.method} certificate ---")
    print(f"answer           : {result.answer}")
    print(f"samples drawn    : {result.n_samples}")
    for k, v in result.extra.items():
        print(f"{k:<17}: {v}")
    print(
        f"\nProperty (compiled model agrees with original up to density "
        f"{args.theta}): {'SATISFIED' if satisfied else 'NOT satisfied'}"
    )

    out = save_results(task, result, args, backend, satisfied)
    print(f"\nSaved cert.json to {out}")
    return result


if __name__ == "__main__":
    main()
