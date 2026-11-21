"""Binary samplers for :class:`src.verification.PROVERO.provero.Provero`.

A sampler exposes ``sample(n) -> int`` returning the number of property
successes among ``n`` i.i.d. draws. PROVERO only ever interacts with the model
through such a sampler (black-box access).

Provided here:
    * :class:`BernoulliSampler` -- synthetic Bernoulli(p) source for testing.
    * :class:`AdversarialRobustnessSampler` -- the paper's adversarial-density
      property for a torch classifier: psi = 1 iff a random perturbation inside
      the Lp epsilon-ball flips the prediction (Section II-A).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["BernoulliSampler", "AdversarialRobustnessSampler"]


class BernoulliSampler:
    """Synthetic sampler: each draw is Bernoulli(``p``). For tests/benchmarks."""

    def __init__(self, p: float, seed: Optional[int] = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int) -> int:
        # Number of successes among n draws ~ Binomial(n, p).
        return int(self._rng.binomial(n, self.p))


class AdversarialRobustnessSampler:
    """Adversarial-density property for a torch classifier (Section II-A).

    For a base input ``x`` with reference label ``y``, the property holds
    (psi = 1) when a perturbation ``delta`` drawn uniformly from the ``Lp``
    epsilon-ball causes a misclassification, i.e. ``f(x + delta) != y``. The
    unknown probability ``p`` PROVERO bounds is then the *adversarial density*
    around ``x``; certifying ``p <= theta`` means the input is robust up to a
    density ``theta``.

    Args:
        model: a torch classifier mapping ``(B, ...) -> (B, num_classes)`` logits.
        x: the base input, shape ``(...)`` (no batch dimension).
        y: reference class. If ``None``, the model's own prediction on ``x`` is
            used (so density measures deviation from the clean prediction).
        epsilon: perturbation radius.
        norm: ``"inf"`` (uniform in the L-inf box) or ``"2"`` (uniform in the
            L2 ball).
        clip: optional ``(low, high)`` to clamp perturbed inputs to the valid
            input range (e.g. ``(0.0, 1.0)``).
        batch_size: number of perturbations evaluated per forward pass.
        device: torch device; defaults to the model's first parameter device.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        model,
        x,
        y: Optional[int] = None,
        *,
        epsilon: float = 0.1,
        norm: str = "inf",
        clip: Optional[Tuple[float, float]] = None,
        batch_size: int = 256,
        device=None,
        seed: Optional[int] = None,
        mask=None,
    ) -> None:
        import torch

        if norm not in ("inf", "2"):
            raise ValueError("norm must be 'inf' or '2'")

        self._torch = torch
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.x = x.to(self.device)
        # Optional support mask (shape of x): perturb only where mask == 1,
        # e.g. a fixed image patch; the ball is then taken within that support.
        self.mask = None if mask is None else mask.to(self.device).reshape(self.x.shape)
        self.epsilon = float(epsilon)
        self.norm = norm
        self.clip = clip
        self.batch_size = int(batch_size)
        self._gen = torch.Generator(device=self.device)
        if seed is not None:
            self._gen.manual_seed(seed)

        # Reference label: model prediction on the clean input if not supplied.
        if y is None:
            with torch.no_grad():
                logits = self.model(self.x.unsqueeze(0).to(self.device))
            self.y = int(logits.argmax(dim=1).item())
        else:
            self.y = int(y)

    def _perturb(self, b: int):
        """Draw ``b`` perturbations uniformly from the Lp epsilon-ball."""
        torch = self._torch
        shape = (b, *self.x.shape)
        if self.norm == "inf":
            # Uniform in [-eps, eps] per coordinate.
            u = torch.rand(shape, generator=self._gen, device=self.device)
            delta = (2.0 * u - 1.0) * self.epsilon
            if self.mask is not None:
                delta = delta * self.mask
        else:  # L2: direction ~ N(0, I), radius ~ eps * U^(1/d)
            # Ball volume scales as radius**d in the subspace of active coordinates.
            d = self.x.numel() if self.mask is None else int(self.mask.sum().item())
            g = torch.randn(shape, generator=self._gen, device=self.device)
            if self.mask is not None:
                g = g * self.mask
            flat = g.reshape(b, -1)
            unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            radius = self.epsilon * torch.rand(
                b, 1, generator=self._gen, device=self.device
            ) ** (1.0 / d)
            delta = (unit * radius).reshape(shape)
        return delta

    def sample(self, n: int) -> int:
        """Evaluate ``n`` random perturbations; count the misclassified ones."""
        torch = self._torch
        adversarial = 0
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                b = min(self.batch_size, remaining)
                inputs = self.x.unsqueeze(0) + self._perturb(b)
                if self.clip is not None:
                    inputs = inputs.clamp(self.clip[0], self.clip[1])
                # Pad a partial multi-sample batch up to batch_size so the
                # compiled model only ever sees a fixed set of batch shapes
                # ({1, batch_size}); this avoids torch.compile going fully
                # dynamic on a 3rd shape, which breaks the onnxrt export of some
                # models. Only the first b predictions are counted.
                nb = inputs.shape[0]
                if 1 < nb < self.batch_size:
                    pad = inputs[-1:].expand(
                        self.batch_size - nb, *inputs.shape[1:])
                    inputs = torch.cat([inputs, pad], dim=0)
                preds = self.model(inputs).argmax(dim=1)[:nb]
                adversarial += int((preds != self.y).sum().item())
                remaining -= b
        return adversarial

    def _flags(self, b: int):
        """Evaluate ``b`` fresh perturbations and return their 0/1 flags in
        draw order (a list)."""
        torch = self._torch
        with torch.no_grad():
            inputs = self.x.unsqueeze(0) + self._perturb(b)
            if self.clip is not None:
                inputs = inputs.clamp(self.clip[0], self.clip[1])
            nb = inputs.shape[0]
            if 1 < nb < self.batch_size:
                pad = inputs[-1:].expand(self.batch_size - nb, *inputs.shape[1:])
                inputs = torch.cat([inputs, pad], dim=0)
            preds = self.model(inputs).argmax(dim=1)[:nb]
            return (preds != self.y).int().tolist()

    def draw(self) -> int:
        """Single 0/1 draw of the property (for ProCerDCL's ``draw()`` API).

        Draws are generated in GPU batches of ``batch_size`` perturbations and
        handed out one at a time from a FIFO buffer, so a sequential tester that
        checks after every sample no longer pays one padded forward pass per
        sample (which made an unbuffered sampler 10-50x slower).
        The draws remain i.i.d.; only the amortisation changes.
        """
        buf = getattr(self, "_buf", None)
        if not buf:
            self._buf = buf = self._flags(self.batch_size)
            self._buf_pos = 0
        v = buf[self._buf_pos]
        self._buf_pos += 1
        if self._buf_pos >= len(buf):
            self._buf = []
        return v


class ConditionalDisagreementSampler(AdversarialRobustnessSampler):
    """Compiler-attributable property around a seed input.

    ``property="cond"``      psi = 1[ M(x') == y0  and  C(x') != y0 ]
    ``property="pointwise"`` psi = 1[ M(x') != C(x') ]
    ``property="clean"``     psi = 1[ C(x') != y0 ]  (the RQ1 property)

    Both models are queried on the same perturbed batch. Everything else
    (ball geometry, batching, buffered ``draw``) is inherited.
    """

    def __init__(self, model, ref_model, x, y=None, *, property: str = "cond", **kw):
        if property not in ("cond", "pointwise", "clean"):
            raise ValueError("property must be cond|pointwise|clean")
        super().__init__(model, x, y, **kw)
        self.ref_model = ref_model.eval()
        self.property = property

    def _flags(self, b: int):
        torch = self._torch
        with torch.no_grad():
            inputs = self.x.unsqueeze(0) + self._perturb(b)
            if self.clip is not None:
                inputs = inputs.clamp(self.clip[0], self.clip[1])
            nb = inputs.shape[0]
            if 1 < nb < self.batch_size:
                pad = inputs[-1:].expand(self.batch_size - nb, *inputs.shape[1:])
                inputs = torch.cat([inputs, pad], dim=0)
            pc = self.model(inputs).argmax(dim=1)[:nb]
            if self.property == "clean":
                return (pc != self.y).int().tolist()
            pm = self.ref_model(inputs).argmax(dim=1)[:nb]
            if self.property == "pointwise":
                return (pm != pc).int().tolist()
            # A reference flip contributes a zero draw to the joint event.
            return ((pm == self.y) & (pc != self.y)).int().tolist()

    def sample(self, n: int) -> int:
        cnt = 0
        remaining = n
        while remaining > 0:
            b = min(self.batch_size, remaining)
            cnt += sum(self._flags(b))
            remaining -= b
        return cnt


class TwoModelPropertySampler(AdversarialRobustnessSampler):
    """Adversarial-density sampler over a property of BOTH models (G2).

    :class:`AdversarialRobustnessSampler` queries only the compiled model and
    reports ``C(x') != y0``, which mixes compiler-induced divergence with the
    model's own sensitivity to perturbation. To isolate the compiler we need
    properties that query the reference model too:

    ``conditional``
        ``psi = 1[M(x') == y0 and C(x') != y0]`` -- the reference model still
        predicts the clean label but the compiled one does not, i.e. the flip is
        attributable to compilation rather than to a natural class change.
    ``pointwise``
        ``psi = 1[M(x') != C(x')]`` -- the two models simply disagree.

    Every draw costs two forward passes, one per model, which is why these
    densities are an order of magnitude more expensive to certify.
    """

    def __init__(self, cl_model, ref_model, x, y, *, prop="conditional", **kw):
        super().__init__(cl_model, x, y=y, **kw)
        self.ref_model = ref_model.eval()
        if prop not in ("conditional", "pointwise"):
            raise ValueError("prop must be 'conditional' or 'pointwise'")
        self.prop = prop
        # ProCerDCL consumes the stream one draw at a time via draw(). Serving
        # that literally would run a batch of ONE through two models per draw,
        # which at these thresholds (1e5-1e6 draws per probe) is the difference
        # between minutes and hours. Draw in GPU batches and hand them out from
        # a buffer instead; the stream is statistically identical.
        self._buf: list = []

    def _flags(self, b: int) -> list:
        """Property indicator for ``b`` fresh perturbations."""
        torch = self._torch
        with torch.no_grad():
            inputs = self.x.unsqueeze(0) + self._perturb(b)
            if self.clip is not None:
                inputs = inputs.clamp(self.clip[0], self.clip[1])
            nb = inputs.shape[0]
            if 1 < nb < self.batch_size:
                pad = inputs[-1:].expand(self.batch_size - nb, *inputs.shape[1:])
                inputs = torch.cat([inputs, pad], dim=0)
            cl = self.model(inputs).argmax(dim=1)[:nb]
            ref = self.ref_model(inputs).argmax(dim=1)[:nb]
            hit = ((ref == self.y) & (cl != self.y)) if self.prop == "conditional" \
                else (ref != cl)
        return hit.int().tolist()

    def draw(self) -> int:
        if not self._buf:
            self._buf = self._flags(self.batch_size)
        return self._buf.pop()

    def sample(self, n: int) -> int:
        """Count how many of ``n`` random perturbations satisfy the property."""
        torch = self._torch
        hits = 0
        remaining = n
        with torch.no_grad():
            while remaining > 0:
                b = min(self.batch_size, remaining)
                inputs = self.x.unsqueeze(0) + self._perturb(b)
                if self.clip is not None:
                    inputs = inputs.clamp(self.clip[0], self.clip[1])
                nb = inputs.shape[0]
                if 1 < nb < self.batch_size:
                    pad = inputs[-1:].expand(
                        self.batch_size - nb, *inputs.shape[1:])
                    inputs = torch.cat([inputs, pad], dim=0)
                cl = self.model(inputs).argmax(dim=1)[:nb]
                ref = self.ref_model(inputs).argmax(dim=1)[:nb]
                if self.prop == "conditional":
                    hit = (ref == self.y) & (cl != self.y)
                else:
                    hit = ref != cl
                hits += int(hit.sum().item())
                remaining -= b
        return hits
