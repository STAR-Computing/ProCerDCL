# ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization

**ProCerDCL** checks how often a compiled model changes the prediction of its reference model under a chosen input distribution. It draws perturbed inputs, evaluates a binary property on each draw, and stops when the observations support a verdict or the test reaches its sampling budget.

## What it does

- **Certifies a violation probability.** Given independent draws and a stateless oracle, the main sequential test returns `Yes` for evidence that the violation probability is at most `theta`, `No` for evidence that it exceeds `theta + eta`, and `None` when the evidence is inconclusive. The error budget is `delta`. [Method and assumptions](docs/method.md)
- **Distinguishes two kinds of disagreement.** The pointwise property counts any difference between the reference and compiled predictions. The reference-stable property counts draws where the reference keeps the seed label and the compiled prediction changes. [Property definitions](docs/method.md#properties)
- **Measures the cost of a certificate.** A draw cache lets ProCerDCL and the PROVERO baseline see the same perturbations. The paired experiments report the sample count and the certified radius for each method. [Draw cache](src/verification/trace.py) · [Paired results](rq2/rev2_paired.csv)

## Layout

| Path | Contents |
|---|---|
| [`src/verification/`](src/verification) | Sequential tests, the PROVERO baseline, property samplers and paired replay |
| [`src/models/`](src/models) and [`utils.py`](utils.py) | Model definitions, the registry of 27 model–dataset tasks and compiler IDs |
| [`verify_bound.py`](verify_bound.py) and [`verifier.py`](verifier.py) | Clean-label radius search library and verifier registry |
| [`rq1/`](rq1), [`rq2/`](rq2), [`rq3/`](rq3), [`rq4/`](rq4) and [`rq5/`](rq5) | Per-input records, aggregate results, table generators and CPU checks |
| [`docs/`](docs) | Method, reproduction commands and the paper artifact index |
| [`archive/`](archive) | Saved plots and tables from exploratory analyses |

## Experiments

| Study | Question | Results |
|---|---|---|
| [RQ1 · Properties and verdicts](rq1) | What do local consistency verdicts and the clean-label radius tell us? | [Radius table](rq1/paper_tables/rq1_all24.tex) · [Property table](rq1/paper_tables/cond_rev2.tex) · [Pointwise verdicts](rq1/pointwise24_summary.csv) |
| [RQ2 · Sampling cost and conclusiveness](rq2) | How many draws do the methods use, and how often do they reach a decision? | [Sample counts](rq2/paper_tables/rq2_query_number.tex) · [Paired comparison](rq2/paper_tables/pair_misc_base50.tex) · [Seed stability](rq2/paper_tables/seed_stability.tex) |
| [RQ3 · Robustness across configurations](rq3) | How do quantization, GPU choice and perturbation norm affect the radius? | [Simulated INT8](rq3/quant_persetting.pdf) · [TensorRT INT8](rq3/int8_real_scatter.pdf) · [GPU comparison](rq3/xhw_persetting.pdf) · [L-infinity](rq3/linf_persetting.pdf) |
| [RQ4 · Parameter sensitivity](rq4) | How sensitive are the results to the threshold, margin and error budget? | [Parameter table](rq4/paper_tables/param_study.tex) · [Sweep data](rq4/param_study.csv) |
| [RQ5 · Other distributions](rq5) | What changes when perturbations are confined to a 3 × 3 patch? | [Patch table](rq5/paper_tables/patch.tex) · [Measurements](rq5/patch3_summary.csv) |

At `theta = 0.001`, the pointwise study covers 21 image tasks on TorchInductor and TensorRT. Across 840 input–backend pairs, the verifier returned 817 `Yes`, 4 `No` and 19 `None`. The per-input records identify the model, backend and verdict behind each summary row. [Per-input records](rq1/pointwise24) · [Artifact index](docs/artifacts.md)

## Reproduce

**Rebuild the pointwise summary on CPU.** From the repository root, run the following command with Python 3. It reads the included per-input CSVs, prints counts for each backend and writes `rq1/pointwise24_summary.csv`.

```bash
python rq1/analyze_pointwise24.py
```

**Try the sequential test on CPU.** Install NumPy and SciPy, then run this example from the repository root. The seeded sampler returns a violation with probability 0.001. The example prints the verdict and the number of draws the test used.

```python
from random import Random
from src.verification.ProCerDCL.demo_alg import certify_single

rng = Random(7)
result = certify_single(
    theta=0.01, eta=0.005, delta=0.01, batch_size=32,
    draw=lambda: int(rng.random() < 0.001), tester="cs",
)
print(result.answer, result.n_samples)
```

The table generators rebuild the parameter-study and radius-sweep exports from their recorded CSV snapshots. [Reproduction guide](docs/reproduce.md) · [Artifact index](docs/artifacts.md)

## Reference

Zexin Li, Weibo Zhang, Linyi Li, Arvind Easwaran, and Simin Chen. “ProCerDCL: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization.” 2026. [arXiv:2610.36499](https://arxiv.org/abs/2610.36499) · [Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=iHt-2zwAAAAJ&sortby=pubdate&citation_for_view=iHt-2zwAAAAJ:QIV2ME_5wuYC)

```bibtex
@article{li2026procerdcl,
  title   = {{ProCerDCL}: Query-Efficient Black-Box Probabilistic Verification of Deep Learning Compiler Optimization},
  author  = {Li, Zexin and Zhang, Weibo and Li, Linyi and Easwaran, Arvind and Chen, Simin},
  journal = {arXiv preprint arXiv:2610.36499},
  year    = {2026}
}
```
