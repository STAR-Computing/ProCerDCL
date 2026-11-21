# Reproducing summaries and table exports

Run the commands from the repository root. The table and summary scripts use the Python standard library. The sequential-test example in the README and the threshold-equivalence check require NumPy and SciPy. [CPU example](../README.md#reproduce)

## Pointwise verdicts

```bash
python rq1/analyze_pointwise24.py
```

The script groups per-input CSV records by task and compiler backend. Each backend has 420 records. TorchInductor contributes 401 `Yes`, 3 `No` and 16 `None`. TensorRT contributes 416 `Yes`, 1 `No` and 3 `None`. [Input records](../rq1/pointwise24) · [Generated summary](../rq1/pointwise24_summary.csv)

## Typeset exports

The commands below regenerate the recorded CSV snapshots. The artifact index links the final paper tables for each research question. [Paper artifact index](artifacts.md)

| Command | Input | Generated output |
|---|---|---|
| `python rq4/make_param_tex.py` | [Eight parameter configurations](../rq4/param_study.csv) | [Parameter table](../rq4/param_study.tex) |
| `python rq1/make_all24_tex.py` | [Archived per-task radius statistics](../rq1/rq1_all24_table.csv) | [Archived sweep](../rq1/rq1_all24.tex) and [qualified tasks](../rq1/rq1_attack.tex) |

## CPU checks

```bash
python -m pip install numpy scipy
python rq2/test_rev2_testers.py --runs 2
```

The check compares vectorized binomial stopping thresholds with scalar Clopper–Pearson intervals. The first output line reports mismatches. The `--runs` option controls how many seeds the subsequent empirical check uses.

The calibration and synthetic checks use the registry in `verifier.py`, which also imports PyTorch. The calibration script takes a run count, worker count and output location. The following example generates a small independent CSV. [Calibration script](../rq2/calibration_check.py) · [Published calibration results](../rq2/calibration_check.csv)

```bash
python -m pip install torch
python rq2/calibration_check.py --runs 2 --workers 1 --out outputs/calibration.csv
```

The larger synthetic grid evaluates all registered methods and writes `rq2/synthetic_check.csv`. [Synthetic grid script](../rq2/synthetic_check.py) · [Published grid](../rq2/synthetic_check.csv)
