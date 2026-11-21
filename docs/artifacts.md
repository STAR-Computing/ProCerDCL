# Paper artifact index

Each research question has a top-level directory. The paper column links to the final LaTeX result tables and PDF figure panels used in the evaluation. The measurements column links to their associated records and analysis outputs. [Reproduction guide](reproduce.md)

| Research question | Paper table or figure | Measurements and supporting outputs |
|---|---|---|
| [RQ1 · Properties and verdicts](../rq1) | [Table 2](../rq1/paper_tables/rq1_all24.tex) (`tab:all24`), clean-label radius sweep | [Qualification comparison](../rq1/qualification_compare.csv) · [Earlier sweep snapshot](../rq1/rq1_all24_table.csv) |
| RQ1 | [Table 3](../rq1/paper_tables/cond_rev2.tex) (`tab:cond`), reference-stable and pointwise properties | [Pointwise records](../rq1/pointwise24) · [840 verdicts](../rq1/pointwise24_summary.csv) · [Conditional summary](../rq1/rev2_cond.csv) · [Tight-threshold comparison](../rq1/rare_event_compare.csv) |
| RQ1 | [Table 5](../rq2/paper_tables/pair_misc_base50.tex) (`tab:miscompile`), injected compiler faults. Tables 5 and 6 share a typesetting file in RQ2. | [Measured fault densities](../rq1/fault_density_full.csv) · [Fault calibration](../rq1/miscompile_calib.csv) |
| RQ1 | [Figure 2a](../rq1/rq1_representative_4.pdf) (`fig:rq1_scatter`), radii across four settings. [Figure 2b](../rq1/selfm_scatter.pdf) (`fig:selfm`), reference-model radii. | [Scatter records](../rq1/rq1_scatter_pairs.csv) · [Reference-model comparisons](../rq1/selfm_compare.csv) |
| [RQ2 · Sampling cost and conclusiveness](../rq2) | [Table 4](../rq2/paper_tables/rq2_query_number.tex) (`tab:rq2_query`), per-task sample counts. [Table 6](../rq2/paper_tables/pair_misc_base50.tex) (`tab:rq2_base50`), paired comparison. [Table 7](../rq2/paper_tables/seed_stability.tex) (`tab:seeds`), seed stability. | [Paired replay](../rq2/rev2_paired.csv) · [Single-test variants](../rq2/single_test_paired.csv) · [Timing](../rq2/rev2_timing.csv) · [Seed stability](../rq2/seed_stability.csv) · [Calibration](../rq2/calibration_check.csv) |
| [RQ3 · Robustness across configurations](../rq3) | [Figure 3a](../rq3/quant_persetting.pdf) (`fig:quant`), simulated quantization. [Figure 3b](../rq3/int8_real_scatter.pdf) (`fig:int8real`), TensorRT INT8 engines. | [Simulated pairs](../rq3/quant_pairs.csv) · [TensorRT pairs](../rq3/int8_real_pairs.csv) · [INT8 summary](../rq3/int8_real_summary.csv) |
| RQ3 | [Figure 4a](../rq3/xhw_persetting.pdf) (`fig:xhw`), GPU comparison. [Figure 4b](../rq3/linf_persetting.pdf) (`fig:linf`), L-infinity perturbations. | [GPU pairs](../rq3/xhw_pairs.csv) · [L-infinity pairs](../rq3/linf_pairs.csv) |
| [RQ4 · Parameter sensitivity](../rq4) | [Table 8](../rq4/paper_tables/param_study.tex) (`tab:param`), clean-label radius searches in eight parameter configurations | [Measurements](../rq4/param_study.csv) · [Table generator](../rq4/make_param_tex.py) |
| [RQ5 · Other distributions](../rq5) | [Table 9](../rq5/paper_tables/patch.tex) (`tab:patch`), fixed-support 3 × 3 patch on LeNet-5/MNIST | [Summary](../rq5/patch3_summary.csv) · [Supplemental scatter plot](../rq5/patch3_scatter.png) |

## Reading the exports

The final RQ1 table contains 24 qualified image tasks and three text tasks. The earlier image-sweep snapshot records 16 qualified tasks under its historical qualification rule. The RQ1 table generator rebuilds that snapshot. RQ2 and RQ4 also retain their earlier typeset exports beside the recorded CSVs. The files in each `paper_tables/` directory contain the final evaluation tables. [Final RQ1 table](../rq1/paper_tables/rq1_all24.tex) · [Earlier qualification list](../rq1/qualified_tasks.txt) · [Snapshot generator](../rq1/make_all24_tex.py)

The final PDF panels are in RQ1 and RQ3. Earlier renderings of the representative RQ1 scatter plot and two RQ3 quantization plots are available as separate PDF and PNG exports. [Alternate figure exports](../archive/figure_exports)

## Supporting records

The radius-search example illustrates the method. Oracle stability and input clipping characterize the experimental setup. The budget comparison measures the cost of fixed and split per-probe error budgets. [Probe sequence](../rq1/running_example.csv) · [Oracle checks](../rq1/oracle_checks.csv) · [Input clipping](../rq1/clip_domain.csv) · [Budget comparison](../rq2/dsplit_compare.csv)

Two RQ1 compiler case studies report observed prediction agreement and local verdicts. The archive groups exploratory plots and tables by sweep and qualification rule. [Interpolation study](../rq1/realbug_stn.csv) · [Window-attention agreement](../rq1/trt_winattn.json) · [Local outcomes](../rq1/realbug_winattn_buggy.json) · [Control outcomes](../rq1/realbug_winattn_correct.json) · [Exploratory outputs](../archive/artifacts)
