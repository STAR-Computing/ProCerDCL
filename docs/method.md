# Method and implementation

## Decision contract

Let `p` be the probability that a binary oracle returns one on a draw from a specified distribution. The main sequential test takes a threshold `theta`, a positive separation margin `eta` and an error budget `delta`. The guarantee assumes independent draws and a stateless oracle.

| Verdict | Interpretation |
|---|---|
| `Yes` | The evidence supports `p <= theta`. |
| `No` | The evidence supports `p > theta + eta`. |
| `None` | The draw budget ended before either stopping rule fired. |

For the main test, the probability of each kind of wrong decisive verdict is bounded by `delta`. In the separation band `theta < p <= theta + eta`, either decisive assertion would be wrong. The comparison methods include fixed-sample and forced-decision variants with different decision contracts. [Sequential tests](../src/verification/ProCerDCL/demo_alg.py)

## Properties

Write `M` for the reference model, `C` for the compiled model and `y0` for the reference prediction at the seed. Each draw evaluates the property at a sampled input `x`. [Property samplers](../src/verification/PROVERO/samplers.py)

| Property | A draw counts as a violation when |
|---|---|
| Clean-label robustness | `C(x) != y0` |
| Pointwise disagreement | `C(x) != M(x)` |
| Reference-stable disagreement | `M(x) == y0` and `C(x) != y0` |

The reference-stable property measures the probability of a joint event across all draws. The samplers support uniform perturbations in an L2 ball or an L-infinity box. A support mask restricts the set of varying coordinates.

## Verifier IDs

| IDs | Method |
|---|---|
| 9, 10 | Main single test with repeated-check-corrected Clopper–Pearson bounds or an anytime-valid confidence sequence |
| 0, 6 | Interval-family tests with draw reuse and the corresponding stopping rules |
| 7, 8 | Interval-family tests with a fixed-sample decision at the draw cap |
| 4, 5 | Interval-family ablations for draw reuse and error allocation |
| 11, 12 | Single-test and interval-family variants with geometric checkpoints |
| 1, 2, 3 | PROVERO, its fixed-sample tester and a truncated sequential probability ratio test |

The registry maps each ID to a name and a runner. The cache gives each verifier a separate cursor on the same sequence of oracle draws, which makes per-input sample counts comparable. [Registry](../verifier.py) · [Draw cache](../src/verification/trace.py)

## Clean-label radius

`find_max_yes_epsilon` takes two models and a seed input with a batch dimension. It evaluates radii by exponential bracketing and bisection and returns the largest tested radius with a `Yes` verdict. Interpreting this radius as a boundary assumes that the violation probability is nondecreasing with radius. The result records the radius, verdict and draw count for each probe. [Radius search](../verify_bound.py)

The historical `fixed` mode spends the supplied error budget at each probe. With `probe_delta_mode="split"`, the function divides that budget by an upper bound on the number of radius probes. The search resolution is set by `epsilon_step`. [Worked probe record](../rq1/running_example.csv)
