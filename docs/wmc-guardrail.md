# wmc-guardrail v3.2

`CompiledGuardrail` attaches independent variable weights to a compiled CNF support graph. Hard constraints describe which assignments are permitted; they do not establish that those constraints capture every real-world safety requirement.

```python
from fractions import Fraction
import numpy as np
from alg_sharpsat import (
    parse_dimacs, compile_guardrail, wmc_sharp_sat,
    exact_marginals, guardrail_project,
)

formula = parse_dimacs("p cnf 3 2\n1 2 0\n-2 3 0\n")
compiled = compile_guardrail(formula, num_groups=2, max_width=3)
p = [Fraction(1, 5), Fraction(7, 10), Fraction(2, 5)]
result = compiled.evaluate(p, mode="exact_rational")
assert result.partition_function == Fraction(17, 50)
assert result.posterior_marginals == [Fraction(29, 85), Fraction(14, 17), Fraction(76, 85)]
action, prior_weight = compiled.map_inference(p)
assert action == (0, 1, 1)
assert prior_weight == Fraction(28, 125)
samples = compiled.sample_conditional(p, num_samples=1000, seed=7)
assert samples.shape == (1000, 3)

batch = np.random.default_rng(7).uniform(.05, .95, (256, 3))
batched = compiled.evaluate(batch, mode="float64")
assert batched.posterior_marginals.shape == (256, 3)
assert wmc_sharp_sat(formula, [[1, 1]] * 3) == 4
projected = guardrail_project(formula, p, mode="exact_rational")
assert projected.map_assignment == action
```

## Inputs and results

- `compile_guardrail(cnf, num_groups=2, semi_join_pushdown=True, max_width=None, max_degree=None, ...)` builds structure on CPU. Optional width reduction and degree splitting preserve unique extensions. Sparse resource options such as `max_local_assignments`, `max_join_pairs`, `max_memory_bytes`, and `max_work` apply; guards raise `ResourceLimitError`, never invoke a reference solver.
- `compiled.evaluate(probs, mode=...)` accepts `(n,)` probabilities or floating batches `(K,n)`. Use `weights=` for explicit `(n,2)` or `(K,n,2)` false/true weights. This distinguishes a two-variable probability batch from a weight matrix. Exact rational evaluation accepts one vector per call.
- `WMCResult` exposes `partition_function`, `posterior_marginals`, `log_partition`, `acceptance_prob`, and `kl_divergence_nats`. It also unpacks as `(Z, marginals)`.
- `wmc_sharp_sat(cnf, weights)` returns the partition function and defaults to rational arithmetic. `exact_marginals(cnf, probs, mode="float64")` returns `WMCResult`; select `mode="exact_rational"` for rational values. The compiled convenience method `compiled.exact_marginals(probs)` defaults to rational arithmetic.
- `guardrail_project` returns `GuardrailResult`, adding `map_assignment` and `map_prob`. It compiles once, performs sum-product inference, then a max-product traversal. Projection and MAP accept one vector; use `evaluate` for batches.

All user weights apply **once, only to the original variables**. Original variables reused as the first copy during splitting keep their weights. Fresh auxiliary copies, OR-gate variables, and incoming proxies have weights `(1,1)`. A free original variable contributes its weight sum and retains its normalized prior marginal.

For general weights, `partition_function` is unnormalized WMC. The normalized prior has mass `total = product(w_i(0)+w_i(1))`, so `acceptance_prob = Z/total` and KL is `-log(Z/total)`. With Bernoulli probabilities, `total=1`. `map_prob` is the original unnormalized assignment weight (the prior joint probability for Bernoulli inputs), not its posterior probability.

If mass is truly zero, evaluation returns `Z=0`, `log_partition=-inf`, and undefined marginals (`None` in rational mode, `NaN` in floating mode); conditioning and KL are undefined. A zero total prior also makes acceptance undefined. MAP and sampling raise `ValueError("Formula is unsatisfiable or has zero probability mass")`. Floating underflow of a positive Z is different: `log_partition`, KL, and posterior inference remain available.

## One compiled tree, all marginals

Compilation retains v3.1's owned-variable frontier solving and unweighted source-support semi-joins. It caches each local row's owned original bits and joint boundary key. A binary sparse join plan retains a shared master until its last incident group, including hyperedges across three or more groups. A backward structural pass removes keys and rows that cannot reach the root.

The upward pass sums weighted local rows by key and contracts the tree. The reverse pass propagates each output's cavity weight to both children, multiplied by the other child's upward value. Each leaf then weights its rows by that cavity and obtains all owned-variable moments together. There are no per-variable recompilations or `n+1` solver calls, and no global satisfying-assignment table. Exact modular inference runs this same upward/downward pass for each required prime.

`compiled.raw_sparse_stats` records frontier construction; `compiled.stats` also reports compile seconds, phase-1 counts, master supports, and retained assignment counts. `compiled.leaves` and `compiled.nodes` expose the compact plan for inspection; treat them as immutable. Device index/assignment tensors are cached on first use. After rational evaluation, `compiled.last_evaluation` reports the integer bound, arithmetic backend, primes, and pass count.

## Numerical guarantees and devices

**`exact_rational`:** accepts `Fraction`, `Decimal`, integers, and `(numerator, denominator)` rational scalars. Decimal representations of ordinary float inputs are converted to fractions; provide `Fraction` for explicit semantics. Each variable's weights share an integer scale. A proven nonnegative intermediate bound selects signed-int64 operations or enough 30-bit CRT primes to exceed that bound; reductions are chunked to avoid overflow. Z and marginals are exact Fractions. Logarithms and KL remain floating approximations, since logarithms are generally irrational.

**`float64`:** CPU batched inference uses stable per-key log-sum-exp for upward and cavity messages. Posterior row matrix multiplication computes all moments; unusually tiny row masses use a log-domain marginal fallback to retain representable subnormal results. Z itself may underflow to zero or overflow to infinity; use `log_partition`. This is finite-precision inference, not rational exactness. Queries may be chunked to fit the configured workspace.

**`float32`:** batched CPU, MPS, or CUDA evaluation uses the same stabilized algorithm at float32 precision. Use `mode="float32", device="mps"` or the `mode="mps"` alias. `device="auto"` probes CUDA, MPS, then CPU; explicit unsupported requests fail. These probes check floating scatter/reduction and matrix operations independently of v1's modular int64 requirements. Float64 requests require CPU. Inputs are prepared on the host, so accelerator timings include transfers; native hardware support and a speedup are not assumed.

MAP defaults to exact integer max-product and deterministic first-choice tie breaking. Exact sampling uses `randrange(total_integer_weight)` and cached conditional cumulative sums, followed by tree backtracking and independent free-variable draws. It has no rejection stage. Floating MAP/sampling modes instead use CPU log weights and finite-precision categorical arithmetic. MAP and sampling have message, traversal, and output resource guards.

## Reproduction and scope

```sh
python -m pytest -q --junitxml=verification-v3.2.xml
python -m alg_sharpsat.guardrail_benchmarks --sizes 18 22 26 30 \
  --batch-size 256 --repeats 3 --json benchmark-v3.2-guardrail.json
```

The benchmark compares the existing seed-0 planted coupled/bipartite fixtures against an independent recursive weighted DPLL that uses `n+1` counting passes. It validates distinct batch vectors against single evaluations and selected vectors against DPLL. Timings separate compilation, warm inference, fresh compile-plus-inference, and batch throughput; CPU one/four-thread and available accelerator results are reported separately.

The rare-event showcase is a deliberately simple 20-unit-clause formula with uniform priors and acceptance `2^-20`. It isolates rejection overhead; it is not evidence that every rare-event constraint compiles cheaply. The report bounds rejection simulation to 10,000 proposals and records expected work analytically. Sparse compilation and join width can still grow exponentially. This release does not benchmark a neural model or claim semantic safety beyond the supplied CNF.
