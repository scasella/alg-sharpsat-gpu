# wmc-guardrail and Guardrail Studio v3.3

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

## V3.3: interactive Guardrail Studio

The Studio adds named variables, a restricted Boolean language, exact policy diffs,
and a local browser interface. It uses the existing compiled sparse weighted
engine. There is no Node build, CDN, additional web framework, or external model
call. HTML, CSS, and JavaScript are packaged in the wheel.

```sh
python -m pip install -e .
wmc-guardrail-studio
# Equivalent: python -m alg_sharpsat.webapp
wmc-guardrail-studio --host 127.0.0.1 --port 8765 --no-browser
wmc-guardrail-studio --demo-cli
```

Open `http://127.0.0.1:8765`. Select a domain and preset, adjust the prior sliders,
and inspect Z, KL in bits, the threshold action, every violated rule, MAP changes,
and prior/posterior bars. Enable **Exact rational (Fraction)** to display exact
numerators and denominators. Disable, edit, or add a rule to see states and prior
probability mass exposed/revoked relative to the original built-in policy. Restore
rules to return to the starting policy. Resampling produces five independent
conditional draws; repeats are legitimate.

The three packs are **editable engineering fixtures**: cloud operations (20
variables / 16 rules), clinical prescribing (22 / 19), and fintech treasury
(20 / 16). Each has routine, conflict, and rare-event presets. The cloud presets
have acceptance probabilities approximately **0.9321**, **0.01051**, and
**10^-24**. The clinical examples deliberately omit dosing, patient history,
exceptions, and professional judgment; the combined pregnancy/lactation flag is
a simplification. They are not validated clinical protocols. Treasury examples
are not a compliance certification. Primary clinical references and explicit
simplifications are documented in [`domains.py`](../src/alg_sharpsat/domains.py).

All variables, including context and control flags, are inferred jointly.
**Set observed facts to probability 0 or 1 to clamp them.** Otherwise posterior
conditioning and MAP can change their values. Setting an “approval” flag in a
model does not obtain an approval or perform a real safeguard. The Studio only
inspects Boolean assignments; it executes no operational, clinical, or financial
actions. MAP is the highest-prior-probability compliant tuple, not a minimum
Hamming-distance repair. Its displayed probability is the prior joint weight,
not its conditional probability.

### Named policies in Python

```python
from fractions import Fraction
from alg_sharpsat import VariableSpec, RuleSpec, PolicySpec, diff_policies

variables = [
    VariableSpec("deploy", "Deploy the proposed build", category="action"),
    VariableSpec("tested", "Integration tests passed", category="evidence"),
    VariableSpec("approved", "Peer approved this change", category="evidence"),
]
policy = PolicySpec(variables, [
    RuleSpec("release_gate", "Deploy only with tests and approval",
             "deploy -> (tested and approved)"),
])
compiled = policy.compile_guardrail()
probs = {"deploy": Fraction(4, 5), "tested": Fraction(3, 5),
         "approved": Fraction(2, 5)}
result = compiled.evaluate(probs, mode="exact_rational")
assert compiled.cnf.num_vars == 3  # This common form needs no auxiliaries.
assert compiled.model_count == 5
assert result.partition_function == Fraction(49, 125)
assert [r.rule_id for r in policy.check_violations(
    {"deploy": 1, "tested": 1, "approved": 0})] == ["release_gate"]
action, probability = compiled.map_inference(probs)
assert action == (1, 1, 1)
assert probability == Fraction(24, 125)
samples = compiled.sample_conditional(probs, num_samples=5, seed=7)

relaxed = PolicySpec(variables, [
    RuleSpec("release_gate", "Deploy only with tests", "deploy -> tested"),
])
diff = diff_policies(policy, relaxed, probs)
assert (diff.old_allowed_count, diff.new_allowed_count) == (5, 6)
assert diff.newly_exposed_count == 1
assert diff.newly_exposed_prob == Fraction(36, 125)
assert diff.revoked_count == 0
```

Names are case-sensitive identifiers, not Python code. No `eval`, attribute
access, imports, indexing, or arbitrary function calls are accepted. Missing
variables, duplicate IDs, unknown names, and invalid binary assignments raise
errors. Rule diagnostics return the enabled `RuleSpec` objects that failed.

| Syntax | Meaning |
|---|---|
| `not a`, `~a`, `!a` | Negation |
| `a and b`, `a & b` | Conjunction |
| `a or b`, `a \| b` | Disjunction |
| `a -> b`, `implies(a, b)` | Implication |
| `a <-> b`, `iff(a, b)` | Equivalence |
| `xor(a, b)` | Exactly one of two expressions |
| `at_most_one(a, b, ...)` | Pairwise exclusion |
| `at_least_one(a, b, ...)` | One or more true |
| `exactly_one(a, b, ...)` | One and only one true |
| `forbid(a, b, ...)` | `not (a and b and ...)` |
| `require(a, b, ...)` | `a and b and ...` |

Precedence, highest first: negation, `and`, `xor`, `or`, implication,
equivalence. Implication associates right. Parentheses are recommended for
mixed operators. `require()` and `at_most_one()` are true; `forbid()`,
`at_least_one()`, and `exactly_one()` are false. Empty helper semantics follow
empty conjunction/disjunction identities.

The compiler emits direct clauses for bounded expansions of simple expressions,
including `a -> (b and c)` and pairwise exclusions. More complex expressions use
**full deterministic equivalences** for Tseitin auxiliaries. Every named
assignment has a unique extension, so auxiliary weights must be `(1,1)`.
`CompiledPolicy` applies original Bernoulli weights once, keeps auxiliaries
neutral, exposes only named marginals/actions/samples, and reports acceptance
as Z. Do not pass probabilities of 1/2 for auxiliaries to a raw CNF evaluator.

`CompiledPolicy.evaluate` accepts named dictionaries, ordered vectors, or numeric
NumPy/Torch batches in floating modes. Numeric floating batches bypass rational
conversion. Rational mode takes one vector, with `Fraction`, `Decimal`, or
rational strings; nested sequences denote batches, not numerator/denominator
pairs. `.compiled` exposes the underlying `CompiledGuardrail`, with its normal
resource options available at compilation.

Policy diffs compile old, new, and their intersection through the sparse engine.
They calculate `new - intersection` and `old - intersection`, for counts and
exact rational masses. Rules are compiled together with fresh auxiliary IDs;
concatenating separately allocated auxiliary CNFs would be incorrect. Policies
may order the same names differently; probabilities are aligned by name. Without
probabilities the Python diff returns counts and `None` mass fields.

### Local REST API

`GET /api/domains` returns all variable/rule definitions and all nine named
presets. `GET /` returns the self-contained interface. Requests and responses
are JSON; custom policies use `variables` and `rules` instead of `domain`.
A named probability dictionary must contain exactly the policy's names.

```sh
curl -s http://127.0.0.1:8765/api/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"variables":[{"name":"deploy","description":"Deploy"},
                    {"name":"approved","description":"Approval exists"}],
       "rules":[{"rule_id":"approval","description":"Approval required",
                 "expr":"deploy -> approved","enabled":true}],
       "probs":{"deploy":"4/5","approved":"2/5"},
       "mode":"exact_rational","num_samples":5,"seed":7}'
```

| Endpoint | Inputs and output |
|---|---|
| `POST /api/evaluate` | `domain` or custom variables/rules; `probs`; mode `float64` or `exact_rational`; optional `num_samples` (0–100), `seed`. Returns Z, log Z, KL, named marginals, threshold violations, exact MAP, repaired flags, samples, and pruning telemetry. |
| `POST /api/policy-diff` | `domain` or custom variables; `base_rules`, `modified_rules`, `probs`. Returns old/new/intersection/exposed/revoked counts and masses, with exact rational mass strings. |
| `POST /api/batch-benchmark` | Current policy and priors. Runs 256 distinct deterministic perturbations through the cached CPU float64 graph; returns median of three batches and vectors/s. |

`exact_rational_z`, `exact_rational_marginals`, `exact_rational_map_prob`, and
sample rational probability fields use `numerator/denominator` strings. Numeric
JSON fields approximate those values. MAP and sampling use exact integer
arithmetic even when posterior display uses float64. Sample `probability` is
its prior joint mass; `conditional_probability` is that mass divided by Z.
With true zero mass, the API returns `null` marginals/MAP/KL and empty samples,
plus a diagnostic. Floating underflow is distinguished by a finite
`log_acceptance_prob`. No NaN or Infinity tokens are emitted in JSON.

`eval_ms` measures posterior evaluation only. `map_ms`, `sample_ms`, and
`total_ms` report the additional work; HTTP transport/rendering is outside those
timers. `compile_ms` includes first compilation and the exact model-count query,
and is zero on cache hits. `compile_historical_ms` retains the original cost.
The LRU caches eight compiled supports, keyed by canonical clauses, CNF variable
count, and named-original count. Changed descriptions or priors reuse a plan;
changed policy logic uses the corresponding plan. Engine operations are serialized
to avoid races in mutable numerical caches; the HTTP server uses bounded threads.

Pruning reports **per-group local rows**, at owned phase 1, after proxy/semi-join
construction, and after backward structural pruning. Proxy expansion can make
semi-join rows exceed phase-1 rows. These numbers are not directly comparable
to `2**n` global Boolean states. The UI computes the actual semi-join-to-backward
reduction; it does not transplant the v3.2 benchmark's 99.85% cumulative key
reduction onto unrelated domain fixtures.

The server defaults to loopback and is intended for local use. It has no user
authentication or TLS. Origin/Host validation, a 128 KiB JSON limit, parser
bounds, probability-size checks, and sparse-work limits reject oversized requests.
The web API supports up to 24 named variables and 64 rules; complex policies can
still hit its compilation budget. Avoid exposing the unauthenticated development
server to untrusted networks. Set `GUARDRAIL_TORCH_THREADS=4` before starting to
compare CPU threading; the default is one thread. The Studio's batch button
measures CPU inference, not MPS. The v3.2 benchmark CLI remains available for
native accelerator comparisons.

### Verification

`pytest -m g16` exercises DSL truth tables, counting-preserving auxiliary
extensions, exact weighted marginals, policy intersection diffs, all nine domain
MAP scenarios, planned/emergency database authorization, and the clinical fixture
contracts. `pytest -m g17` starts an ephemeral loopback server and exercises every
endpoint, exact and floating output, caching, zero mass, subnormal acceptance,
request validation, and the terminal demo. The full suite also retains all
pre-Studio engine regressions.
