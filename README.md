# alg-sharpsat-gpu v3.4

An exact Python/PyTorch prototype of the supplied cut-and-interpolate #SAT
construction. It implements counting-preserving CNF transformations, local
integer measurements, finite-field boundary factors, Lagrange and Hermite interpolation,
and arbitrary-precision CRT reconstruction. V3 adds sparse boundary contraction
and an executable miniature SWAP–Möbius reference. CUDA, MPS, and CPU selection
is subject to an actual arithmetic capability check. V3.1 filters incoming
proxy tuples using source groups’ feasible master assignments. V3.2 adds
`wmc-guardrail`: compiled weighted counting, posterior marginals, MAP, and
rejection-free conditional sampling. V3.3 adds a local interactive Studio,
named policy DSL, and exact policy blast-radius auditing. V3.4 adds live
Codex Luna scenario extraction and evidence locks across inference.

**Start here:** [ELI5 explanation and four runnable use cases](docs/eli5-and-use-cases.md).
For integration, read the [API and Studio guide](docs/wmc-guardrail.md).
Coding agents should start with [AGENTS.md](AGENTS.md) ([Claude entry point](CLAUDE.md)).

> **V3.4 interception:** Describe a scenario, extract candidate facts with the
> authenticated local `codex exec -m gpt-6-luna` CLI, and compute an evidence-clamped
> MAP action. [Live interception guide](docs/wmc-guardrail.md#v34-live-interception-and-evidence-locks).
>
> **V3.3 Studio:** Run `wmc-guardrail-studio` for named rules, live posterior
> calibration, MAP repairs, and exact policy diffs. [Studio guide](docs/wmc-guardrail.md#v33-interactive-guardrail-studio).
>
> **V3.2:** [wmc-guardrail API](docs/wmc-guardrail.md) adds exact rational
> inference and batched floating-point posterior projection on the sparse graph.
>
> **Summary of Findings & Documentation**
>
> Read the [technical blog post](docs/blog-post.md) for the executive findings
> and [source research notes](docs/v3-research.md) for the mathematical audit.
>
> - **v1 Lagrange:** exact static-partition interpolation suppresses fewer than
>   one boundary link per node on average.
> - **v2 Hermite-Jet:** exact Hasse jets retain a static-partition degree barrier;
>   the recorded work is 1.2–3.28× the v1 signature/subset states.
> - **v3.1 Sparse Semi-Join Contraction:** owned frontiers and unweighted master
>   supports reduce bipartite `n=30` peak joint rows by 90.9%. Recorded CPU time
>   is 6.858 ms versus the reference DPLL's 9.840 ms; `n=26` is about 1,839×
>   faster than v1 interpolation. These are fixture-specific results.
> - **stage4 Executable SWAP–Möbius Reference:** exact miniature locality,
>   spectral, Möbius, and Hermite checks expose the finite-size overhead; they
>   do not establish the full manuscript's asymptotic theorem.
> - **v3.2–v3.4 Guardrail Studio:** compile-once weighted inference, named rules,
>   exact policy diffs, and live scenario extraction with evidence locks. The
>   recorded bipartite funnel removes 99.8557% of Group 0 joint keys cumulatively;
>   the historical CPU batch reaches 476,981 vectors/s. Live posterior inference
>   takes 0.45–0.56 ms in two recorded examples, with separate MAP, compilation,
>   and 11.6–14.0 s model-call costs. See the [measured results](docs/blog-post.md).

This is an experimental micro-engine. The supplied suppression schedule does
**not** establish subexponential running time or a practical speedup over DPLL.
The benchmark compares engines and verifies every completed count against DPLL.

## Install and run

```sh
cd alg-sharpsat-gpu
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest
python -m alg_sharpsat.benchmarks
```

Python 3.11 or later is required. Dependencies are declared in `pyproject.toml`.
The current release tag is `v3.4.0`; the Python distribution version is `0.3.4`.
The original v3.1 release is retained as tag `v3.1.0`.
The [v3.1 release verification report](verification-release.xml) records 437 passed
tests and six skips for unavailable CUDA/MPS hardware. The GitHub Actions
workflow runs the suite and a CLI smoke test on Python 3.11 and 3.12.
XML reports retain test results and timings; machine-specific repository paths
have been replaced with relative paths for publication.
The workspace's `.venv` was created with `--system-site-packages` to reuse the
already installed dependencies; a fresh installation can use the commands above.

### Documentation and release checks

```sh
python scripts/check_repository.py --check-snippets
python -m pytest -q
# Optional authenticated model check; sends a synthetic scenario through Codex:
WMC_LIVE_CODEX=1 python -m pytest -q -m g18
```

The repository checker validates local Markdown files and heading links, runs
the four standalone use-case recipes on request, parses JSON/XML artifacts,
and checks common credential patterns, machine-specific home paths, and ignored
cache directories. External URLs and Git history are outside this check.
GitHub Actions runs the offline suite, documentation audit, and CLI smoke tests
on Python 3.11 and 3.12. Live extraction requires a separately authenticated
`codex` CLI; ordinary CI uses deterministic extraction and subprocess mocks.

The final v3.4 release recheck recorded **643 passed, 8 skipped offline**
([report](verification-v3.4-release.xml)), then **63 passed** with authenticated
g18 enabled ([report](verification-v3.4-live.xml)). Together they validate
**644 distinct passing tests**: the live test replaces one offline skip, while
seven CUDA/exact-integer MPS checks remain unavailable. Native MPS floating
inference passed. The original [full live-enabled report](verification-v3.4.xml)
is retained alongside these fresh results.

```python
from alg_sharpsat import parse_dimacs, gpu_algebraic_sharp_sat

formula = parse_dimacs("p cnf 3 2\n1 2 0\n-2 3 0\n")
assert gpu_algebraic_sharp_sat(formula, device="cpu") == 4
```

### Native CUDA and Apple Silicon runs

Use a native terminal outside restricted execution environments with a
CUDA-enabled PyTorch installation, or native arm64 Python/PyTorch on Apple
Silicon. The installed `alg-sharpsat-bench` command is equivalent to
`python -m alg_sharpsat.benchmarks`:

```sh
alg-sharpsat-bench --engine v3 --family coupled --sizes 18 --repeats 3 --device cuda --torch-threads 1
alg-sharpsat-bench --engine v3 --family coupled --sizes 18 --repeats 3 --device mps --torch-threads 1
```

Acceleration requires the device to pass the exact integer-arithmetic probes;
MPS kernel support varies. An explicit unsupported device raises an error.
Use `--device auto` to try CUDA, then MPS, then CPU, with fallback reasons in
telemetry. `--torch-threads` controls CPU PyTorch threads, not GPU concurrency.
Compare native measurements with the recorded CPU baseline; GPU availability
alone does not guarantee a speedup on these small instances.

V2 adds an exact Taylor/Hasse jet engine while retaining the v1 API:

```python
from alg_sharpsat import jet_sharp_sat

assert jet_sharp_sat(formula, device="cpu", num_nodes=2) == 4
assert jet_sharp_sat(formula, device="cpu", mode="single_node_taylor") == 4
```

V3 provides a practical sparse counter and a bounded mathematical reference:

```python
from alg_sharpsat import contract_sharp_sat, auto_sharp_sat, swap_mobius_count

assert contract_sharp_sat(formula, device="cpu", num_groups=2) == 4
assert contract_sharp_sat(formula, device="cpu", num_groups=4) == 4
assert auto_sharp_sat(formula, device="cpu") == 4
assert swap_mobius_count(formula, d=2, L=2, b=1) == 4
```

`auto_sharp_sat` selects sparse contraction for `B > 4` to avoid static
interpolation overhead; it also uses direct sparse contraction for smaller
boundaries. The v1 and v2 entry points retain their explicit interpolation paths.

`Literal.var` is zero-based (`0..num_vars-1`). DIMACS uses its standard signed,
one-based identifiers. Declared unused variables contribute a factor of two.
The empty formula has `2**num_vars` models; an empty clause has none.

## V3.4: live interception with evidence locks

The Studio now accepts natural-language scenarios through **`gpt-6-luna`** using
an authenticated `codex` CLI on `PATH`. It extracts soft variable probabilities
and candidate evidence with literal source quotes. WMC, exact MAP, and sampling
run locally against the existing cached policy. The model receives the scenario,
variable descriptions, and active rules; it has no operational execution role.

```sh
python -m pip install -e .
wmc-guardrail-studio
wmc-guardrail-studio --domain cloud_ops_agent --intercept \
  "Sev-1 outage in production. Drop the table now. MFA is verified; peer review is not done."
# Explicit offline mode; never used silently after a live failure:
wmc-guardrail-studio --domain cloud_ops_agent --backend deterministic \
  --intercept "prod_environment=true; incident_sev1_active=true; run_schema_migration=true"
```

Each variable has an **evidence**, **action**, or **safeguard** role. A locked
observation replaces its prior with exactly 0 or 1. Posterior inference, raw
thresholding, MAP, samples, policy-diff masses, and batch queries preserve it.
The Studio offers nine text examples, per-variable locks, source quotes, and
separate extraction/inference timings. Existing locks survive new interceptions;
conflicting new evidence is rejected until the user resolves the conflict.

`repaired_variables` contains action/safeguard changes. Unlocked evidence can
still be inferred differently by MAP and is reported separately in
`inferred_context_changes`; this does not change a known fact. Zero-mass
constraints yield a diagnostic and no MAP or samples.

Live model probabilities are **uncalibrated estimates**, and a quoted source
does not prove its interpretation or truth. Review extracted facts and lock
known observations. Exactness applies to the supplied probabilities, evidence,
and Boolean rules. The illustrative domain policies do not establish operational,
clinical, or financial safety.

Validation: **644 passed, 7 skipped**, with the authenticated live test enabled.
The seven skips cover unavailable CUDA and unsupported MPS exact integer kernels.
[Verification report](verification-v3.4.xml) · [Recorded live runs](interception-v3.4.json).

| Live scenario | Locked facts | Raw rule violations | Extraction | Exact posterior | Exact MAP | Conditional Z |
|---|---:|---:|---:|---:|---:|---:|
| Cloud outage | 8 | 5 | 11.59 s | 0.56 ms | 5.94 ms | 1.09375e-07 |
| Clinical fixture | 7 | 11 | 14.02 s | 0.45 ms | 3.53 ms | 1.2005e-05 |

Both recorded MAP tuples satisfy every active rule and preserve every clamp.
These are single observations on 2026-10-27, with other tests running concurrently,
not latency medians or calibrated risk estimates. First-query compilation was
about 89 ms per policy; it is excluded from the posterior and MAP columns.
The Codex model call dominates end-to-end latency.


## V3.3: wmc-guardrail-studio — named policies and live decisions

Validation: **581 passed, 7 skipped**, including 93 new Studio tests. Native
MPS floating inference passed; CUDA and unsupported MPS exact integer checks
were skipped. [Verification report](verification-v3.3.xml).

```sh
python -m pip install -e .
wmc-guardrail-studio                         # Opens http://127.0.0.1:8765
python -m alg_sharpsat.webapp --no-browser   # Same local server
wmc-guardrail-studio --demo-cli              # Nine-scenario terminal report
```

A self-contained dark-mode dashboard runs on Python's standard-library HTTP
server, with **no extra packages or Node build**. Choose cloud operations,
clinical prescribing, or fintech treasury; change probability sliders; inspect
prior/posterior shifts and all violated rules; view the most probable compliant
assignment; and draw rejection-free conditional samples. Toggle, edit, or add
DSL rules to measure exact newly exposed/revoked state counts and probability
mass. A live benchmark evaluates 256 distinct perturbed vectors on the cached
CPU float64 graph. Exact rational mode shows Fraction results; MAP and sampling
remain exact in both display modes.

| Example pack | Variables / rules | Routine Z | Conflict Z | Rare-event Z |
|---|---:|---:|---:|---:|
| Cloud operations | 20 / 16 | 0.9321 | 0.01051 | 1.00 × 10⁻²⁴ |
| Clinical prescribing fixture | 22 / 19 | 0.9049 | 1.082 × 10⁻⁵ | 2.797 × 10⁻²³ |
| Fintech treasury fixture | 20 / 16 | 0.9226 | 9.801 × 10⁻⁹ | 9.811 × 10⁻¹² |

These are measured acceptance masses of the supplied synthetic presets, not
real-world risk estimates. Context variables are jointly inferred: clamp known
facts to 0 or 1. MAP maximizes prior probability rather than minimizing flips.
The clinical and financial packs are illustrative engineering fixtures, not
validated decision protocols. No external action is executed.

The public `VariableSpec`, `RuleSpec`, `PolicySpec`, and `diff_policies` APIs
support human-readable Boolean expressions, deterministic neutral auxiliaries,
and exact intersection-based policy audits. The REST endpoints are
`/api/domains`, `/api/evaluate`, `/api/policy-diff`, and `/api/batch-benchmark`.
See the [Studio guide and API examples](docs/wmc-guardrail.md#v33-interactive-guardrail-studio)
for syntax, numerical contracts, timing fields, cache limits, and reproduction.

## V3.2: wmc-guardrail — weighted marginals, MAP, and sampling

[API and numerical contracts](docs/wmc-guardrail.md) ·
[benchmark artifact](benchmark-v3.2-guardrail.json) ·
[verification report](verification-v3.2.xml)

Validation: **487 passed, 8 skipped** (unavailable CUDA/MPS hardware),
including 150 rational differential cases, 50 MAP cases, exact sampling,
subnormal probabilities, expandable CRT, and distinct batched inputs.

Compile a hard CNF once, then evaluate independent neural probability vectors
against the cached sparse graph. One upward/downward sum-product traversal
returns the partition function and **all original-variable marginals**.
Max-product backtracking returns a safe MAP assignment; conditional tree
sampling draws valid assignments without rejection. Auxiliary and proxy
weights remain neutral, and unused original variables keep their prior factors.

```python
from fractions import Fraction
from alg_sharpsat import parse_dimacs, compile_guardrail

formula = parse_dimacs("p cnf 3 2\n1 2 0\n-2 3 0\n")
compiled = compile_guardrail(formula, num_groups=2)
p = [Fraction(1, 5), Fraction(7, 10), Fraction(2, 5)]
result = compiled.evaluate(p, mode="exact_rational")
assert result.partition_function == Fraction(17, 50)
assignment, prior_probability = compiled.map_inference(p)
samples = compiled.sample_conditional(p, num_samples=1000, seed=7)
```

`exact_rational` returns exact Fractions using bounded int64 or expandable CRT;
MAP and sampling default to exact integer arithmetic. `float64` runs batched
log-domain inference on CPU; `float32` supports CPU and capability-probed MPS
or CUDA. The floating modes are numerical approximations. `log_partition`
remains available when Z underflows, with a log-domain fallback for unusually
tiny marginals. General weights use normalized acceptance `Z/product(w0+w1)`;
for probabilities, acceptance is Z and intervention KL is `-log(Z)`.
Zero mass has no posterior: MAP/sampling fail explicitly, and evaluation marks
marginals undefined. See the API documentation for shapes and edge cases.

The original CPU-only run used **arm64 macOS, PyTorch 2.10.0**, one CPU
Torch thread, and medians of three runs. MPS and CUDA were unavailable in this
execution environment. The table uses the existing seed-0 planted families;
DPLL here computes WMC plus all marginals using `n+1` passes, so these baseline
times should not be compared with the unweighted DPLL timings below.
Batch timings cover **256 distinct probability vectors**, not repeated copies.

| Family | n | DPLL WMC + marginals ms | Compile + float64 ms | Compile + rational ms | Warm single ms | Batch 256 ms | Vectors/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coupled | 18 | 29.499 | 1.357 | 1.319 | 0.190 | 0.663 | 386,003 |
| Coupled | 22 | 60.013 | 1.792 | 2.187 | 0.196 | 1.197 | 213,838 |
| Coupled | 26 | 107.695 | 1.986 | 1.679 | 0.198 | 0.594 | 431,309 |
| Coupled | 30 | 228.912 | 2.200 | 1.911 | 0.300 | 1.065 | 240,282 |
| Bipartite | 18 | 19.618 | 2.240 | 1.928 | 0.184 | 0.537 | 476,981 |
| Bipartite | 22 | 56.717 | 2.481 | 2.345 | 0.186 | 0.735 | 348,161 |
| Bipartite | 26 | 82.814 | 4.109 | 3.753 | 0.207 | 1.762 | 145,279 |
| Bipartite | 30 | 219.428 | 17.518 | 17.797 | 0.311 | 18.915 | 13,535 |

A subsequent [native MPS run](benchmark-v3.2-native.json) passed float32
partition-function and marginal checks on all eight cases. CPU was faster on
all eight batch comparisons: bipartite n=30 took 20.03 ms on one CPU thread,
8.72 ms on four threads, and 27.57 ms on MPS. The first MPS call took 1.83 s;
this three-sample run includes startup and has no explicit warmup. See
[the blog's Section 6](docs/blog-post.md#section-6-real-time-neuro-symbolic-guardrails-v32)
for the complete native table and the **over 99.85% cumulative key reduction**
(442,784 → 639); backward pruning alone reduces the v3.1 table by 98.41%.

Single compiled queries meet the sub-millisecond target on all eight fixtures;
a complete 256-vector batch does not meet it on every fixture. The report also
contains four-thread CPU measurements. Bipartite n=30 throughput there was
29,271 vectors/s; multithreading was not uniformly faster.
Compilation remains a real cost and can be exponential in local or join width.
Backward structural pruning reduces the n=30 local tables from 51,562/17,198
rows to 831/727 rows, with 639 compatible keys in each group.

For the separate 20-unit-clause rare-event fixture, `Z=2^-20≈9.54e-7`.
Exact MAP took **0.147 ms**, and
**1,000 exact valid samples took 5.466 ms**.
Naive rejection would require 1,048,576,000 proposals on average for 1,000
samples; a bounded 10,000-proposal simulation accepted 0.
This simple fixture isolates rejection overhead, not general compilation cost.
All completed counts and marginals match the independent reference; no neural
model or accelerator speedup is claimed by these CPU results.

```sh
python -m alg_sharpsat.guardrail_benchmarks --sizes 18 22 26 30 \
  --batch-size 256 --repeats 3 --json benchmark-v3.2-guardrail.json
# Equivalent installed command: wmc-guardrail-bench
```

On a native Apple Silicon or CUDA installation, use
`compiled.evaluate(batch, mode="float32", device="mps")` or `device="cuda"`.
The benchmark probes available accelerators automatically and records unsupported
or unavailable devices. The older sparse/interpolation CLI's device checks are
separate from these floating-point probes.

## V3: sparse boundary contraction

`stage3_sparse_contract.py` packs each satisfying local assignment into a key
containing **both outgoing master and incoming proxy bits**. All copies of one
master use the same bit position. Equal keys are grouped with integer counts;
interior variables contribute multiplicity rather than key bits. Declared free
variables are factored out exactly. The CPU implementation uses NumPy for local
frontiers, grouping and sorted-key search, with PyTorch for count reductions
and general sparse joins; accelerator paths use PyTorch tensors throughout.

Two groups contract by sorted-key intersection and a weighted dot product.
For more groups, sparse joins match shared key projections and sum out variables
only after their last incident group is included. This handles a master shared
by three or more groups, which cannot be eliminated after its first pairwise
join. Join ordering favors shared variables and smaller tables. The code never
allocates a dense source-signature by proxy-signature matrix.

Counts use signed 64-bit arithmetic only when an assignment bound proves every
intermediate safe. Otherwise they use modular products, chunked reductions,
and enough CRT primes for exact reconstruction. Keys currently support at most
63 distinct boundary masters. Explicit frontier, join-pair, workspace, and work
budgets raise `ResourceLimitError`; they do not invoke a reference counter.
The workspace bound covers tensor allocations, not the spectral partitioner's
dense adjacency or total process memory.

Balanced original-variable ownership does **not** imply that local tables,
including proxies, have size at most `ceil(n/G)`. Proxy variables and intermediate
join width are real costs and are reported separately. Likewise, v2's incoming
signature counts are marginal counts: they cannot be used as the sizes of the
joint boundary tables without losing master/proxy correlations.

## V3.1: master-first semi-join pushdown

Sparse construction now begins with owned variables: original or transformed
variables assigned to the group, excluding incoming proxies. It evaluates all
clauses containing only owned variables and retains the resulting assignment
frontier. Variables used only by bridge clauses or outgoing links are included;
truly unused variables retain their exact factor-of-two treatment.

Before expanding any group's proxies, the engine projects **every** owned
frontier onto its outgoing masters. For each destination, it projects the
source's support onto exactly the incoming masters referenced there and removes
duplicates. Proxy expansion accepts only tuples supported by these projections,
while applying bridge clauses as their variables become available.

The projections are sets, not weighted tables: multiplying source multiplicity
into a destination would double-count models. Correlations between masters from
one source are preserved, and each incoming source supplies its own restriction.
Projecting onto fewer masters may admit extra assignments; the final joint
contraction still enforces all equalities and counts multiplicities exactly.
Every globally satisfying assignment survives because its source restriction
satisfies those source-internal clauses. All source supports are computed before
filtering, so cyclic group connections require no circular assumptions.

This pushdown is enabled by default in `contract_sharp_sat` and `auto_sharp_sat`.
For a reproducible comparison with the original sparse construction:

```python
from alg_sharpsat.stage3_sparse_contract import sparse_count_with_stats

filtered, filtered_stats = sparse_count_with_stats(formula, device="cpu")
original, original_stats = sparse_count_with_stats(
    formula, device="cpu", semi_join_pushdown=False)
assert filtered == original
```

Telemetry distinguishes `phase1_assignment_counts`, distinct
`feasible_master_mask_counts`, per-group `peak_frontier_rows` (accepted
assignment rows between expansion steps), `peak_temporary_rows` (largest
candidate batch), and `peak_joint_rows` (largest compressed initial or
intermediate factor). These are row counts, not process-memory measurements.
Filtering is a necessary-condition reduction; source bridge constraints can
still remove additional rows during the final contraction.

## V3: executable SWAP–Möbius reference

`stage4_swap_mobius.py` constructs `M = U f(W)` exactly over rational numbers.
It builds the full orthogonal tensor basis from symmetric and antisymmetric
vectors with entries in `{0, 1, -1}`. Rows whose eigenvalue is not one form `C`;
all exceptional directions are retained. If `v_i` is an exceptional basis row,
then

```text
(I-M)z = sum_i ((1-lambda_i) / ||v_i||²) * v_i * (v_i · z).
K_omega(T,z) = z_omega + sum_eta M[omega,eta] * (phi[omega,eta](T)-1) * z_eta.
```

The finite tests verify the exact rank, support condition, measurement recovery,
`K(0,z)=z`, degree at most `b`, and zero residual outside the assigned group at
**every** node `1..L`. Matrices have at most 256 coordinates, covering all
requested `d=2`, `L=2,3,4`, `1<=b<L` configurations. These small parameters do
not satisfy the exposition's asymptotic concentration assumptions; the finite
identities are checked directly.

The default `swap_mobius_count` uses `OccurrenceSwapCounter`. Original variable
`i` receives a first-register word `alpha_i`; clause `c` receives a second-register
word `beta_c`. Its literal occurrences occupy `(alpha_i, beta_c)`. Conflicting
cells are packed into additional SWAP blocks; repeated occurrences always read
the same original variable. Unoccupied cells are fixed to zero. At node `t`,
variables partition by `alpha_i[t]` and the entire clause belongs to group
`beta_c[t]`. Thus clauses remain intact throughout evaluation. This miniature
uses deterministic cyclic words, rather than the source's asymptotic balanced
design construction.

Block measurements are expanded into original-variable coordinates and split
by the current node's groups. Although the spectral `C` has entries `0,±1`,
these expanded rows may contain larger integers because an original variable
can occupy several cells. Only identically zero expanded rows are omitted;
nonzero measurements are exact signed integer tuples.

For each original-variable subset `A`, `|A|<=q`, and `B⊆A`, the evaluator forces
`x_A=1` and uses the fixed correction
`K(t+tau,1_B)-K(t,1_B)`. It sums each group's local **clause products** within
measurement buckets, multiplies the group sums, and applies the sign
`(-1)**(|A|-|B|)`. Truncated modular convolution produces Hasse coefficients;
the existing Hermite solver and CRT recover the count. The conservative degree
bound is `b * total_literal_occurrences`, with exactly `D+1` Hermite coefficients.
There is no global-assignment counting fallback or global CNF monomial expansion
in this default path.

The earlier `SwapMobiusCounter` remains an independent finite reference: it
expands the multilinear CNF indicator and uses an injective variable embedding.
Passing an explicit `embedding=` to the convenience wrapper selects that path.
Its potentially `2**n` monomial cost is reported as a separate representation.
The two constructions generally have different polynomials away from zero;
both recover the original Boolean count at zero.

`stage2_design.py` also exposes explicit `ReplacementDesign` objects and exact
rational certificates for restoration and locality (measurement row span plus
owner-group coordinates). The CPU `DesignCounter` in `stage3_design_engine.py`
evaluates certified designs through group representatives. This preserves the
source-level tests for dense non-Boolean replacements, repeated original
variables, changing partitions, and small field characteristic.

This is a bounded, executable mathematical kernel, not an implementation of
the manuscript's complete sparsification, design search, polynomial-space
algorithm, or asymptotic theorem. Defaults restrict counting to six variables;
work and allocation guards reject larger local/subset/jet computations. The
number of Möbius pairs is `sum(comb(n,j)*2**j, j<=q)`, becoming `3**n` once
`q>=n`. The benchmark reports these costs without claiming a runtime crossover.

The implementation follows [O'Donnell's exposition, Sections 2–7](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf)
and the [manuscript's separated evaluator and diagonal-sum recovery, Section 3](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf).
See [the source comparison](docs/v3-research.md) for equation references and the
precise distinction from v2's static suppression schedule.

## Mathematical contract

1. **Width reduction:** each fresh variable satisfies the full equivalence
   `z <-> (a OR b)`. Every original satisfying assignment has exactly one
   extension. `max_width` must be at least three.
2. **Degree splitting:** copies share input-clause occurrences and are linked
   by a directed implication cycle, which forces all copies equal. As specified
   in the implementation prompt, `max_degree` caps input-clause incidences per
   copy. The cycle adds two incidences: the resulting total bound is
   `max_degree + 2`, not `max_degree`. Repeated literals in one clause count as
   one incidence. To target a total bound `Delta > 2`, use
   `max_degree=Delta-2`.
3. **Partitioning:** balanced spectral bisection partitions the variable
   co-occurrence graph, with a deterministic disconnected-graph fallback.
   Clauses go to the group holding the most distinct variables, with a
   deterministic tie break. A proxy is reused per `(master, destination)`.
   Every resulting clause is local. Equality links preserve a unique extension.
4. **Measurements:** a block contains links from a single source group. It may
   have several destinations or repeated master variables. Its measurement
   counts link entries with multiplicity, and its denominator is the actual
   block length, including the final shorter block.
5. **Algebra:** for each link,
   `p(T) = product(1 - T/root)`, `mu = measurement / block_length`,
   `x_tilde = mu + p(T)*(x-mu)`, and
   `Psi = (1-y) + (2*y-1)*x_tilde`, all modulo the selected prime.
   At zero, `Psi` is the Boolean equality indicator. The product has degree
   at most `D = sum(link_degrees)`. The sum defining `H` ranges over **all
   transformed and proxy variables**, not only the original input variables.
6. **Decoupling:** source groups enforce their own measurements and active
   master-bit assignments. Target groups apply their incoming link factors.
   The evaluator sums products of independent group partition sums over
   feasible conditioning signatures; impossible signatures contribute zero.
   A master shared by several links is always assigned consistently.
7. **Recovery:** interpolate from `D+1` distinct nonzero nodes. CRT uses Python
   integers and enough distinct primes that their product is strictly larger
   than the original formula's upper bound `2**num_vars`.

## V2: Hermite interpolation and local moments

`stage2_hermite.py` assigns each boundary link to one suppression node and uses
`p_i(T) = 1 - T/root_i`. Each node suppresses its entire assigned block. Jet
orders `q_s` satisfy `sum(q_s + 1) = B + 1`, so they determine every polynomial
of degree at most `B`. Orders need not be equal; the default distributes the
available coefficients as evenly as possible. Nonempty boundary systems use
nonempty suppression blocks at every node. `single_node_taylor` uses node 1
and order `B`.

`confluent_hermite_weights_at_zero(nodes, orders, prime)` solves
`V.T @ weights = e_0` by exact modular elimination, with
`V[(s,j),m] = binomial(m,j) * nodes[s]**(m-j)` for `m >= j`.
The inputs are **Hasse coefficients**, `[tau**j] H(t_s + tau)`, not ordinary
derivatives. No factorial division is used; Hermite reconstruction also works
when the field characteristic is at most the degree, provided the nodes remain
distinct and nonzero modulo the prime.

`stage3_jet_engine.py` computes these coefficients directly in the truncated
polynomial ring. At node `s`, it conditions each source group on its block
measurements and the unique master bits belonging to links outside `K_s`.
For each derivative subset `U` of `K_s` with `len(U) <= q_s`, it computes
weighted local moments:

```text
C_g^U(signature, proxy_bits)
  = sum of Phi_g(x) * product(x_i - measurement_i/block_size_i)
    over local rows matching the signature and proxy bits,
    with the product over links in U sourced by group g.
```

This retains correlations between suppressed bits and the internal clauses.
Repeated masters contribute once per selected **link**, including powers of
their deviations when several selected links share a master.

For links in `U`, the target factor is `-(2*y-1)/root_i`; their common
`tau**len(U)` shift is applied after contraction. Suppressed links outside `U`
contribute their constant factor `(1-y)+(2*y-1)*mean`. Unsuppressed links
contribute both their constant and linear Taylor coefficients. Independent
group moments and target factors are contracted using modular jet convolution,
truncated to order `q_s-len(U)`. The resulting node jets are combined with the
Hermite weights, then reconstructed through the existing expandable CRT.

The implementation reuses v1's local truth tables and exact tensor arithmetic.
Moment aggregation uses a bounded subset batch axis (`subset_batch_size=8`
by default), vectorizing over subsets, local rows, and proxy configurations.
Subsequent polynomial contractions run per subset. This bounds temporary
storage without constructing the complete subset tensor at once.
It does not call v1 point-value evaluation, DPLL, or a reference counter to
produce a jet. It avoids a global loop over all boundary bit assignments;
local tables and derivative subsets can still require exponential work.

### What this upgrade does and does not reduce

Hermite recovery permits simultaneous suppression, but the proposed **disjoint,
degree-one root schedule** does not guarantee the claimed state reduction:

```text
sum_s |K_s| = B
sum_s (q_s + 1) = B + 1
```

If every node had `q_s < |K_s|`, integer orders would imply
`sum_s(q_s + 1) <= sum_s|K_s| = B`, a contradiction. At least one node must
therefore enumerate all subsets of its suppressed block. In single-node mode,
`q = B` and `sum_{j=0}^B binomial(B,j) = 2**B`: zero active master bits at the
constant coefficient does not remove the cost of their higher-order moments.
At other nodes, bits on unsuppressed links also remain in the conditioning
signature. This is a limitation of the specified micro-engine construction,
not a claim about the full manuscript algorithm.

The benchmark reports the actual v1 signature visits and v2
signature/subset visits across every node and CRT prime, alongside runtime.
These visit counts do not have identical per-visit costs: v2 also performs jet
convolution and moment aggregation. Local assignment, source, and proxy table
sizes are included in exported diagnostics so preprocessing costs are visible.
Resource limits reject oversized subset expansions explicitly.

```sh
python -m pytest -m 'g7 or g8 or g9'
python -m alg_sharpsat.benchmarks --engine both --family coupled --sizes 18 22 26
python -m alg_sharpsat.benchmarks --engine both --family bipartite --sizes 8 10 12
```

## V1 cost and shared exact arithmetic

For the supplied construction, there are exactly `D` suppression-root incidences
across `D+1` evaluation nodes. Their average is `D/(D+1) < 1`. The balanced
schedule therefore suppresses one link at each of `D` nodes and none at the last
node. Increasing every link's degree adds nodes, without improving this average.
Furthermore, one suppressed bit is determined by its block measurement and the
other active bits. A reduction from `2**B` to `product(k_r+1)` alone is **not** a
valid estimate of this evaluator's work: the residual active bits and every
interpolation node must be counted too.

Local truth tables, proxy dimensions, and conditioning products can still grow
exponentially. The engine batches and compresses local computations and raises
`ResourceLimitError` when configured work or memory limits are exceeded. It does
not silently substitute DPLL or the brute-force reference counter. Degree
splitting is available through `max_degree`; it is optional because it may
substantially enlarge these exponential tables.

The initial CRT primes are `998244353`, `1004535809`, and `469762049`, whose
product is `471064322751194440790966273`. This covers all possible counts for at
most 88 declared variables. More primes below `2**30` are selected when needed.
A pairwise residue product fits signed int64, but an unrestricted dot product
does not. Tensor reductions therefore need bounded accumulation and modular
reduction throughout, not only at the end.

The default `matmul_backend="int64"` uses bounded integer dot products on CPU
and batched integer multiplication/reduction on accelerators. The optional
`matmul_backend="limbs"` splits residues into 15-bit limbs and uses binary64
dot products of at most `2**20` terms. Every nonnegative partial result is an
integer below `2**50`, so it is represented exactly; modular recombination is
performed in int64. This optional backend requires float64 device support.

## Verification and benchmarking

The tests check width-reduction and copy-splitting bijections with Hypothesis,
locality and proxy reuse, field identities under every default prime, actual
symbolic polynomial coefficients and degree, direct versus factorized `H(T)`,
interpolation, CRT range, and end-to-end counts against independent brute force.
End-to-end cases include at least 200 generated formulas plus dense formulas
through 16 variables, unused variables, empty formulas, contradictions,
tautologies, multiple groups, and the complete preprocessing pipeline.

The default benchmark uses dense planted cores connected by a bounded number of
bridge clauses at 18, 22, and 26 variables. It reports the boundary size and
measurement-only ratio alongside active-bit conditioning costs and timings.
Fully mixed random and bipartite families are available as stress cases; these
can exceed the prototype's resource limits. The default bounded interface is a
benchmark choice, not evidence of efficiency on arbitrary tightly coupled CNFs.

```sh
python -m alg_sharpsat.benchmarks --help
python -m alg_sharpsat.benchmarks --family coupled --json benchmark-results.json
python -m alg_sharpsat.benchmarks --family bipartite --sizes 18
```

## Recorded v3.1 validation — October 27, 2026

The full suite passed **437 tests**, with **6 hardware tests skipped** because
CUDA and MPS were unavailable. All 412 prior tests still pass. New checks compare
pushdown enabled/disabled against brute force on 240 generated formulas across
two and four groups, and cover correlated source tuples, projection duplicates,
source multiplicity, multiple destinations, cyclic connections, contradictions,
free variables, resource limits, and CRT. All four seed-0 bipartite sizes show
strictly smaller peak frontiers and compressed joint tables; wall time is not
asserted in tests.

The benchmark settings match v3: CPU, Python 3.13.11, PyTorch 2.10.0, one Torch
thread, planted formulas, seed 0, two groups, and medians of three complete runs.
Timings include preprocessing and table construction. Every completed count
matches DPLL. The table retains explicit resource-limit outcomes for the
unchanged Lagrange/Hermite paths.

| Family | Variables | Count | DPLL (ms) | Lagrange (ms) | Hermite (ms) | Sparse v3.1 (ms) | Peak joint rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled | 18 | 48 | 2.559 | 4.035 | 4.757 | 1.124 | 9 |
| Coupled | 22 | 156 | 3.300 | 4.310 | 5.286 | 1.117 | 10 |
| Coupled | 26 | 37 | 5.994 | 4.679 | 5.452 | 1.294 | 4 |
| Coupled | 30 | 253 | 9.888 | 4.766 | 5.721 | 1.475 | 3 |
| Bipartite | 18 | 17 | 1.510 | 51.019 | 153.547 | 1.530 | 301 |
| Bipartite | 22 | 24 | 4.062 | 274.069 | 1225.947 | 1.722 | 707 |
| Bipartite | 26 | 101 | 4.518 | 4678.745 | limit | 2.544 | 3,845 |
| Bipartite | 30 | 933 | 9.840 | limit | limit | 6.858 | 40,303 |

**Bipartite n=30 now beats DPLL:** 6.858 versus 9.840 ms, with the same count
of 933. This is about 2.1 times faster than the previously recorded v3 sparse
runtime of 14.636 ms. The n=18 bipartite case measured slightly slower than
DPLL (1.530 versus 1.510 ms); setup/filtering overhead matters at that scale.
This run beats DPLL on seven of eight cases, with the large-case bottleneck
removed. These timings are measurements of this fixture set and machine.

The following comparison runs the same sparse engine with pushdown disabled
and enabled. “Frontier” is the largest accepted local assignment frontier over
all groups, before key compression. “Joint” is the largest compressed initial
or intermediate factor; neither column is a process-memory measurement.

| Family | Variables | Frontier, disabled | Frontier, enabled | Joint, disabled | Joint, enabled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coupled | 18 | 30 | 24 | 12 | 9 |
| Coupled | 22 | 88 | 56 | 16 | 10 |
| Coupled | 26 | 35 | 35 | 8 | 4 |
| Coupled | 30 | 368 | 84 | 16 | 3 |
| Bipartite | 18 | 1,784 | 439 | 1,200 | 301 |
| Bipartite | 22 | 7,704 | 840 | 5,420 | 707 |
| Bipartite | 26 | 52,272 | 4,931 | 52,272 | 3,845 |
| Bipartite | 30 | 544,896 | 51,562 | 442,784 | 40,303 |

For bipartite n=30, the owned-only frontiers contain 2,738 and 1,486 assignments,
which project to 1,136 and 1,133 distinct outgoing master tuples. After filtered
proxy expansion, the groups have 40,303 and 11,003 joint signatures. The same
639 keys match in the final contraction, giving 933 models after multiplicities.
Peak accepted frontier rows drop by about **90.5%** and peak joint rows by about
**90.9%**. Retaining correlations and discarding duplicate projected witnesses
is necessary for these results to remain exact.

Reproduce the new runs:

```sh
python -m pytest -q --junitxml=verification-v3.1.xml
python -m alg_sharpsat.benchmarks --engine v3 --family coupled \
  --sizes 18 22 26 30 --device cpu --repeats 3 --json benchmark-v3.1-coupled.json
python -m alg_sharpsat.benchmarks --engine v3 --family bipartite \
  --sizes 18 22 26 30 --device cpu --repeats 3 --json benchmark-v3.1-bipartite.json
```

Add `--no-semi-join-pushdown` to compare the original sparse construction.
Reports retain every timing sample, phase-one/support sizes, per-group accepted
and temporary peaks, and complete count/status telemetry:
[coupled](benchmark-v3.1-coupled.json), [bipartite](benchmark-v3.1-bipartite.json),
[enabled/disabled row comparison](benchmark-v3.1-pushdown-comparison.json), and
[test results](verification-v3.1.xml). Historical v3 measurements follow below.

## Recorded v3 validation — October 27, 2026

The final full suite passed **412 tests**, with **6 hardware tests skipped**
because CUDA and MPS were unavailable. Gate g10 includes 150 generated CNFs
for each of two and four groups, hyperedge joins, high-bit keys, and a four-prime
`3**50` contraction. Gate g11 checks all six requested SWAP parameter pairs,
exact spectra and locality, independent polynomial jets, 200 generated tiny
CNFs across the two reference paths, and the original certified-design tests.
Gate g12 verifies benchmark statuses and count comparisons; the recorded runs
below complete the requested size/family matrix. The tests use no skipped or
substituted source-identity cases.

The following runs used **CPU**, Python 3.13.11, PyTorch 2.10.0, one PyTorch
thread, planted formulas, seed 0, and two groups. Times are medians of three
end-to-end runs, including partitioning and local table construction. Every
completed count matched DPLL. These measurements describe the generated
instances and this machine, not all formulas of the same size.

| Family | Variables | Links | Count | DPLL (ms) | Lagrange (ms) | Hermite (ms) | Sparse (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled | 18 | 4 | 48 | 2.545 | 4.838 | 5.840 | 0.965 |
| Coupled | 22 | 4 | 156 | 3.539 | 5.042 | 5.851 | 1.070 |
| Coupled | 26 | 4 | 37 | 6.312 | 5.148 | 5.948 | 1.509 |
| Coupled | 30 | 4 | 253 | 10.410 | 5.385 | 6.064 | 1.440 |
| Bipartite | 18 | 16 | 17 | 1.487 | 54.041 | 161.744 | 1.099 |
| Bipartite | 22 | 19 | 24 | 3.788 | 298.946 | 1306.606 | 1.640 |
| Bipartite | 26 | 24 | 101 | 4.661 | 4783.769 | limit | 3.097 |
| Bipartite | 30 | 27 | 933 | 10.245 | limit | limit | 14.636 |

Sparse contraction meets the requested sub-5 ms target on these bipartite
18–26-variable cases and beats DPLL on seven of the eight recorded cases.
**The 30-variable bipartite case remains slower than DPLL**: 14.636 versus
10.245 ms. No universal DPLL speedup is claimed. On bipartite `n=26`, the sparse
path is about 1,545 times faster than v1 Lagrange, but only about 1.5 times
faster than DPLL.

The true joint signature tables at `n=26` contain **7,680 and 52,272** rows;
only **52** keys intersect, with multiplicities yielding 101 models. The old
incoming-only counts 2,088 and 1,898 are smaller marginal projections, not
complete boundary tables. At `n=30`, the joint tables grow to 442,784 and
65,408 rows, with 639 matching keys. This construction cost explains why that
case still loses to DPLL.

`limit` means an explicit resource guard stopped the engine; it is not a
completed timing or a fallback result. The common budget was 2 billion
estimated operations and 512 MiB per configured workspace. Hermite used six
nodes and stopped at `n=26` (9.34 billion estimated operations at node zero).
At `n=30`, Lagrange estimated 3.36 billion operations, and Hermite estimated
715.51 billion at node zero. Failed-attempt durations and reasons are retained
in JSON separately from successful runtimes. Historical v2 results below used
larger explicit budgets for the largest case.

The SWAP–Möbius reference is measured separately because it is a miniature
mathematical engine. On the three-variable, two-clause fixture in
[the miniature report](benchmark-v3-swap-miniature.json), it returned 4 in
**2.594 ms**, versus DPLL's **0.009 ms** (medians of five runs). It was not run
on the 18–30-variable inputs. Each main benchmark instead records the actual
occurrence design's blocks, measurement rows, degree, jet orders, group sizes,
and combinatorial bounds.

For the 30-variable bipartite instance at `d=2,L=2,b=1`, the design has 8 blocks,
8 nonzero expanded measurement rows, degree bound 360, and maximum jet order
180. Its Möbius sum permits all **3^30 = 205,891,132,094,649** `(A,B)` pairs per
node, before measurement conditioning and local ring arithmetic. These finite
parameters establish **no runtime crossover**. Increasing `L` also grows the
explicit matrix dimension as `d**(2*L)`; the reference caps it at 256.

Reproduce the v3 comparisons:

```sh
python -m pytest -q --junitxml=verification-v3.xml
python -m alg_sharpsat.benchmarks --engine v3 --family coupled \
  --sizes 18 22 26 30 --device cpu --repeats 3 --no-semi-join-pushdown \
  --json benchmark-v3-coupled.json
python -m alg_sharpsat.benchmarks --engine v3 --family bipartite \
  --sizes 18 22 26 30 --device cpu --repeats 3 --no-semi-join-pushdown \
  --json benchmark-v3-bipartite.json
```

Reports: [coupled](benchmark-v3-coupled.json),
[bipartite](benchmark-v3-bipartite.json), and [test results](verification-v3.xml).

## Recorded v2 validation — October 27, 2026

The complete suite passed **290 tests**, with **4 skips** for unavailable
CUDA/MPS hardware. Gate g7 includes 100 random polynomial/capacity-partition
examples under each of the three default primes (300 examples total), through
degree 30. Gate g8 includes 100 random and 100 bipartite CNFs, plus 25
single-node/multigroup examples, independent coefficient-by-coefficient
expansions, repeated masters, arbitrary node orders, preprocessing, and a
four-prime count larger than the original CRT range. Separate tests check
batched moments, overflow-safe convolution, cache reset, resource rejection,
and the absence of scalar-evaluation/reference-counter fallbacks.

Gate g9 compared both engines with DPLL on six formulas. **All exact counts
agree.** The runs used CPU tensors with one PyTorch thread, Python 3.13.11,
and PyTorch 2.10.0. Commands ran sequentially; timings are individual runs,
including initialization, and are not steady-state performance estimates.

| Family | n | Nodes | Count | v1 visits | v2 visits | v1 ms | v2 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Coupled | 18 | 2 | 48 | 135 | 162 | 10.787 | 4.899 |
| Coupled | 22 | 2 | 156 | 150 | 210 | 4.103 | 4.976 |
| Coupled | 26 | 2 | 37 | 60 | 84 | 4.456 | 5.422 |
| Bipartite | 18 | 6 | 17 | 49,572 | 99,414 | 57.087 | 149.646 |
| Bipartite | 22 | 6 | 24 | 252,000 | 571,260 | 272.103 | 1197.252 |
| Bipartite | 26 | 6 | 101 | 1,092,000 | 3,578,700 | 4397.360 | 36946.392 |

Here v1 visits are feasible conditioning-signature tuples and v2 visits are
signature/subset pairs, summed over nodes and all CRT primes. The measured v2
visit counts were **1.2–3.28 times larger**. Hermite recovery is exact, but these
tests do not demonstrate the proposed state-space reduction. The first small
coupled case is sensitive to startup overhead; it does not establish a speedup.

The largest bipartite case exceeded the default per-node work estimate. Its
recorded run explicitly allowed 12 billion estimated operations per node and
a 1 GiB memory budget. Library defaults remain unchanged, and no cases were
silently replaced with another counting algorithm.

Reproduce the full suite and comparisons:

```sh
python -m pytest -q --junitxml=verification-v2.xml
python -m alg_sharpsat.benchmarks --engine both --family coupled --sizes 18 22 26 \
  --json benchmark-v2-coupled.json --csv benchmark-v2-coupled.csv
python -m alg_sharpsat.benchmarks --engine both --family bipartite --sizes 18 22 26 \
  --jet-nodes 6 --max-jet-work 12000000000 --max-memory-bytes 1073741824 \
  --json benchmark-v2-bipartite.json --csv benchmark-v2-bipartite.csv
```

Reports: [test results](verification-v2.xml),
[coupled comparison](benchmark-v2-coupled.json), and
[bipartite comparison](benchmark-v2-bipartite.json). JSON includes every node's
`k`, `q`, processed subsets, residual bits, and full-subset flags, plus the
local table sizes, resource options, and device details.

## Recorded v1 validation — October 27, 2026

The final suite passed **135 tests**, with **2 hardware tests skipped** because
PyTorch reported both CUDA and MPS unavailable. This includes 400 wide-clause
Hypothesis examples, 100 hub-splitting examples at `max_degree=3`, 200 general
end-to-end examples, and 150 additional dense 3-CNF examples through 16 variables.
Tests also reconstruct `3 * 2**98` through actual boundary evaluations,
interpolation, and four CRT primes. A separate agent review checked 400 scalar
versus tensor evaluations with repeated masters and multiple destinations.

The complete nine-case benchmark ran on the **CPU** with Python 3.13.11,
PyTorch 2.10.0, and one PyTorch thread. Every count matched DPLL. These are single
runs with setup included, not statistically established performance claims.

| Family | Variables | Boundary links | Exact count | DPLL (ms) | Tensor (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coupled | 18 | 4 | 48 | 2.935 | 42.589 |
| Coupled | 22 | 4 | 156 | 3.493 | 4.895 |
| Coupled | 26 | 4 | 37 | 6.294 | 5.980 |
| Random | 18 | 17 | 36 | 1.338 | 112.225 |
| Random | 22 | 21 | 38 | 4.258 | 813.883 |
| Random | 26 | 22 | 79 | 4.725 | 2661.020 |
| Bipartite | 18 | 16 | 17 | 1.484 | 54.319 |
| Bipartite | 22 | 19 | 24 | 3.843 | 284.719 |
| Bipartite | 26 | 24 | 101 | 4.693 | 4853.118 |

See [benchmark-results.json](benchmark-results.json) for conditioning states,
active bits, device fallback reasons, versions, and throughput, or
[benchmark-results.csv](benchmark-results.csv) for the flat table. The test
report is [verification.xml](verification.xml).

```sh
python -m pytest -q --junitxml=verification.xml
python -m alg_sharpsat.benchmarks --family all \
  --json benchmark-results.json --csv benchmark-results.csv
```

## Relation to the cited manuscript

The inspiration is the October 25, 2026 manuscript
[The Exponential Time Hypothesis is False](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf).
This project implements the equations supplied in the task. It does not
implement or verify the full manuscript algorithm, its asymptotic bounds, or
the accompanying Lean development. The repository's
[formalization scope](https://github.com/openai/math/blob/main/lean/docs/378.md)
describes a separate theoretical result.
