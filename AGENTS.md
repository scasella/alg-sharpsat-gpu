# Working in alg-sharpsat-gpu

This prototype audits counting identities and implements bounded exact #SAT and
weighted inference. It does not establish the manuscript's asymptotic theorem
or a universal speedup. Follow the user's requested scope; this guide supplies
repository context, not additional approval requirements.

## Start and finish

1. Read [README.md](README.md) for engine scope and measured results. For policy,
   inference, Studio, or extraction changes, read [the guardrail guide](docs/wmc-guardrail.md).
   For occurrence-design mathematics, read [the research notes](docs/v3-research.md).
2. Identify the modules and gate contracts below that the change affects. Keep
   public exports, telemetry, documentation, and tests consistent with the change.
3. Run the affected gates and relevant unmarked arithmetic tests. Before a
   release, run the full suite and repository audit commands below. Report
   hardware skips, explicit limits, and live extraction separately from success.

## Source map

Every file under `src/alg_sharpsat` has a distinct responsibility:

| File | Responsibility |
| --- | --- |
| [__init__.py](src/alg_sharpsat/__init__.py) | Public imports and compatibility surface; update when exposing APIs. |
| [cnf.py](src/alg_sharpsat/cnf.py) | Literal/CNF representation, DIMACS, independent reference counters. |
| [stage1_sparsify.py](src/alg_sharpsat/stage1_sparsify.py) | Deterministic width reduction, implication-cycle copies, partitioning, boundary proxies. |
| [stage2_algebraic.py](src/alg_sharpsat/stage2_algebraic.py) | Finite-field factors, suppression schedules, interpolation, CRT primes/reconstruction. |
| [stage2_hermite.py](src/alg_sharpsat/stage2_hermite.py) | Hasse/Hermite schedules and confluent interpolation weights. |
| [stage2_design.py](src/alg_sharpsat/stage2_design.py) | Bounded occurrence replacements and restoration/locality certificates. |
| [stage3_gpu_engine.py](src/alg_sharpsat/stage3_gpu_engine.py) | v1 tensor counter, device probes, bounded modular arithmetic and resource errors. |
| [stage3_jet_engine.py](src/alg_sharpsat/stage3_jet_engine.py) | v2 local moment/subset Hasse jets and exact reconstruction. |
| [stage3_sparse_contract.py](src/alg_sharpsat/stage3_sparse_contract.py) | v3 sparse key factors, source-support semijoins and boundary contraction. |
| [stage3_design_engine.py](src/alg_sharpsat/stage3_design_engine.py) | Bounded CPU occurrence-design/Möbius jet reference. |
| [stage4_swap_mobius.py](src/alg_sharpsat/stage4_swap_mobius.py) | Exact miniature SWAP spectra, Boolean Möbius identities and counting reference. |
| [stage5_wmc_guardrail.py](src/alg_sharpsat/stage5_wmc_guardrail.py) | Compile sparse support/join trees; weighted partition function, marginals and batch evaluation. |
| [_guardrail_exact.py](src/alg_sharpsat/_guardrail_exact.py) | Private bounded integer/modular upward/downward weighted passes and CRT. |
| [_guardrail_inference.py](src/alg_sharpsat/_guardrail_inference.py) | Private MAP backtracking and rejection-free conditional sampling. |
| [policy_dsl.py](src/alg_sharpsat/policy_dsl.py) | Named variables/rules, parser, deterministic CNF compilation, clamps and exact policy diff. |
| [domains.py](src/alg_sharpsat/domains.py) | Domain packs, variable roles, rules, presets and scenario examples. |
| [llm_bridge.py](src/alg_sharpsat/llm_bridge.py) | Live Codex extraction, explicit deterministic backend, schema/provenance validation. |
| [webapp.py](src/alg_sharpsat/webapp.py) | Studio HTTP server, request validation, evaluate/intercept/diff APIs and CLI. |
| [ui_studio.html](src/alg_sharpsat/ui_studio.html) | Packaged Studio UI, locks, request lifecycle, extraction and posterior views. |
| [benchmarks.py](src/alg_sharpsat/benchmarks.py) | #SAT fixtures, engine comparisons, count verification and telemetry exports. |
| [guardrail_benchmarks.py](src/alg_sharpsat/guardrail_benchmarks.py) | Weighted inference benchmark and verification runner. |

## Mathematical and API invariants

- **Counting bijections:** each original satisfying assignment has exactly one
  deterministic gate/copy extension. Unused declared variables still contribute
  their multiplicity. Literal indices are zero-based internally and one-based,
  signed in DIMACS.
- **Honest limits:** `max_work`, `max_memory_bytes`, `max_local_assignments`, and
  `max_join_pairs` must raise `ResourceLimitError` when their bounds are exceeded.
  Production counting paths must not silently
  replace their algorithm with DPLL, brute force, scalar interpolation, or a
  reference counter. Benchmarks may call independent references to verify results.
- **Exact arithmetic:** reduce bounded accumulations before int64 overflow.
  Expand CRT coverage to the proven count/weight bound. The optional 15-bit limb
  backend relies on exact binary64 partial sums; preserve its chunk bounds and
  device capability probes. Device availability alone is insufficient.
- **Jets:** use Hasse coefficients, including the specified derivative orders and
  suppressed subsets. Single-node Taylor mode can enumerate `2**B` subsets;
  report that work. A measurement-only state count omits residual bits and nodes.
- **Sparse joins:** a master shared by several destinations is a hyperedge, not
  independent pairwise copies. Join on complete shared keys; preserve correlated
  tuples, high bits and multiplicities. Source-support semijoins are unweighted
  feasibility filters: deduplicate projected witnesses and count each local
  assignment once. Source multiplicity belongs in factors, never in the filter.
- **Weights:** apply original-variable weights once. Deterministic auxiliaries
  and boundary proxies have neutral weights `(1, 1)`; weighting every copy changes
  the distribution. `exact_rational` uses exact rational inputs/outputs and
  bounded integer/CRT passes. Guardrail `float64` evaluation runs on CPU;
  `float32` supports capability-checked tensor devices and is approximate.
  Floating evaluation and exact counting have different guarantees.
- **Policy diffs:** compile the conjunction of old and new rules in a fresh
  auxiliary namespace. Align probabilities by variable name even when variable
  order differs. Added mass/count is new minus intersection; removed is old
  minus intersection. Reusing independently allocated auxiliary IDs is unsound.
- **Clamps:** false evidence sets weights to `(1, 0)` and true evidence to
  `(0, 1)`. With evidence applied, `Z` is the remaining policy-satisfying
  probability mass under the clamped product distribution; posterior marginals
  condition on both those clamps and the policy. Low-level library `clamped_vars` accepts manual binary assignments
  to known policy variables. The extractor and Studio/API boundary additionally
  restrict clamps by role: evidence and safeguards are eligible; proposed actions
  remain soft. Preserve this distinction rather than imposing role restrictions
  on the general library API.
- **Provenance:** extraction requires a nonempty, contiguous, case-preserving
  verbatim scenario substring for each clamp. A quote proves textual provenance,
  not truth or entailment. Preserve uncertain inputs as probabilities, merge
  existing evidence locks without overwriting conflicts, and keep live extraction
  failures visible. The deterministic backend is an explicitly selected offline
  parser, not a silent live fallback. UI changes must retain stale-response
  protection and clear locks when domains/presets change.
- **Inference:** preserve zero-mass failures, evidence in MAP/samples, exact
  integer-weight sampling, and satisfaction of enabled rules. A proposed safeguard
  in a repaired tuple does not execute verification or obtain an approval.

## Test gates

Markers are defined in [pyproject.toml](pyproject.toml); run `python -m pytest
-m g18` (substitute a gate) or combine markers with `or`. Files may also contain
unmarked tests: a gate run does not replace the full suite.

| Gate | Actual test files under `tests/` | Contract |
| --- | --- | --- |
| g1 | [test_stage1_invariants.py](tests/test_stage1_invariants.py) | Deterministic OR-gate extension/counting bijection. |
| g2 | [test_stage1_invariants.py](tests/test_stage1_invariants.py) | Implication-cycle copy bijection and degree bounds. |
| g3 | [test_stage2_polynomial.py](tests/test_stage2_polynomial.py) | Finite-field boundary identities. |
| g4 | [test_stage2_polynomial.py](tests/test_stage2_polynomial.py), [test_end_to_end_exact.py](tests/test_end_to_end_exact.py) | Degree, factorization, interpolation and CRT coverage. |
| g5 | [test_end_to_end_exact.py](tests/test_end_to_end_exact.py) | Exact differential counts, preprocessing and edge cases. |
| g6 | [test_benchmarks.py](tests/test_benchmarks.py) | Benchmark smoke tests and count verification. |
| g7 | [test_hermite_weights.py](tests/test_hermite_weights.py) | Confluent interpolation of Hasse coefficients. |
| g8 | [test_hermite_jet.py](tests/test_hermite_jet.py) | Exact jets, independent coefficient expansions and resource bounds. |
| g9 | [test_hermite_jet.py](tests/test_hermite_jet.py), [test_benchmarks.py](tests/test_benchmarks.py) | Measured Lagrange/Hermite work and runtime comparisons. |
| g10 | [test_sparse_contract.py](tests/test_sparse_contract.py), [test_sparse_semijoin.py](tests/test_sparse_semijoin.py), [test_sparse_host_keys.py](tests/test_sparse_host_keys.py), [test_v3_source_identities.py](tests/test_v3_source_identities.py) | Exact sparse joins, hyperedges, host/high-bit keys, unweighted semijoins and source identities. |
| g11 | [test_swap_mobius.py](tests/test_swap_mobius.py), [test_swap_occurrence.py](tests/test_swap_occurrence.py) | SWAP spectra/locality, Möbius and certified occurrence identities. |
| g12 | [test_benchmarks_v3.py](tests/test_benchmarks_v3.py) | v3 statuses, comparisons and finite crossover estimates. |
| g13 | [test_wmc_guardrail.py](tests/test_wmc_guardrail.py) | Exact rational weighted counting and all marginals. |
| g14 | [test_wmc_guardrail.py](tests/test_wmc_guardrail.py) | Safe MAP and rejection-free conditional samples. |
| g15 | [test_wmc_guardrail.py](tests/test_wmc_guardrail.py) | Batching and verified guardrail benchmarks. |
| g16 | [test_policy_dsl.py](tests/test_policy_dsl.py), [test_guardrail_domains.py](tests/test_guardrail_domains.py) | DSL/domain semantics, neutral auxiliaries and exact policy diffs. |
| g17 | [test_guardrail_app.py](tests/test_guardrail_app.py) | Studio REST/CLI, validation and integration. |
| g18 | [test_llm_intercept.py](tests/test_llm_intercept.py), [test_evidence_clamping.py](tests/test_evidence_clamping.py), [test_intercept_api.py](tests/test_intercept_api.py) | Extraction schema/provenance, role-aware boundary clamps and evidence preservation. |

For arithmetic changes, also run [test_tensor_arithmetic.py](tests/test_tensor_arithmetic.py)
and [test_jet_arithmetic.py](tests/test_jet_arithmetic.py), which contain unmarked
coverage. Hardware skips must identify the unavailable device, not claim a pass.
The unmarked [repository-audit regression tests](tests/test_repository_audit.py)
verify local link detection and run as part of the full suite.

## Offline and live workflows

Run from the repository root with Python 3.11+ and dependencies installed:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pytest -q
python -m alg_sharpsat.benchmarks --engine v3 --family coupled --sizes 18 --device cpu --repeats 1
wmc-guardrail-studio --domain fintech_treasury --backend deterministic --intercept "circuit_breaker_tripped=true; smart_contract_bridge_tx=true"
```

The offline suite uses mocks/deterministic extraction; it does not establish live
model behavior. For a live smoke test, use an authenticated local Codex CLI:

```sh
wmc-guardrail-studio --domain clinical_prescribing --mode exact_rational --intercept "The patient has severe renal impairment and an acute bacterial infection. The agent proposes ketorolac without an antibiotic."
```

To enable the opt-in live g18 tests, run:

```sh
WMC_LIVE_CODEX=1 python -m pytest -q -m g18
```

This invokes `codex exec -m gpt-6-luna` through the bridge. Record the actual
backend/model, evidence quotes, status and latency when reporting a live result.
To inspect the browser UI, run `wmc-guardrail-studio` and open its printed local
URL; API payloads and endpoint examples are in [the guide](docs/wmc-guardrail.md).

## Change recipes

**New domain**

1. Extend `domains.py` with stable variable names, explicit roles, meaningful
   descriptions, named rules, presets and scenarios. Register the new pack's
   factory in `domains.get_domains()` so `get_domain()` and Studio can discover it.
   Complete when every variable has a role and every preset's claimed violation
   agrees with `check_violations`.
2. Add the new presets to `SCENARIOS` and update the fixed-domain expectations in
   `tests/test_guardrail_domains.py`. Add extraction and evidence-preservation
   coverage for the new pack, then run g16–g18. Complete when policy semantics,
   extraction validation and evidence preservation pass for the new pack.
3. Open Studio. Complete when the pack appears, scenarios load, and changing the
   domain/preset clears evidence locks.

**External agent loop**

1. Build a `PolicySpec` and call `compiled = policy.compile_guardrail()` once to
   obtain a reusable `CompiledPolicy`. Obtain a complete soft probability mapping
   and validate extraction through the bridge or an equivalent boundary. Complete
   when malformed extraction and attempted action clamps are rejected.
2. Pass the same `clamped_vars` to `compiled.evaluate()`,
   `compiled.map_inference()` and `compiled.sample_conditional()`. Complete when
   contradictory locks and zero-mass inputs fail visibly, and returned tuples
   preserve evidence and satisfy enabled rules.
3. Inspect violations and the full repaired tuple before the caller decides an
   action. Complete when the integration treats repaired safeguards as conditions
   to satisfy and keeps external action execution explicit in the caller.

**Tensor backend**

1. Extend device probes and kernels together. Complete when integer bounds, limb
   precision and memory/work estimates justify enabling the backend, including
   explicit unsupported-device errors and auto-selection telemetry.
2. Compare small inputs against independent exact references. Run unmarked
   tensor/jet tests plus g3–g5, g8, g10 and g13–g15 as applicable. Complete when
   exact counts agree and approximate weighted results meet stated tolerances.
3. Measure startup and complete-run timing separately. Complete when the report
   records device, versions, threads, budgets, repeats and verification status.

## Reproducible release audit

Use temporary destinations for fresh reports so historical measurements remain
attributable. Run the audit after source/docs changes:

```sh
python -m pytest -q --junitxml=/tmp/alg-sharpsat-release.xml
python scripts/check_repository.py
python scripts/check_repository.py --check-snippets
python -m pip wheel . --no-deps --wheel-dir /tmp/alg-sharpsat-wheels
git diff --check
git status --short
git ls-files .venv venv build dist .pytest_cache .hypothesis .ruff_cache
```

When changing algorithms, performance behavior or published benchmark results,
reproduce the affected families and sizes; documentation-only changes do not
require rerunning the full benchmark matrix. For the recorded v3 matrix:

```sh
python -m alg_sharpsat.benchmarks --engine v3 --family coupled --sizes 18 22 26 30 --device cpu --repeats 3 --json /tmp/alg-sharpsat-coupled.json
python -m alg_sharpsat.benchmarks --engine v3 --family bipartite --sizes 18 22 26 30 --device cpu --repeats 3 --json /tmp/alg-sharpsat-bipartite.json
```

The repository checker audits documentation links and release artifacts; its
snippet option checks executable documentation examples. Inspect the wheel for
`ui_studio.html` and the public modules when packaging changes. Keep `.venv`,
caches, bytecode, build outputs and machine-specific paths out of tracked files.
Release reports should identify seed, family, groups, device, thread count,
budgets, repeats, verification status and failed-attempt reasons. Update version
references consistently with `pyproject.toml`; retain historical results as
historical, and derive fresh test totals from the current report.
