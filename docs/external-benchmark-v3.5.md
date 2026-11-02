# External benchmark: alg-sharpsat-gpu v3.5.1 (2026-10-27)

This report repeats the v3.4 and v3.5.0 comparison with GANAK, SharpSAT-TD,
PySDD and PySAT RC2 on the repository's own fixtures, after the v3.5.1
changes. The harness is [scripts/external_bench](../scripts/external_bench/common.py).
The same caveat applies as before: at the headline sizes (n ≤ 30) every tool
finishes in about a millisecond, so launch time dominates the external tools.
GANAK and SharpSAT-TD are built for instances that take seconds or minutes.

## Summary

- **All six v3.5.1 gates pass.** v3.5.0 missed only the single-query target;
  v3.5.1 meets it and brings policy compilation back under 2.5 ms.
- **A single query is one fused call.** On the policies one query (Z and all
  marginals) takes 7.5 / 10.8 / 8.9 µs, down from 72 / 75 / 62 µs; PySDD
  answers in 5–6 µs. Across the 58 weighted instances single queries take
  3.8–76 µs, faster than v3.5.0 on every instance and faster than PySDD on 41.
  The first float64 query after a compile also builds the fused tables
  (0.2–1.4 ms).
- **Compilation is about twice as fast and picks the same plans.** Policy
  compile takes 1.77 / 2.34 / 1.91 ms (v3.5.0: 3.74 / 4.98 / 3.65 ms), against
  PySDD's 14.4 / 11.8 / 12.6 ms, so 8.1× / 5.0× / 6.6× faster; the median
  speedup over v3.5.0 across all 58 instances is 1.8×. Compile plus one exact
  query now beats launch-adjusted GANAK on 57 of 58 instances (policies
  2.6–3.3 ms against 9.0–27.5 ms); the exception is bipartite n=30 seed 4
  (16 ms against 9 ms).
- **Exact everywhere.** All 135 counts agree across every tool that finished.
  GANAK's exact rational WMC equals the repository's on 58/58 weighted
  instances, and RC2's MAP weight equals the repository's on 58/58. Every
  float64 marginal, for single queries and all 256 rows of every batch, is
  within 5.3e-15 of the exact rational value (Z within 2.3e-14 relative).
- **Batches and MAP are unchanged; counting got slightly faster.** The 256-query batch (policies
  0.84–1.10 ms against PySDD's 1.47–2.07 ms loop) and exact MAP (0.08–0.11 ms
  against RC2Stratified's 0.27–0.32 ms) beat the comparators on all 58
  instances, as in v3.5.0. The counting algorithms are unchanged; the shared
  counter code is 7–35% faster in the contraction column. The coupled n=100
  median is 9.8 ms, n=150 / 200 / 300 finish in 62 ms / 0.56 s / 11.1 s, and
  the repository is the fastest tool on 35 of the 40 coupled seeds from n=18
  to n=80; it still loses at n=300 (11.1 s against SharpSAT-TD's 4.3 s).
  SharpSAT-TD remains fastest on 68 of the 75 bipartite and random instances;
  the repository now wins seven of them, all with n ≤ 26.

## Gates

| Gate | Target | Measured (clinical / cloud ops / fintech) | Result |
|---|---|---|---|
| g1 exactness | `pytest`; 135 counts; 58/58 GANAK; 58/58 RC2; float marginals within 1e-12 | see the [README verification note](../README.md#v351-fused-single-queries-and-faster-compilation); 135/135; 58/58; 58/58; max error 5.3e-15 | pass |
| g2_batch | < 1.5 ms, faster than PySDD's 256 loop | 0.84 / 1.10 / 0.95 ms against 1.47 / 2.07 / 1.50 ms | pass |
| g2_map | < 0.20 ms, faster than RC2Stratified | 0.11 / 0.097 / 0.078 ms against 0.32 / 0.28 / 0.27 ms | pass |
| g2_single | < 0.030 ms | 0.0075 / 0.011 / 0.0089 ms (v3.5.0: 0.072 / 0.075 / 0.062 ms) | pass |
| g2_compile | < 2.5 ms, at least 5× faster than PySDD | 1.77 / 2.34 / 1.91 ms; 8.1× / 5.0× / 6.6× (v3.5.0: 3.74 / 4.98 / 3.65 ms) | pass; cloud ops is at the 5× edge (4.7× in an earlier run) |
| g3 coupled n=100..300 | all finish; n=100 < 15 ms | 20/20 finish; n=100 per seed 13.9, 12.6, 7.7, 8.8, 9.8 ms | pass |
| g4 report | updated with the gate results | this document | pass |

The v3.5.0 gate table is in the
[v3.5.0 release](https://github.com/scasella/alg-sharpsat-gpu/blob/v3.5.0/docs/external-benchmark-v3.5.md#gates).

## What changed in v3.5.1

- **Fused single queries** ([_guardrail_fast.py](../src/alg_sharpsat/_guardrail_fast.py),
  [_guardrail_native.py](../src/alg_sharpsat/_guardrail_native.py)). On the
  first float64 query a plan is flattened into a table listing every
  root-consistent combination of leaf signatures (938 / 384 / 212
  combinations for cloud ops / clinical / fintech). A query is then a fixed
  sequence: multiply each row's literal weights, sum rows per signature,
  multiply signatures per combination, and scatter the combination products
  back to signatures and marginals. Weights are used unscaled, with global
  bounds that keep every intermediate within `exp(±600)`. The whole query,
  including input checks and result assembly, is one call to a small C
  kernel compiled with the system compiler on first use and cached in a
  private per-user directory. Without a compiler the same algorithm runs in
  NumPy (about 21–29 µs on the policies). Invalid weights, a variable with two
  zero weights, out-of-range weights and oversized tables use the v3.5.0 path.
- **Same plans, cheaper candidates.** Plan selection still compiles the 2-, 6-
  and 4-group candidates and keeps the lowest `plan_cost`. The candidates now
  share bisection orders, clause variable sets and the CPU kernel probe;
  connected components, signature keys, distinct keys and pruning use fewer
  array operations; and a candidate's rows stay bit-packed until it is
  adopted. A fixture of the v3.5.0 choices, candidate costs, retained rows and
  compile work on 73 formulas is part of the test suite, and on those formulas
  the plan arrays are byte-identical to v3.5.0's.
- **Why not a structural cost.** The task suggested choosing the group count
  from structure alone (boundary states and `Σ 2^|V_g|`) and compiling only
  the winner. That cannot reproduce v3.5.0's choices, which depend on row and
  pair counts after backward pruning: on fintech the six- and four-group plans
  differ by 3% (11,054 against 10,735). The exact cost was kept, and each
  candidate was made cheaper instead.
- **Deviations from the task.** The fused tables are built on the first query
  rather than inside `compile_guardrail`, so the compile column does not
  include them and plans used only for exact queries never pay for them; the
  first-query cost is reported separately. The kernel allocates its scratch
  space per call instead of reusing buffers on the compiled object, which
  keeps concurrent queries on one plan safe.

## What changed in v3.5

- **Bridge-cut counting** ([stage3_bridge_cut.py](../src/alg_sharpsat/stage3_bridge_cut.py)).
  When the two-way spectral bisection is crossed by at most six clauses,
  `auto_count_with_stats` removes those bridges and restores them by
  inclusion–exclusion. Each side is counted once by a component-caching search
  on bit-parallel clause sets; its result is a factor keyed by the side's
  bridge variables, so all 2^m conditioned terms are sums of factor entries.
  Plain `sparse_count_with_stats` still contracts unless asked.
- **Cost-based plan selection** (`num_groups="auto"`, the new default for
  `compile_guardrail`). The two-group plan is compiled first; heavy plans also
  compile six and four groups (and the direct model table when cheaper), and
  the plan with the lowest estimated per-query work is kept. The candidate set
  is fixed, so the choice is deterministic. On the policies it selects six or
  four groups, cutting leaf rows from 1,300–12,500 to 400–600.
- **Linear-domain float64 evaluation** ([_guardrail_fast.py](../src/alg_sharpsat/_guardrail_fast.py)):
  per-leaf matrix products and per-query shifts, sparse segment sums, and a
  one-vector `bincount` path. Queries whose leaf log ranges exceed 600 nats
  use the previous log-domain path.
- **Float-guided exact MAP**: a float64 max-product search whose choices are
  accepted only when every comparison wins by more than its rounding bound;
  otherwise the exact integer pass runs. Exact inputs also skip per-variable
  `Fraction` construction.
- A sort-based distinct-key step replaces NumPy's hash-based `unique`, which
  slowed some compiles by up to 40× on structured bitmask keys.

### On the requested direct model table

The task proposed compiling small, high-cut policies into a table of all
satisfying assignments. Measured against the v3.4 two-group plans, the
policies have 21,126 / 170,700 / 24,868 models but only 458 / 192 / 32
boundary signatures, so a full table would have been 10–100× larger than the
plan it replaced. After backward pruning every retained leaf row extends to a
model, so a partitioned plan never needs more row work than the model table;
the table can only win through its single node. It is a plan-selection
candidate on those terms and was chosen for two bipartite seeds. The policy
speedups come from six- and four-group plans instead.

## Setup

Machine: Apple M4 Pro (14 cores, 24 GB), macOS 27.0.1, with other work running
(load average 1.3–5.2 during the v3.5.1 runs). Repository: v3.5.1 working
tree on top of `4b9cfbd` (v3.5.0), Python 3.13.11, PyTorch 2.10.0, NumPy 2.4.2
(Accelerate BLAS), SciPy 1.17.1; the optional single-query kernel was built by
Apple clang 21.0.0 (`cc -O2 -ffp-contract=off`). Repository workers use one
Torch thread and one BLAS thread (`VECLIB_MAXIMUM_THREADS=1`), CPU only.

Fixtures are the repository's seeded generators (coupled, bipartite and random
planted 3-SAT, seeds 0–4) and the first preset of each of the three Studio
domain policies. The DIMACS export is byte-identical to the v3.4 run's. Each
table cell is the median over seeds of per-instance medians. Repeats per
instance: counting 3; weighted single query 300 (30 in v3.5.0); compile 9
(3 in v3.5.0); batch, exact query and MAP 5; cold exact, first query and
1,000 samples 3; external one-shot tools 3;
SharpSAT-TD with `-decot 1` once. Every external call has a 300 s limit
(600 s for `-decot 1`).

| Tool | Version / source | How it was run |
|---|---|---|
| repo contraction | `sparse_count_with_stats(bridge_cut=False)`, 2 groups, CLI budgets | in-process, warm; the v3.4 engine |
| repo auto | `auto_count_with_stats`, same budgets | in-process, warm; bridge-cut search when a two-way cut has ≤ 6 clauses, else contraction |
| repo DPLL | `dpll_count` | in-process, n ≤ 60 |
| repo guardrail | `compile_guardrail(cnf)` (`num_groups="auto"`) | in-process, warm; single queries ran through the fused C kernel on all 58 instances |
| GANAK | v2.8.0 prebuilt macOS arm64 binary | subprocess; default and `--fast`; `--mode 1` for weighted |
| SharpSAT-TD | github.com/Laakeri/sharpsat-td @ 0c234c1, built locally | subprocess; `-decot 0.02` (no TD), plus `0.1` and `1` (TD on) at larger sizes |
| PySDD | 1.0.6 wheel | in-process; `from_cnf_string`, dynamic minimisation on |
| PySAT RC2Stratified | python-sat 1.9.dev15 | subprocess per call; time measured inside the worker |
| d4 | not run (download not approved) | — |

## Method notes and deviations

- v3.5.1 run: the weighted benchmark was rerun in full. For counting, only the
  repository's contraction and auto columns were re-measured
  (`REFRESH_REPO=1`); the GANAK, SharpSAT-TD and DPLL columns are from the
  v3.5.0 run (same tools, settings and fixtures), and all 135 counts were
  checked again. Single-query repeats rose from 30 to 300 and compile repeats
  from 3 to 9, to reduce noise on the gated policy timings.
- The first float64 query after a compile also builds the fused single-query
  tables; it is timed separately (median of three fresh compiles) and not
  included in "repo query". The kernel library was already cached for these
  runs; its one-time build (about 0.24 s with Apple clang) is not included.
- Run-to-run spread: an earlier full weighted run of the same compiler code
  recorded cloud-ops compile at 2.38 ms against PySDD's 11.3 ms (4.7×); this
  run records 2.34 ms against 11.8 ms (5.0×). Single queries matched within
  0.7 µs between the two runs.
- Same fixtures, seeds, budgets and external-tool settings as the v3.4
  run. Changes in v3.5.0: the repository now reports two counting columns (contraction,
  as in v3.4, and the auto selector); the guardrail compiles with its new
  default `num_groups="auto"` instead of an explicit 2, and the selected group
  count is shown; BLAS is pinned to one thread; the cold exact column and the
  float64-versus-exact accuracy checks are new.
- "repo cold exact" times `compile_guardrail(cnf).evaluate(weights, mode="exact_rational")`
  end to end in a warm interpreter, so it includes plan selection. "GANAK exact
  (one-shot, −launch)" is the faster of GANAK default and `--fast` wall time,
  each minus its own launch floor on a three-variable formula.
- Float64 accuracy: each instance's single query and all 256 batch rows were
  compared with exact rational evaluations of the same weights.
- SharpSAT-TD needed two local patches to build and run on arm64 macOS: its
  x86-only clhash was replaced by a portable keyed 64-bit hash (two keys, so
  cache keys stay 128-bit), and a small shim provides GNU `timeout`. It skips
  tree decomposition when `-decot` < 0.099 s. "SSTD internal" is its own
  `Solved in` timer, which runs from program start (it includes parsing and
  preprocessing, not process launch). GANAK prints its timer to two decimals,
  so it is compared by wall time minus launch floor, and only when its wall time
  exceeds twice that floor. "fastest (per seed)" compares the repository's auto
  counter with those per-tool estimates, including the TD-enabled SharpSAT-TD runs.
- RC2's default configuration ran for over 55 minutes on one fixture in v3.4;
  all MAP times use RC2Stratified with integer-scaled −log weights (scale 1e6).
- Not compared: sampling (no external sampler installed), d4, accelerators.

## Unweighted model counting

Times in ms; medians over seeds 0–4. Launch floors (ms): {'ganak': '3.72', 'ganak_fast': '4.30', 'sharpsat_td': '3.60'}
"repo contraction" is `sparse_count_with_stats` (the v3.4 engine); "repo auto"
is `auto_count_with_stats`. The repository columns were re-measured for v3.5.1;
the external and DPLL columns are from the v3.5.0 run, and all 135 counts were
checked again.

| family | n | repo contraction | repo auto | auto engine | repo DPLL | GANAK | GANAK --fast | SharpSAT-TD | SSTD internal | best external − launch | fastest (per seed) | agree |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| bipartite | 18 | 0.94 | 1.01 | sparse_contract | 1.43 | 4.77 | 4.79 | 3.66 | 0.91 | 0.057 | sharpsat_td 4, repo 1 | 5/5 |
| bipartite | 22 | 1.13 | 1.30 | sparse_contract | 3.39 | 5.26 | 5.43 | 3.91 | 0.97 | 0.31 | sharpsat_td 4, repo 1 | 5/5 |
| bipartite | 26 | 1.70 | 1.76 | sparse_contract | 6.68 | 31.6 | 31.0 | 4.93 | 2.20 | 1.33 | sharpsat_td 3, repo 2 | 5/5 |
| bipartite | 30 | 4.07 | 4.24 | sparse_contract | 7.42 | 27.4 | 27.4 | 4.67 | 1.75 | 1.07 | sharpsat_td 5 | 5/5 |
| bipartite | 34 | 7.19 | 7.20 | sparse_contract | 14.5 | 9.61 | 9.34 | 4.59 | 1.76 | 0.99 | sharpsat_td 5 | 5/5 |
| bipartite | 38 | 27.8 | 28.2 | sparse_contract | 26.7 | 47.9 | 48.0 | 6.84 | 3.89 | 3.24 | sharpsat_td 5 | 5/5 |
| bipartite | 42 | 44.5 | 43.8 | sparse_contract | 39.1 | 59.2 | 58.8 | 6.93 | 3.97 | 3.33 | sharpsat_td 5 | 5/5 |
| bipartite | 50 | 798.1 (3/5 limit) | 795.4 (3/5 limit) | sparse_contract | 83.2 | 331.9 | 335.1 | 10.9 | 8.05 | 7.33 | sharpsat_td 5 | 5/5 |
| bipartite | 60 | limit | limit | — | 205.4 | 380.1 | 379.6 | 12.6 | 9.29 | 8.95 | sharpsat_td 5 | 5/5 |
| coupled | 18 | 0.67 | 0.44 | bridge_cut | 1.80 | 5.09 | 4.98 | 3.84 | 0.62 | 0.23 | repo 5 | 5/5 |
| coupled | 22 | 0.77 | 0.52 | bridge_cut | 2.92 | 8.51 | 8.42 | 3.95 | 0.85 | 0.35 | repo 5 | 5/5 |
| coupled | 26 | 0.88 | 0.55 | bridge_cut | 3.95 | 5.73 | 5.38 | 4.00 | 0.86 | 0.40 | repo 5 | 5/5 |
| coupled | 30 | 0.94 | 0.74 | bridge_cut | 8.02 | 9.38 | 9.18 | 4.21 | 1.05 | 0.61 | repo 4, sharpsat_td 1 | 5/5 |
| coupled | 40 | 1.28 | 0.96 | bridge_cut | 20.9 | 9.75 | 9.56 | 4.31 | 1.08 | 0.71 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 50 | 1.64 | 1.76 | bridge_cut | 61.3 | 44.9 | 44.2 | 6.27 | 3.14 | 2.67 | repo 5 | 5/5 |
| coupled | 60 | 2.37 | 2.44 | bridge_cut | 216.5 | 63.8 | 64.2 | 8.24 | 5.06 | 4.64 | repo 5 | 5/5 |
| coupled | 80 | 5.81 | 4.77 | bridge_cut | — | 82.6 | 82.6 | 12.3 | 8.84 | 8.70 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 100 | 40.8 | 9.81 | bridge_cut | — | 357.6 | 355.8 | 13.3 | 9.85 | 9.66 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 150 | limit | 62.0 | bridge_cut | — | 764.4 | 763.4 | 44.0 | 40.3 | 40.4 | sharpsat_td 5 | 5/5 |
| coupled | 200 | limit | 555.1 | bridge_cut | — | 2,189 | 2,211 | 562.7 | 555.6 | 559.1 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 300 | limit | 11,095 | bridge_cut | — | 10,592 | 6,167 | 4,328 | 4,308 | 4,324 | sharpsat_td 3, ganak_fast 2 | 5/5 |
| random | 18 | 0.86 | 0.94 | sparse_contract | 1.22 | 5.22 | 4.82 | 3.61 | 0.76 | 0.008 | sharpsat_td 4, repo 1 | 5/5 |
| random | 22 | 1.11 | 1.18 | sparse_contract | 3.00 | 8.24 | 8.07 | 3.91 | 0.98 | 0.31 | sharpsat_td 4, repo 1 | 5/5 |
| random | 26 | 2.01 | 1.96 | sparse_contract | 4.56 | 8.98 | 9.01 | 4.46 | 1.19 | 0.86 | sharpsat_td 4, repo 1 | 5/5 |
| random | 30 | 3.39 | 3.43 | sparse_contract | 8.32 | 31.1 | 31.7 | 4.92 | 2.14 | 1.32 | sharpsat_td 5 | 5/5 |
| random | 40 | 28.0 | 27.3 | sparse_contract | 27.0 | 31.0 | 30.8 | 6.12 | 3.06 | 2.52 | sharpsat_td 5 | 5/5 |
| random | 50 | 507.8 (4/5 limit) | 501.6 (4/5 limit) | sparse_contract | 64.9 | 227.2 | 225.5 | 10.3 | 6.92 | 6.71 | sharpsat_td 5 | 5/5 |

## Weighted inference

Times in ms; medians over seeds 0–4, or the single domain instance. "plan
groups" lists the group counts selected across seeds; they are the same as in
v3.5.0 on every instance. "repo query" ran through the fused C kernel.

| instance | n | plan groups | repo compile | SDD compile | repo query | SDD query | repo 256 batch | SDD 256 loop | repo exact query | repo cold exact (compile+query) | GANAK exact (one-shot, −launch) | repo MAP | RC2 MAP | repo 1k samples |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bipartite | 18 | 2 | 0.75 | 21.2 | 0.004 | 0.005 | 0.30 | 1.45 | 0.49 | 1.24 | 4.48 | 0.050 | 0.73 | 3.75 |
| bipartite | 22 | 2 | 1.00 | 38.9 | 0.005 | 0.005 | 0.28 | 1.61 | 0.63 | 1.71 | 5.06 | 0.054 | 0.94 | 3.68 |
| bipartite | 26 | 1, 2 | 3.16 | 87.3 | 0.006 | 0.013 | 0.56 | 4.12 | 0.88 | 3.96 | 37.8 | 0.064 | 1.27 | 4.06 |
| bipartite | 30 | 1, 2 | 9.28 | 219.3 | 0.008 | 0.011 | 0.64 | 3.64 | 1.17 | 11.0 | 29.3 | 0.068 | 1.28 | 4.10 |
| coupled | 18 | 2 | 0.53 | 7.70 | 0.004 | 0.004 | 0.19 | 0.94 | 0.44 | 1.00 | 3.88 | 0.049 | 0.67 | 3.55 |
| coupled | 22 | 2 | 0.61 | 17.3 | 0.004 | 0.005 | 0.28 | 1.39 | 0.61 | 1.27 | 7.48 | 0.055 | 0.83 | 3.77 |
| coupled | 26 | 2 | 0.70 | 21.2 | 0.004 | 0.005 | 0.26 | 1.50 | 0.79 | 1.53 | 4.14 | 0.056 | 0.95 | 3.77 |
| coupled | 30 | 2 | 0.78 | 27.5 | 0.004 | 0.006 | 0.35 | 1.89 | 1.03 | 1.86 | 8.85 | 0.074 | 1.03 | 3.89 |
| coupled | 40 | 2 | 1.05 | 62.5 | 0.005 | 0.007 | 0.39 | 2.27 | 1.62 | 2.73 | 9.18 | 0.072 | 1.47 | 4.19 |
| coupled | 50 | 2 | 3.57 | 176.4 | 0.015 | 0.021 | 1.08 | 6.59 | 3.41 | 7.18 | 44.6 | 0.093 | 2.22 | 5.22 |
| coupled | 60 | 2 | 4.42 | 353.2 | 0.026 | 0.037 | 1.65 | 11.1 | 5.29 | 10.3 | 62.6 | 0.10 | 2.82 | 6.44 |
| clinical_prescribing_standard_afib_infection | 22 | 6 | 1.77 | 14.4 | 0.008 | 0.005 | 0.84 | 1.47 | 0.83 | 2.63 | 9.00 | 0.11 | 0.32 | 11.3 |
| cloud_ops_agent_routine_canary | 20 | 6 | 2.34 | 11.8 | 0.011 | 0.006 | 1.10 | 2.07 | 0.80 | 3.26 | 27.5 | 0.097 | 0.28 | 10.6 |
| fintech_treasury_compliant_rebalance | 20 | 4 | 1.91 | 12.6 | 0.009 | 0.005 | 0.95 | 1.50 | 0.68 | 2.62 | 25.5 | 0.078 | 0.27 | 7.29 |

Checks: {'ganak_equals_repo_exact': '58/58', 'rc2_map_equals_repo': '58/58', 'repo_ok': '58/58'} | counts agree 135/135 | float64 vs exact: max marginal abs err 5.3e-15, max Z rel err 2.3e-14 | max SDD Z rel err 1.4e-14

## Repository-only comparison with the v3.5.0 run

The v3.5.0 numbers come from the earlier run of the same fixtures on the same
machine (different time and background load).

| rows | previous contraction | this run contraction | previous auto | this run auto |
|---|---:|---:|---:|---:|
| coupled n=50 | 2.42 | 1.64 | 1.92 | 1.76 |
| coupled n=60 | 3.24 | 2.37 | 2.56 | 2.44 |
| coupled n=80 | 6.99 | 5.81 | 4.78 | 4.77 |
| coupled n=100 | 44.1 | 40.8 | 10.2 | 9.81 |
| coupled n=150 | limit | limit | 63.1 | 62.0 |
| coupled n=200 | limit | limit | 563.0 | 555.1 |
| coupled n=300 | limit | limit | 11,263 | 11,095 |

| instance | metric | previous | this run |
|---|---|---:|---:|
| clinical_prescribing_standard_afib_infection | compile | 3.74 | 1.77 |
| clinical_prescribing_standard_afib_infection | query | 0.072 | 0.008 |
| clinical_prescribing_standard_afib_infection | cold exact | 4.51 | 2.63 |
| clinical_prescribing_standard_afib_infection | 256 batch | 0.83 | 0.84 |
| clinical_prescribing_standard_afib_infection | exact query | 0.81 | 0.83 |
| clinical_prescribing_standard_afib_infection | MAP | 0.094 | 0.11 |
| cloud_ops_agent_routine_canary | compile | 4.98 | 2.34 |
| cloud_ops_agent_routine_canary | query | 0.075 | 0.011 |
| cloud_ops_agent_routine_canary | cold exact | 5.81 | 3.26 |
| cloud_ops_agent_routine_canary | 256 batch | 1.07 | 1.10 |
| cloud_ops_agent_routine_canary | exact query | 0.81 | 0.80 |
| cloud_ops_agent_routine_canary | MAP | 0.098 | 0.097 |
| fintech_treasury_compliant_rebalance | compile | 3.65 | 1.91 |
| fintech_treasury_compliant_rebalance | query | 0.062 | 0.009 |
| fintech_treasury_compliant_rebalance | cold exact | 4.32 | 2.62 |
| fintech_treasury_compliant_rebalance | 256 batch | 0.93 | 0.95 |
| fintech_treasury_compliant_rebalance | exact query | 0.70 | 0.68 |
| fintech_treasury_compliant_rebalance | MAP | 0.083 | 0.078 |

## Reproduce

```sh
python scripts/external_bench/export.py TOOLS WORK
python scripts/external_bench/run_counting.py TOOLS WORK
python scripts/external_bench/run_weighted.py TOOLS WORK
python scripts/external_bench/final_tables.py WORK [PREVIOUS_WORK]
```

TOOLS must contain `ganak/x/ganak`, `sstd/` (SharpSAT-TD and
`flow_cutter_pace17`), `shim/timeout` and an unpacked `pysdd` wheel. They are
not repository dependencies and were obtained separately. With
`REFRESH_REPO=1`, `run_counting.py` re-measures only the repository columns
of an existing `counting.json`. The single-query kernel is built on first use
when a C compiler is available; run with `ALG_SHARPSAT_NATIVE=0` to measure
the NumPy path instead.
