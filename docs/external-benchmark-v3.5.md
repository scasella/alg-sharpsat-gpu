# External benchmark: alg-sharpsat-gpu v3.5.0 (2026-10-27)

This report repeats the v3.4 comparison with GANAK, SharpSAT-TD, PySDD and
PySAT RC2 on the repository's own fixtures, after the v3.5 changes. The
harness is [scripts/external_bench](../scripts/external_bench/common.py).
The same caveat applies as before: at the headline sizes (n ≤ 30) every tool
finishes in about a millisecond, so launch time dominates the external tools.
GANAK and SharpSAT-TD are built for instances that take seconds or minutes.

## Summary

- **Exact everywhere.** All 135 counts agree across every tool that finished.
  GANAK's exact rational WMC equals the repository's on 58/58 weighted
  instances, and RC2's MAP weight equals the repository's on 58/58. Every
  float64 marginal, for single queries and all 256 rows of every batch, is
  within 5.3e-15 of the exact rational value (Z within 2.3e-14 relative).
- **Coupled counting now finishes at every size.** The auto selector counts
  `coupled` formulas (two dense halves joined by four clauses) by bridge-cut
  inclusion–exclusion. The n=100 median falls from 44.2 ms (contraction) to
  10.2 ms, and n=150, 200 and 300, which v3.4 could not count, finish in
  63 ms, 0.56 s and 11.3 s. The repository is the fastest tool on 34 of the
  40 coupled seeds from n=18 to n=80, ties SharpSAT-TD at n=200 (563 ms
  against 556 ms internal; fastest on 3 of 5 seeds, including the hardest,
  1.46 s against 3.3 s for GANAK and 16.7 s for SharpSAT-TD), and loses at
  n=300 (11.3 s against 4.3 s and 6.2 s for GANAK `--fast`).
- **Bipartite and random counting are unchanged.** They have no small clause
  cut, so the auto selector contracts as before. SharpSAT-TD is fastest on
  all 75 of those instances; contraction still stops at bipartite n ≥ 50 and
  random n = 50.
- **Weighted batches and MAP now beat the comparators on every instance.**
  The 256-query batch is faster than PySDD's 256-query loop on all 58
  instances (domain policies 0.83–1.07 ms against 1.46–2.04 ms; v3.4 took
  8.4–80 ms), and exact MAP is faster than RC2Stratified on all 58 (policies
  0.08–0.10 ms against 0.26–0.29 ms; v3.4 took 0.69–6.5 ms).
- **Single queries are still slower than PySDD.** One query (Z and all
  marginals) takes 29–90 µs against PySDD's 3–68 µs; on the policies it is
  62–75 µs against 5–7 µs, missing the 30 µs target. The remaining cost is
  NumPy dispatch: a six-group plan needs dozens of array calls per query, at
  roughly a microsecond each.
- **Compilation got slower on the policies.** Plan selection compiles three
  candidate plans, so policy compile time rose from 0.9–1.8 ms to 3.7–5.0 ms.
  The better plans make exact queries faster, and compile-plus-exact-query
  ("cold exact") still beats launch-adjusted GANAK on 56 of 58 instances and
  on every table row (policies 4.3–5.8 ms against 8.5–26.9 ms). The two
  exceptions are coupled n=40 seed 0 (3.5 ms against 3.4 ms) and bipartite
  n=30 seed 4 (22 ms against 9 ms, from its 21 ms two-group compile).

## Gates

| Gate | Target | Measured | Result |
|---|---|---|---|
| v35_g1 exactness | `pytest`; 135 counts; 58/58 GANAK; 58/58 RC2; float marginals within 1e-12 | see the [README verification note](../README.md#v35-cost-based-plans-fast-float64-and-bridge-cut-counting); 135/135; 58/58; 58/58; max error 5.3e-15 | pass |
| v35_g2 policy batch | < 1.5 ms, beating PySDD's 256 loop | 0.83 / 1.07 / 0.93 ms against 1.53 / 2.04 / 1.46 ms | pass |
| v35_g2 policy MAP | < 0.20 ms, beating RC2Stratified | 0.094 / 0.098 / 0.083 ms against 0.29 / 0.28 / 0.26 ms | pass |
| v35_g2 policy query | < 0.03 ms | 0.072 / 0.075 / 0.062 ms (v3.4: 0.39 / 0.49 / 0.22 ms) | **missed** |
| v35_g3 coupled n=100..300 | finish; n=100 < 15 ms | 20/20 finish; n=100 per seed 14.2, 12.9, 7.8, 8.9, 10.2 ms | pass |
| v35_g4 report | regenerated with the cold exact column | this document | pass |

Policy order in the gate rows: clinical prescribing, cloud ops, fintech treasury.

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
(load average 2.5–3.8 during the runs). Repository: v3.5.0 working tree on top of `a7eede1`,
Python 3.13.11, PyTorch 2.10.0, NumPy 2.4.2 (Accelerate BLAS), SciPy 1.17.1.
Repository workers use one Torch thread and one BLAS thread
(`VECLIB_MAXIMUM_THREADS=1`), CPU only.

Fixtures are the repository's seeded generators (coupled, bipartite and random
planted 3-SAT, seeds 0–4) and the first preset of each of the three Studio
domain policies. The DIMACS export is byte-identical to the v3.4 run's. Each
table cell is the median over seeds of per-instance medians. Repeats per
instance: counting 3; weighted single query 30; batch, exact query and MAP 5;
compile, cold exact and 1,000 samples 3; external one-shot tools 3;
SharpSAT-TD with `-decot 1` once. Every external call has a 300 s limit
(600 s for `-decot 1`).

| Tool | Version / source | How it was run |
|---|---|---|
| repo contraction | `sparse_count_with_stats(bridge_cut=False)`, 2 groups, CLI budgets | in-process, warm; the v3.4 engine |
| repo auto | `auto_count_with_stats`, same budgets | in-process, warm; bridge-cut search when a two-way cut has ≤ 6 clauses, else contraction |
| repo DPLL | `dpll_count` | in-process, n ≤ 60 |
| repo guardrail | `compile_guardrail(cnf)` (`num_groups="auto"`) | in-process, warm |
| GANAK | v2.8.0 prebuilt macOS arm64 binary | subprocess; default and `--fast`; `--mode 1` for weighted |
| SharpSAT-TD | github.com/Laakeri/sharpsat-td @ 0c234c1, built locally | subprocess; `-decot 0.02` (no TD), plus `0.1` and `1` (TD on) at larger sizes |
| PySDD | 1.0.6 wheel | in-process; `from_cnf_string`, dynamic minimisation on |
| PySAT RC2Stratified | python-sat 1.9.dev15 | subprocess per call; time measured inside the worker |
| d4 | not run (download not approved) | — |

## Method notes and deviations

- Same fixtures, seeds, repeats, budgets and external-tool settings as the v3.4
  run. Changes: the repository now reports two counting columns (contraction,
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
"repo contraction" is the v3.4 engine; "repo auto" is `auto_count_with_stats`.
The repository columns were re-measured after the distinct-key fix (external
columns from the first pass the same evening; all 135 counts re-verified).

| family | n | repo contraction | repo auto | auto engine | repo DPLL | GANAK | GANAK --fast | SharpSAT-TD | SSTD internal | best external − launch | fastest (per seed) | agree |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|
| bipartite | 18 | 1.26 | 1.29 | sparse_contract | 1.43 | 4.77 | 4.79 | 3.66 | 0.91 | 0.057 | sharpsat_td 5 | 5/5 |
| bipartite | 22 | 1.56 | 1.57 | sparse_contract | 3.39 | 5.26 | 5.43 | 3.91 | 0.97 | 0.31 | sharpsat_td 5 | 5/5 |
| bipartite | 26 | 2.49 | 2.67 | sparse_contract | 6.68 | 31.6 | 31.0 | 4.93 | 2.20 | 1.33 | sharpsat_td 5 | 5/5 |
| bipartite | 30 | 6.13 | 6.29 | sparse_contract | 7.42 | 27.4 | 27.4 | 4.67 | 1.75 | 1.07 | sharpsat_td 5 | 5/5 |
| bipartite | 34 | 11.3 | 10.8 | sparse_contract | 14.5 | 9.61 | 9.34 | 4.59 | 1.76 | 0.99 | sharpsat_td 5 | 5/5 |
| bipartite | 38 | 39.8 | 40.2 | sparse_contract | 26.7 | 47.9 | 48.0 | 6.84 | 3.89 | 3.24 | sharpsat_td 5 | 5/5 |
| bipartite | 42 | 65.2 | 65.0 | sparse_contract | 39.1 | 59.2 | 58.8 | 6.93 | 3.97 | 3.33 | sharpsat_td 5 | 5/5 |
| bipartite | 50 | 961.3 (3/5 limit) | 962.0 (3/5 limit) | sparse_contract | 83.2 | 331.9 | 335.1 | 10.9 | 8.05 | 7.33 | sharpsat_td 5 | 5/5 |
| bipartite | 60 | limit | limit | — | 205.4 | 380.1 | 379.6 | 12.6 | 9.29 | 8.95 | sharpsat_td 5 | 5/5 |
| coupled | 18 | 0.94 | 0.45 | bridge_cut | 1.80 | 5.09 | 4.98 | 3.84 | 0.62 | 0.23 | repo 5 | 5/5 |
| coupled | 22 | 1.07 | 0.57 | bridge_cut | 2.92 | 8.51 | 8.42 | 3.95 | 0.85 | 0.35 | repo 5 | 5/5 |
| coupled | 26 | 1.25 | 0.65 | bridge_cut | 3.95 | 5.73 | 5.38 | 4.00 | 0.86 | 0.40 | repo 4, sharpsat_td 1 | 5/5 |
| coupled | 30 | 1.40 | 0.78 | bridge_cut | 8.02 | 9.38 | 9.18 | 4.21 | 1.05 | 0.61 | repo 4, sharpsat_td 1 | 5/5 |
| coupled | 40 | 1.88 | 1.03 | bridge_cut | 20.9 | 9.75 | 9.56 | 4.31 | 1.08 | 0.71 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 50 | 2.42 | 1.92 | bridge_cut | 61.3 | 44.9 | 44.2 | 6.27 | 3.14 | 2.67 | repo 5 | 5/5 |
| coupled | 60 | 3.24 | 2.56 | bridge_cut | 216.5 | 63.8 | 64.2 | 8.24 | 5.06 | 4.64 | repo 5 | 5/5 |
| coupled | 80 | 6.99 | 4.78 | bridge_cut | — | 82.6 | 82.6 | 12.3 | 8.84 | 8.70 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 100 | 44.1 | 10.2 | bridge_cut | — | 357.6 | 355.8 | 13.3 | 9.85 | 9.66 | sharpsat_td 3, repo 2 | 5/5 |
| coupled | 150 | limit | 63.1 | bridge_cut | — | 764.4 | 763.4 | 44.0 | 40.3 | 40.4 | sharpsat_td 5 | 5/5 |
| coupled | 200 | limit | 563.0 | bridge_cut | — | 2,189 | 2,211 | 562.7 | 555.6 | 559.1 | repo 3, sharpsat_td 2 | 5/5 |
| coupled | 300 | limit | 11,263 | bridge_cut | — | 10,592 | 6,167 | 4,328 | 4,308 | 4,324 | sharpsat_td 3, ganak_fast 2 | 5/5 |
| random | 18 | 1.30 | 1.35 | sparse_contract | 1.22 | 5.22 | 4.82 | 3.61 | 0.76 | 0.008 | sharpsat_td 5 | 5/5 |
| random | 22 | 1.68 | 1.73 | sparse_contract | 3.00 | 8.24 | 8.07 | 3.91 | 0.98 | 0.31 | sharpsat_td 5 | 5/5 |
| random | 26 | 2.87 | 2.89 | sparse_contract | 4.56 | 8.98 | 9.01 | 4.46 | 1.19 | 0.86 | sharpsat_td 5 | 5/5 |
| random | 30 | 5.20 | 5.29 | sparse_contract | 8.32 | 31.1 | 31.7 | 4.92 | 2.14 | 1.32 | sharpsat_td 5 | 5/5 |
| random | 40 | 41.4 | 42.3 | sparse_contract | 27.0 | 31.0 | 30.8 | 6.12 | 3.06 | 2.52 | sharpsat_td 5 | 5/5 |
| random | 50 | 661.2 (4/5 limit) | 659.3 (4/5 limit) | sparse_contract | 64.9 | 227.2 | 225.5 | 10.3 | 6.92 | 6.71 | sharpsat_td 5 | 5/5 |

## Weighted inference

Times in ms; medians over seeds 0–4, or the single domain instance. "plan
groups" lists the group counts selected across seeds.

| instance | n | plan groups | repo compile | SDD compile | repo query | SDD query | repo 256 batch | SDD 256 loop | repo exact query | repo cold exact (compile+query) | GANAK exact (one-shot, −launch) | repo MAP | RC2 MAP | repo 1k samples |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bipartite | 18 | 2 | 1.36 | 19.9 | 0.031 | 0.005 | 0.27 | 1.47 | 0.49 | 1.86 | 4.29 | 0.050 | 0.70 | 3.69 |
| bipartite | 22 | 2 | 1.69 | 36.0 | 0.031 | 0.006 | 0.27 | 1.61 | 0.64 | 2.28 | 4.05 | 0.055 | 0.91 | 3.71 |
| bipartite | 26 | 1, 2 | 5.19 | 87.6 | 0.041 | 0.012 | 0.54 | 3.95 | 0.92 | 5.98 | 36.7 | 0.065 | 1.25 | 3.97 |
| bipartite | 30 | 1, 2 | 15.3 | 214.0 | 0.039 | 0.010 | 0.62 | 3.49 | 1.16 | 17.0 | 28.2 | 0.066 | 1.24 | 4.07 |
| coupled | 18 | 2 | 0.97 | 7.96 | 0.030 | 0.003 | 0.20 | 0.96 | 0.45 | 1.41 | 3.19 | 0.049 | 0.65 | 3.61 |
| coupled | 22 | 2 | 1.12 | 18.2 | 0.031 | 0.005 | 0.28 | 1.43 | 0.63 | 1.71 | 7.18 | 0.054 | 0.84 | 3.80 |
| coupled | 26 | 2 | 1.26 | 21.3 | 0.031 | 0.005 | 0.25 | 1.45 | 0.80 | 2.02 | 3.50 | 0.057 | 0.97 | 3.73 |
| coupled | 30 | 2 | 1.40 | 27.5 | 0.040 | 0.006 | 0.34 | 1.84 | 1.03 | 2.42 | 8.37 | 0.064 | 1.04 | 3.88 |
| coupled | 40 | 2 | 1.82 | 63.0 | 0.031 | 0.007 | 0.38 | 2.22 | 1.66 | 3.53 | 7.88 | 0.073 | 1.55 | 4.16 |
| coupled | 50 | 2 | 7.33 | 177.2 | 0.047 | 0.019 | 1.11 | 6.43 | 3.48 | 10.8 | 42.6 | 0.090 | 2.27 | 5.26 |
| coupled | 60 | 2 | 9.02 | 349.5 | 0.056 | 0.036 | 1.63 | 11.1 | 5.24 | 14.4 | 62.1 | 0.11 | 2.73 | 6.39 |
| clinical_prescribing_standard_afib_infection | 22 | 6 | 3.74 | 14.1 | 0.072 | 0.005 | 0.83 | 1.53 | 0.81 | 4.51 | 8.51 | 0.094 | 0.29 | 11.2 |
| cloud_ops_agent_routine_canary | 20 | 6 | 4.98 | 11.2 | 0.075 | 0.007 | 1.07 | 2.04 | 0.81 | 5.81 | 26.9 | 0.098 | 0.28 | 10.6 |
| fintech_treasury_compliant_rebalance | 20 | 4 | 3.65 | 10.6 | 0.062 | 0.005 | 0.93 | 1.46 | 0.70 | 4.32 | 22.4 | 0.083 | 0.26 | 7.37 |

Checks: {'ganak_equals_repo_exact': '58/58', 'rc2_map_equals_repo': '58/58', 'repo_ok': '58/58'} | counts agree 135/135 | float64 vs exact: max marginal abs err 5.3e-15, max Z rel err 2.3e-14 | max SDD Z rel err 1.4e-14

## Repository-only comparison with the v3.4 run

The v3.4 numbers come from the earlier run of the same fixtures on the same
machine (different time and background load). "previous repo sparse" is v3.4's
contraction engine.

| rows | previous repo sparse | this run repo auto |
|---|---:|---:|
| coupled n=50 | 2.80 | 1.92 |
| coupled n=60 | 3.71 | 2.56 |
| coupled n=80 | 7.82 | 4.78 |
| coupled n=100 | 44.2 | 10.2 |
| coupled n=150 | limit | 63.1 |
| coupled n=200 | limit | 563.0 |
| coupled n=300 | limit | 11,263 |

| instance | metric | previous | this run |
|---|---|---:|---:|
| clinical_prescribing_standard_afib_infection | compile | 1.21 | 3.74 |
| clinical_prescribing_standard_afib_infection | query | 0.39 | 0.072 |
| clinical_prescribing_standard_afib_infection | 256 batch | 53.3 | 0.83 |
| clinical_prescribing_standard_afib_infection | exact query | 2.15 | 0.81 |
| clinical_prescribing_standard_afib_infection | MAP | 4.32 | 0.094 |
| cloud_ops_agent_routine_canary | compile | 1.84 | 4.98 |
| cloud_ops_agent_routine_canary | query | 0.49 | 0.075 |
| cloud_ops_agent_routine_canary | 256 batch | 80.4 | 1.07 |
| cloud_ops_agent_routine_canary | exact query | 2.86 | 0.81 |
| cloud_ops_agent_routine_canary | MAP | 6.48 | 0.098 |
| fintech_treasury_compliant_rebalance | compile | 0.92 | 3.65 |
| fintech_treasury_compliant_rebalance | query | 0.22 | 0.062 |
| fintech_treasury_compliant_rebalance | 256 batch | 8.43 | 0.93 |
| fintech_treasury_compliant_rebalance | exact query | 0.80 | 0.70 |
| fintech_treasury_compliant_rebalance | MAP | 0.69 | 0.083 |

## Reproduce

```sh
python scripts/external_bench/export.py TOOLS WORK
python scripts/external_bench/run_counting.py TOOLS WORK
python scripts/external_bench/run_weighted.py TOOLS WORK
python scripts/external_bench/final_tables.py WORK [PREVIOUS_WORK]
```

TOOLS must contain `ganak/x/ganak`, `sstd/` (SharpSAT-TD and
`flow_cutter_pace17`), `shim/timeout` and an unpacked `pysdd` wheel. They are
not repository dependencies and were obtained separately.
