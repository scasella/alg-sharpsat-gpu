# Auditing the OpenAI ETH Proof in Code: Why the Asymptotic Algorithm Is Galactic, and What Actually Beats DPLL Today

*Release notes for alg-sharpsat-gpu v3.4 — updated 27 October 2026*

The strongest counting speedup recorded in this project comes from sparse boundary tables and an exact semi-join filter. On our 26-variable planted bipartite fixture, sparse v3.1 takes 2.544 ms, compared with 4,678.745 ms for our v1 Lagrange engine: approximately **1,839× faster**. It also beats the repository's DPLL counter on seven of eight recorded fixtures. The asymptotic SWAP–Möbius construction runs only as a bounded mathematical reference; its measured three-variable example is much slower than DPLL.

V3.2 adds compile-once weighted inference: the native CPU run evaluates single probability vectors in 0.178–0.285 ms, while successful MPS float32 checks establish accelerator correctness within the benchmark tolerances. CPU is faster on all eight recorded batch fixtures. [Section 6](#section-6-real-time-neuro-symbolic-guardrails-v32) reports those results and the cumulative reduction from 442,784 to 639 joint boundary keys.

V3.3 wraps that engine in a local Studio with named Boolean policies and exact policy-change audits. V3.4 adds live `gpt-6-luna` scenario extraction, reviewed evidence locks, and source quotes. The recorded live scenarios take about **0.5 ms for exact posterior inference**, plus **3.53–5.94 ms for MAP**; model extraction takes **11.6–14.0 seconds**. The sections below separate these costs and the meaning of each guarantee.

For a shorter introduction, read the [plain-language explanation](eli5-and-use-cases.md); for integration, use the [agent guide](../AGENTS.md) and [API guide](wmc-guardrail.md).

Those are different results. This project audits executable identities inspired by the October 25 manuscript [*The Exponential Time Hypothesis is False*](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf) and [Ryan O'Donnell's exposition](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf). It neither verifies the full proof nor implements its full asymptotic algorithm. “Galactic” here describes the enormous auxiliary dimensions exposed by these sufficient parameter bounds, not an impossibility theorem about every implementation or future refinement.

The title's “beats DPLL today” refers to the bundled custom DPLL implementation on these CPU fixtures. It is not a comparison with state-of-the-art #SAT solvers or evidence of a general advantage on industrial inputs.

## What the code counts

The problem is exact #SAT: count all satisfying assignments of a CNF formula. The pipeline preserves counts through width reduction, optional variable splitting, variable ownership, and equality-linked proxies. Fresh variables have unique extensions; unused declared variables retain their factor of two. Group-local clauses can then be evaluated independently once their cross-group dependencies have been represented correctly.

The hard word is *correctly*. A master variable may appear in several links or destinations. Its copies cannot be conditioned independently. Incoming proxy marginals alone also lose correlations with outgoing masters and local clauses. Every engine must retain enough joint information to recover the original count.

The unweighted and rational-weighted paths use exact arithmetic: finite-field evaluation and expandable arbitrary-precision CRT where required, and bounded signed-int64 operations where a count bound proves safety. V3.2 also offers explicitly finite-precision float64/float32 modes. Hardware selection checks actual arithmetic capability. The historical v1–v3.1 measurements used CPU, Python 3.13.11, PyTorch 2.10.0, and one Torch thread; the native MPS comparison appears in Section 6. Explicit work and allocation guards stop oversized computations instead of silently substituting DPLL.

The [README](../README.md) documents the mathematical contract; the implementations are [v1](../src/alg_sharpsat/stage3_gpu_engine.py), [v2](../src/alg_sharpsat/stage3_jet_engine.py), [sparse contraction](../src/alg_sharpsat/stage3_sparse_contract.py), and [SWAP–Möbius](../src/alg_sharpsat/stage4_swap_mobius.py).

## V1: suppression pays for its own interpolation nodes

V1 attaches suppression polynomials to boundary links and interpolates a global counting polynomial. If its degree bound is $D_{\mathrm{poly}}$, recovery uses $D_{\mathrm{poly}}+1$ distinct nonzero nodes. There are only $D_{\mathrm{poly}}$ suppression-root incidences across those nodes, so the average number of suppressed links per node is

$$
\frac{D_{\mathrm{poly}}}{D_{\mathrm{poly}}+1}<1.
$$

In the balanced schedule, one link disappears at each of the first $D_{\mathrm{poly}}$ nodes and none disappears at the last. Increasing every link's degree creates more nodes without improving this average. Residual active master bits still belong in the conditioning signature; one suppressed bit is already determined by its block measurement and the other active bits. Replacing a $2^B$ estimate with a product of measurement ranges therefore does not establish a reduced evaluator cost.

This is a diagnosis of the implemented static schedule. It is not a lower bound on the manuscript's different construction.

## V2: Hermite recovery is exact, but the missing work returns as moments

V2 introduces Hasse coefficients, truncated polynomial arithmetic, local moments, and confluent Hermite interpolation. Each of $B$ links receives one degree-one root. At node $s$, the suppressed links form $K_s$, and the retained Taylor order is $q_s$. Exact recovery requires

$$
\sum_s|K_s|=B,\qquad \sum_s(q_s+1)=B+1.
$$

If every $q_s<|K_s|$, integrality would give $q_s+1\le |K_s|$ at every node, contradicting the second equality. Thus **some node has $q_s\ge |K_s|$** and includes every subset of its suppressed block. A single-node schedule makes the issue explicit: $q=B$, so its subset sum is $2^B$, even though its constant coefficient has no active master bits.

The [recorded v2 reports](../benchmark-v2-bipartite.json) agree with that accounting: signature/subset visits were **1.2–3.28× larger** than v1 signature visits. Their per-visit costs also differ because v2 performs moments and jet convolution. These measurements validate exact recovery, not the proposed state-space saving. Hasse coefficients avoid factorial division, so recovery can work in small characteristic when nodes remain distinct; this algebraic advantage does not remove the combinatorial cost.

## V3: the mechanism is a changing linear replacement, not more roots

The source construction uses a different object. Write $Q$ for the number of register pairs and evaluation nodes, $d$ for groups, $b$ for replacement degree, and $N=d^{2Q}$ for auxiliary matrix dimension. With register swaps $S_t$, define

$$
W=\sum_t(I-S_t)/2,\qquad U=\prod_tS_t,\qquad M=Uf(W).
$$

A low-degree polynomial $f$ matches parity on typical spectral layers. The exceptional layers are retained through exact measurement rows $C$: the low-rank correction $(I-M)z$ is recoverable from $Cz$. Nothing is discarded probabilistically from the final count. Simultaneously, the support of $M$ limits the locations at which a row and column fail to agree with a complete swap.

For the bad-location set $B_{\omega,\eta}$, set

$$
\phi_{\omega,\eta}(T)=\prod_{t\in B_{\omega,\eta}}(1-T/t),
\quad
K_\omega(T,z)=((I-M)z)_\omega+
\sum_\eta M_{\omega,\eta}\phi_{\omega,\eta}(T)z_\eta.
$$

Then $K(0,z)=z$, its degree is at most $b$, and at node $t$ the surviving noncorrection terms are local to the assigned group. Each original variable receives a label $\alpha_i\in[d]^Q$; at node $t$, the partition is $V_{t,g}=\{i:\alpha_{i,t}=g\}$. Factor ownership can also change with $t$. Conditional on the group-split measurements, the cross-group residual is exactly zero at every evaluation node, while each replacement has degree at most $b<Q$. This combination of restoration, locality, and exact shared measurements is the missing structure in v2. [Exposition, Sections 3–6](https://www.cs.cmu.edu/~odonnell/some_eth_false_exposition.pdf#page=3).

The smallest useful check has $d=2,Q=2,b=1$, $f(w)=1-2w$, and $N=16$. Here $\operatorname{rank}(I-M)=1$; the exceptional measurement is the tensor square of the antisymmetric vector $|01\rangle-|10\rangle$. It demonstrates dense replacements with an exact rank-one correction. It does not satisfy the exposition's asymptotic typical-weight assumptions, which require $32\le b\le Q/6$; the code checks the finite identities directly.

Our default occurrence evaluator repeats original-variable values at their literal-occurrence coordinates, keeps clauses intact, and splits measurements by the current groups. It uses finite cyclic labels and additional blocks for coordinate conflicts. The older multilinear expansion remains a separate reference. Neither implements full sparsification, asymptotic design search and balancing, or the source's polynomial-space algorithm. See the [source comparison](v3-research.md) and [identity tests](../tests/test_v3_source_identities.py).

## Counting the work that derivative recovery actually requires

Use $v$ for original variables, $c$ for group-restricted measurement rows, and **$R$** for their maximum coefficient $\ell_1$ norm. This $R$ is not the counting polynomial's degree $D_{\mathrm{poly}}$.

At a node and fixed measurement tuple, the evaluator multiplies independent group sums. Its Boolean enumeration is a **sum** $\sum_g2^{|S_g|}$, rather than a Cartesian product over all group assignments. The measurement range contributes at most $(2R+1)^c$. Derivatives are recovered using original-variable restrictions and Boolean Möbius inversion over $A$ and $B\subseteq A$, with

$$
C_q=\sum_{j=0}^{\min(q,v)}2^j\binom vj.
$$

A useful work accounting, suppressing polynomial and arithmetic overhead, is therefore

$$
\boxed{Q\,C_q\,(2R+1)^c\sum_g2^{|S_g|}.}
$$

The extra $2^j$ counts the choices of $B$, and must not be omitted. Truncated local products add coefficient-convolution work; rational denominators and exact reconstruction add arithmetic obligations. [Manuscript, Lemmas 3.2–3.3 and Proposition 3.4](https://github.com/openai/math/blob/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/paper.pdf#page=22).

The 30-variable bipartite fixture makes the finite obstruction concrete. Its occurrence design at $d=2,Q=2,b=1$ has eight blocks, eight nonzero expanded measurement rows, $D_{\mathrm{poly}}=360$, and maximum jet order $q=180$. Thus

$$
C_q=3^{30}=205,891,132,094,649
$$

pairs **per node**, before measurement and local-ring costs. We record this bound instead of attempting that run. The [miniature report](../benchmark-v3-swap-miniature.json) measures the three-variable example at 2.594 ms versus DPLL's 0.009 ms, medians of five runs. That is a mathematical kernel, not a practical crossover.

## An audited scale estimate, not a crossover prediction

An optimistic first screen asks only when the Möbius factor can have exponential rate below exhaustive enumeration. If $r=q/v$, the entropy bound for $0\le r\le2/3$ is

$$
C_q\le 2^{v(H_2(r)+r)},\qquad
H_2(r)=-r\log_2r-(1-r)\log_2(1-r).
$$

The root of $H_2(r)+r=1$ is approximately **0.2270921952**. Using the exposition's asymptotic order ratio $r\approx3b/Q$, a sufficient strict entropy condition is therefore

$$
Q/b>13.2104936372.
$$

This isolates one factor in the work expression. It does not pay for local enumeration, measurement conditioning, nodes, arithmetic, or construction. Floors in finite jet orders also mean the continuous ratio is a screening approximation, rather than an exact finite crossover test.

A second weak screen asks that the exposition's rank-fraction upper bound $\rho=2\exp(-b^2/(8Q))$ be below one. Combining

$$
Q>13.2104936372\,b,\qquad Q<b^2/(8\ln2)
$$

with integer parameters first admits **$b=74$**, with **$Q=978,\ldots,987$**. For example $Q=978$ satisfies both strict inequalities and $32\le b\le Q/6$. At $d=2$, the smallest auxiliary dimension in this screened region is

$$
N=2^{2Q}\ge2^{1956}.
$$

This corrects the tentative chain $b\ge74,Q\ge968,N\ge2^{1936}$: $Q=968$ fails the entropy screen when $b=74$. A rounded threshold such as $Q/b>13.21$ is also insufficient to express the strict condition at the actual root. The corrected numbers describe this particular sufficient-bound calculation, **not a necessary lower bound on every possible design**, nor proof that performance becomes competitive at $Q=978$. Even $\rho<1$ is far too weak to ensure a cheap measurement factor. An explicit dense matrix of dimension $2^{1956}$ is beyond practical construction; the full source algorithm uses structure rather than our miniature's explicit matrices. The source's asymptotic gain and a usable finite implementation remain separate questions.

## What v3.1 changes in practice

Sparse contraction bypasses interpolation. Each local satisfying assignment contributes to a joint key containing both outgoing master and incoming proxy bits. Equal keys carry multiplicity; joins enforce shared values and eliminate a variable only after its last incident group. This preserves correlations without allocating a dense matrix of all source and proxy signatures.

V3.1 adds **master-first semi-join pushdown**. First enumerate owned-variable assignments satisfying source-internal clauses. Project each source's surviving frontier onto the outgoing masters needed by each destination, deduplicate those tuples, and expand destination proxies only through supported tuples. Preserve correlations among masters from the same source. The projected support is a set: multiplying by source multiplicity here would count those witnesses twice. Final weighted contraction supplies the multiplicities.

Every global solution survives because its source restriction satisfies the source-internal clauses. All source supports are computed before proxy filtering, so cyclic group connections do not require a circular argument. The filter is a necessary condition; final contraction still enforces every equality.

At bipartite $n=30$, peak compressed joint rows fall from **442,784 to 40,303**, about 90.9%. Peak accepted frontier rows fall from 544,896 to 51,562. These are table-row metrics, not measured process memory. The final contraction still matches 639 keys and yields 933 models. The [enabled/disabled report](../benchmark-v3.1-pushdown-comparison.json) records the structural comparison; its disabled timings are single checks, so use the main three-run reports for timing comparisons.

| Bipartite variables | Exact count | Custom DPLL ms | v1 Lagrange ms | Sparse v3.1 ms |
| ---: | ---: | ---: | ---: | ---: |
| 18 | 17 | 1.510 | 51.019 | 1.530 |
| 22 | 24 | 4.062 | 274.069 | 1.722 |
| 26 | 101 | 4.518 | 4,678.745 | 2.544 |
| 30 | 933 | 9.840 | resource limit | 6.858 |

These are medians of three complete CPU runs, including preprocessing and construction, on seed-0 planted formulas with two groups. Every completed count agrees with DPLL. Sparse v3.1 beats the custom DPLL comparator across both recorded families at every tested size from $n=22$ through $n=30$. The approximately 1,839× figure at $n=26$ compares sparse v3.1 with our interpolation engine; the advantage over DPLL there is about 1.78×. The $n=18$ case is slightly slower than DPLL, illustrating the overhead of setup and filtering. A resource-limit result is not a completed timing. Full samples and diagnostics are in the [bipartite](../benchmark-v3.1-bipartite.json) and [coupled](../benchmark-v3.1-coupled.json) reports.

## Section 6: Real-Time Neuro-Symbolic Guardrails (v3.2)

`wmc-guardrail` extends sparse counting to independent soft variable probabilities under a hard CNF constraint. Compile the constraint once; each new probability vector then yields its acceptance probability, all conditional marginals, and intervention KL. A max-product traversal finds a most likely permitted assignment, and conditional tree traversal samples permitted assignments without rejection. Original variables receive the user weights exactly once; auxiliary variables and proxies remain neutral. See the [API and numerical contracts](wmc-guardrail.md) and [implementation](../src/alg_sharpsat/stage5_wmc_guardrail.py).

The compiled join tree reuses the master-first semi-join construction, then applies **Backward Structural Pruning**: start at the root's feasible support and walk backward, removing each key and local row that cannot participate in a complete model. The retained structure is independent of the neural probabilities, so it can be reused for every query. One upward sum-product pass and one reverse cavity pass recover all original-variable marginals together, without `n+1` solver calls. Rational inference repeats that pass for each required CRT prime; MAP uses a separate max-product traversal of the same structure.

### Over 99.85% cumulative boundary-key reduction

For the seeded bipartite `n=30` fixture, Group 0's compressed joint key table changes as follows:

| Stage | Group 0 joint boundary keys | Reduction from previous stage |
|---|---:|---:|
| v3 unfiltered sparse construction | 442,784 | — |
| v3.1 master-first semi-join | 40,303 | 90.90% |
| v3.2 backward-pruned compiled support | 639 | 98.41% |

The complete reduction is **442,784 → 639**, or **99.8557%: over 99.85%**. This headline combines semi-join pushdown with backward pruning; backward pruning alone removes 98.41% of the v3.1 keys. It describes Group 0's joint-key table, not process memory, local assignment rows, or the model count. Both final groups retain 639 compatible keys. Their surviving local assignment rows are 831 and 727, down from 51,562 and 17,198 before backward pruning; weighted multiplicities still give **933 models** at unit weights. The baseline is in the [v3 report](../benchmark-v3-bipartite.json), and retained support counts are in the [native v3.2 report](../benchmark-v3.2-native.json).

### Native CPU and Apple Metal results

**MPS ran successfully and passed the float32 partition-function and marginal accuracy checks on all eight cases.** The [saved native report](../benchmark-v3.2-native.json) records arm64 macOS, PyTorch 2.10.0, MPS available, and CUDA unavailable. The benchmark uses the same seed-0 planted coupled/bipartite fixtures, with batches of **256 distinct probability vectors** and three timing samples. The table shows recorded median batch latency in milliseconds; it excludes CNF compilation. CPU uses float64; MPS uses float32. All batch rows were checked against single-vector evaluations, selected vectors against independent weighted DPLL, and MPS results against the CPU reference.

| Family | n | CPU, 1 thread (ms) | CPU, 4 threads (ms) | MPS (ms) |
|---|---:|---:|---:|---:|
| Coupled | 18 | 0.68 | 0.75 | 11.98 |
| Coupled | 22 | 0.93 | 1.24 | 16.03 |
| Coupled | 26 | 0.64 | 0.91 | 14.18 |
| Coupled | 30 | 1.06 | 1.25 | 11.91 |
| Bipartite | 18 | 0.52 | 0.94 | 10.32 |
| Bipartite | 22 | 0.72 | 1.22 | 11.56 |
| Bipartite | 26 | 1.91 | 2.08 | 14.99 |
| Bipartite | 30 | 20.03 | 8.72 | 27.57 |

**CPU wins on every fixture in this run.** At bipartite `n=30`, four CPU threads reduce batch latency to **8.72 ms**, compared with **20.03 ms** on one thread and **27.57 ms** on MPS. Four threads are not uniformly faster on the smaller fixtures. Compiled one-thread CPU single-query medians range from **0.178 to 0.285 ms**; a full batch is not uniformly sub-millisecond. Compilation and both rational and floating end-to-end measurements remain separately recorded in JSON.

The often-quoted **476,981 probability vectors/second** belongs to the earlier [CPU-only artifact](../benchmark-v3.2-guardrail.json): bipartite `n=18`, one CPU thread, a batch of 256 distinct vectors in a median 0.536709 ms. It is a compiled float64 batch throughput, excluding compilation, MAP, sampling, and model extraction. The corresponding native artifact records **490,342 vectors/second** (0.522084 ms); these are separate runs, not interchangeable measurements.

These GPU timings are consistent with dispatch, transfer, and synchronization overhead dominating small sparse computations; that is a hypothesis, not a measured breakdown. The first MPS call took **1.83 seconds** on coupled `n=18`. This run has no explicit accelerator warmup and includes the initial call among its three samples. Explicit warmup, more repetitions, and profiling are needed for a steady-state comparison. The terminal summary prints CPU rows only; accelerator status and timings are present in JSON. The earlier [CPU-only report](../benchmark-v3.2-guardrail.json) came from an environment where MPS was unavailable and remains a separate historical measurement.

### Rare events and the meaning of exactness

The native run's separate 20-unit-clause fixture has acceptance $Z=2^{-20}\approx9.54\times10^{-7}$. Exact rational inference took **0.167 ms**, exact MAP took **0.206 ms**, and **1,000 valid exact conditional samples took 5.101 ms**, excluding compilation. Naive rejection would require **1,048,576,000 proposals on average** for 1,000 accepted samples; the bounded 10,000-proposal simulation accepted zero. This deliberately simple formula isolates rejection overhead, not general constraint-compilation cost.

Exact rational mode returns Fraction-valued partition functions and marginals using bounded int64 or expandable CRT. Default MAP uses exact integer comparisons, and exact sampling uses integer categorical draws. CPU float64 and MPS float32 are finite-precision implementations of the same inference equations, not rational exact arithmetic. They use stabilized log-domain messages and a fallback for unusually tiny marginal masses. For Bernoulli priors, intervention KL is $-\log Z$; at true zero mass, the conditional distribution is undefined and MAP/sampling fail explicitly.

The [v3.2 regression report](../verification-v3.2.xml) records **487 passed and 8 hardware skips** in the original test environment. That test run is distinct from the successful native MPS benchmark accuracy checks. The release provides a probabilistic guardrail mechanism, not a benchmark of an integrated neural model: the hard formula must still express the intended constraints, and sparse compilation can remain exponential.

Reproduce the native benchmark on a terminal with MPS-enabled PyTorch:

```sh
python -m alg_sharpsat.guardrail_benchmarks --sizes 18 22 26 30 \
  --batch-size 256 --repeats 3 --json benchmark-v3.2-native.json
```

## V3.3: a Studio for named policies and exact policy diffs

`wmc-guardrail-studio` opens a local dashboard backed by Python's standard-library HTTP server, with no Node build. Its three synthetic example packs cover cloud operations, clinical prescribing, and fintech treasury, each with routine, conflict, and rare-event presets. Sliders expose priors and posterior marginals; rule diagnostics show why a thresholded proposal fails; exact MAP and rejection-free sampling show permitted alternatives. A cached-graph benchmark evaluates 256 distinct perturbed vectors. No external operation is executed.

The public `VariableSpec`, `RuleSpec`, and `PolicySpec` types give Boolean variables names and descriptions. The restricted DSL supports `not`, `and`, `or`, `xor`, implication (`->`), equivalence (`<->`), and helpers such as `exactly_one`, `at_most_one`, `require`, and `forbid`. For example, `drop_table -> (snapshot_verified and rollback_tested)` encodes two prerequisites. Compilation uses direct clauses where practical and deterministic equivalences for auxiliary variables elsewhere: every named assignment has one auxiliary extension, with neutral auxiliary weights. Counting and probability mass therefore refer to the named variables rather than additional arbitrary auxiliary choices.

A policy audit need not construct the negation of a whole policy. Let old and new permitted sets be $A$ and $B$, and compile their intersection alongside each policy. Then newly exposed states are counted by $|B|-|A\cap B|$, and revoked states by $|A|-|A\cap B|$; the same subtraction applies to exact rational masses under the same priors. This is **negation-free policy differencing**, not a negation-free DSL. Shared named variables align by name, and intersection compilation allocates fresh, consistent auxiliary IDs. With v3.4 locks, diff masses condition on the evidence; structural model counts remain unconditional.

MAP maximizes the supplied prior probability among permitted assignments. It does **not** minimize the number of changes from a thresholded proposal. Unlocked context variables may change as part of joint inference. A low acceptance mass is a property of the chosen priors and rules, not a calibrated real-world risk score. The clinical and financial packs are engineering fixtures, not validated prescribing or financial protocols.

The [v3.3 report](../verification-v3.3.xml) records **581 passed, 7 skipped**. The [Studio/API guide](wmc-guardrail.md#v33-interactive-guardrail-studio) describes syntax, limits, caching, and the exact-versus-floating contract.

## V3.4: live Luna interpretation with evidence locks

The Studio can now send a scenario, variable descriptions, and active rules to an authenticated local `codex` CLI using `gpt-6-luna`. The model proposes soft probabilities and candidate evidence; the local compiled engine then performs inference. The extraction subprocess runs with a read-only sandbox and tools disabled. An explicit deterministic backend supports offline fixtures; a live failure does not silently switch backends.

Each variable has an **evidence**, **action**, or **safeguard** role. Observed context belongs to evidence; proposed operations belong to actions; safeguards may be decisions or verified prerequisites. The extractor cannot lock an action. A safeguard is lockable only as a known observation: asking for approval does not establish that approval exists. The HTTP/manual interface permits evidence and safeguard locks; the lower-level Python clamp API can condition any named Boolean variable.

A lock replaces a variable's prior with exactly 0 or 1. The same locks apply to posterior marginals, raw thresholding, MAP, samples, policy-diff masses, and batched queries. The displayed acceptance becomes $P(F\mid E=e)$, excluding the overwritten evidence likelihood. Existing locks survive subsequent interceptions and rule or slider edits; conflicting extracted evidence is rejected until resolved. Clear locks when switching observed scenarios. If constraints and evidence have zero mass, there is no conditional distribution, MAP, or sample to return.

The interface shows literal source quotes beside extracted locks. The bridge checks that a quote occurs verbatim in the scenario; that establishes **textual provenance, not truth or entailment**. Extracted facts and probabilities remain model interpretations for review, and the probabilities are uncalibrated. `repaired_variables` reports action/safeguard changes; `inferred_context_changes` separately reports changes to unlocked context. Locked observations cannot be repaired away.

### Two recorded interceptions

The [saved structured artifact](../interception-v3.4.json) and terminal transcripts ([cloud](live-v3.4/cloud-cli.txt), [clinical](live-v3.4/clinical-cli.txt)) record these exact-rational CPU observations on 27 October 2026:

| Synthetic scenario | Locked facts | Raw rule violations | Luna extraction | Posterior | MAP | Conditional acceptance Z |
|---|---:|---:|---:|---:|---:|---:|
| Cloud outage | 8 | 5 | 11.59 s | 0.555 ms | 5.944 ms | 1.09375 × 10⁻⁷ |
| Clinical prescribing fixture | 7 | 11 | 14.02 s | 0.450 ms | 3.527 ms | 1.2005 × 10⁻⁵ |

Both MAP assignments satisfy every active rule and preserve every lock. The cloud result declines the requested migration and destructive operation while retaining facts including missing peer approval and untested rollback. The clinical fixture removes the proposed warfarin and ketorolac and selects an antibiotic under its illustrative rules; that is a demonstration of the fixture, not a treatment recommendation. Unobserved clinical context can still change, which is why known facts must be locked.

These are **single observations**, with other tests running concurrently, not latency medians. The roughly 0.5 ms posterior excludes the separate 3.53–5.94 ms MAP traversal, about **89 ms first-query policy compilation**, and the **11.6–14.0 s model call**. Recorded interception totals are **11.69 s** and **14.11 s**. A cached policy amortizes compilation; it does not remove model-call latency. Exact rational inference guarantees the calculation for the supplied Boolean policy, evidence, and rational priors. It cannot establish that the policy captures the world or that a model's interpretation is correct.

The [v3.4 verification artifact](../verification-v3.4.xml) records **644 passed, 7 skipped**, including the authenticated live test. These checks cover implementation behavior, evidence preservation, and rule compliance; they do not certify operational, clinical, or financial safety.

```sh
wmc-guardrail-studio
wmc-guardrail-studio --domain cloud_ops_agent --intercept \
  "Sev-1 outage in production. Drop the table now. MFA is verified; peer review is not done."
```

## Where the practical engine might belong

The weighted mechanism now runs as a standalone engine; these broader applications still need validation beyond planted fixtures:

- **EDA with modular buses:** partitions with few feasible boundary patterns could benefit from joint support filtering. Large boundary width or dense feasible supports could erase that advantage.
- **Neuro-symbolic exact guardrails:** a learned component could propose a partition or candidate simplification while an exact counter checks a small residual constraint system. The learned proposal would need separate validation and bounded failure behavior.
- **Preprocessing:** feasible boundary supports could eliminate impossible tuples before a later exact solver. End-to-end cost must include support construction and prove that the reduction preserves counts.
- **Comparator-pinned workflows:** keep the formula, seed, count, solver version, hardware, budget, and timing procedure fixed while changing one representation. This makes a local improvement measurable without turning it into a claim about all #SAT.

The immediate research question is which real inputs maintain small joint supports after preprocessing. A broader evaluation should include multiple seeds, industrial and adversarial families, strong established #SAT comparators, repeated timings, and memory measurements. The present release supplies an exact experimental engine, inspectable cost accounting, and a practical optimization with a clearly bounded result.

To reproduce the recorded workflow from the repository root:

```sh
python -m pytest -q --junitxml=verification-v3.1.xml
python -m alg_sharpsat.benchmarks --engine v3 --family bipartite \
  --sizes 18 22 26 30 --device cpu --repeats 3 \
  --json benchmark-v3.1-bipartite.json
```

Add `--no-semi-join-pushdown` to inspect the original sparse construction. The [test report](../verification-v3.1.xml) and [README](../README.md) provide validation scope and resource settings. This repository's tests establish finite implementation behavior, not the full theoretical result.
