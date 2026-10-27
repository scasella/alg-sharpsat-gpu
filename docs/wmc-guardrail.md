# wmc-guardrail and Guardrail Studio v3.4

[New to the project? Start with the ELI5 and use-case recipes](eli5-and-use-cases.md).
For implementation changes, use the [agent guide](../AGENTS.md); for the
version-by-version results, read the [technical blog](blog-post.md).

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

The rare-event showcase is a deliberately simple 20-unit-clause formula with uniform priors and acceptance `2^-20`. It isolates rejection overhead; it is not evidence that every rare-event constraint compiles cheaply. The report bounds rejection simulation to 10,000 proposals and records expected work analytically. Sparse compilation and join width can still grow exponentially. The v3.2 benchmark does not measure neural-model behavior or establish safety beyond the supplied CNF; v3.4 records extraction latency separately.

## V3.3: interactive Guardrail Studio

The Studio adds named variables, a restricted Boolean language, exact policy diffs,
and a local browser interface. It uses the existing compiled sparse weighted
engine. There is no Node build, CDN, or additional web framework. Manual policy
inference runs locally; v3.4's optional live interceptor calls the model through
Codex. HTML, CSS, and JavaScript are packaged in the wheel.

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


## V3.4: live interception and evidence locks

The interceptor translates a scenario into candidate evidence and soft priors,
then evaluates the current policy using the same compiled sparse support. It
uses **`codex exec -m gpt-6-luna` by default**, with local CLI authentication.
This is a live model call: the scenario, variable metadata, and enabled rules
are sent to OpenAI. Exact inference runs on the local CPU. No database, cloud,
prescription, transaction, or other operational action is executed.

### Roles and conditional inference

`VariableSpec` keeps its UI `category` and adds `role="evidence"`, `"action"`, or
`"safeguard"`. Explicit roles take precedence. For legacy/custom specs,
`context`, `evidence`, and `patient_state` categories infer evidence;
`control`, `controls`, `safeguard`, and `safeguards` infer safeguard; other
categories infer action. Every built-in variable has an explicit role.

Evidence covers observed environment or patient facts. Safeguards can be
controllable decisions or verified preconditions; lock a safeguard only when
its status is known. A request to obtain approval does not establish that
approval already exists. Actions stay soft in the extractor and cannot be
locked by its output. HTTP/manual locks accept evidence and safeguards; the
lower-level Python `clamped_vars` API permits any named Boolean observation.

For a clamp `e=v`, the weight pair becomes `(1,0)` or `(0,1)`, regardless of
the original prior. Thus the reported partition is
`Z = P(F | E=e)`, not `P(F and E=e)`. The overwritten evidence likelihood is
not multiplied into Z. Original inputs remain unchanged, auxiliary weights
remain neutral, and one common clamp mapping can be applied across a batch.

```python
from fractions import Fraction
from alg_sharpsat import PolicySpec, VariableSpec, RuleSpec

policy = PolicySpec([
    VariableSpec("prod", "Production environment", role="evidence"),
    VariableSpec("drop", "Proposed destructive operation", role="action"),
    VariableSpec("snapshot", "Snapshot verified", role="safeguard"),
], [RuleSpec("recover", "Production drops need a snapshot", "(prod and drop) -> snapshot")])
compiled = policy.compile_guardrail()
probs = {"prod": Fraction(1, 10), "drop": Fraction(9, 10), "snapshot": Fraction(1, 5)}
evidence = {"prod": 1, "snapshot": 0}
result = compiled.evaluate(probs, mode="exact_rational", clamped_vars=evidence)
assert result.partition_function == Fraction(1, 10)
assert result.posterior_marginals == [Fraction(1), Fraction(0), Fraction(0)]
action, weight = compiled.map_inference(probs, clamped_vars=evidence)
assert action == (1, 0, 0)
assert weight == Fraction(1, 10)
samples = compiled.sample_conditional(probs, num_samples=5, clamped_vars=evidence)
assert (samples == [1, 0, 0]).all()
```

Locks constrain raw threshold actions, posterior marginals, MAP, and conditional
samples consistently. `repaired_variables` lists controllable action/safeguard
changes only. `inferred_context_changes` separately describes MAP assignments
to **unobserved** context; these are inferences, not repairs to the world. If
there is no positive-weight satisfying extension, the API returns Z=0,
null MAP/marginals/KL, empty samples, and a diagnostic; the Python MAP/sampling
methods raise `ValueError`. This may arise from inconsistent rules/evidence or
from zero-weight priors, so it is not always structural unsatisfiability.

### Live CLI, structured extraction, and explicit offline mode

The installed CLI used for verification was **Codex 0.162.1**. Before
implementation, `codex exec --help` was inspected and a one-line JSON smoke
request to `gpt-6-luna` succeeded. The bridge uses an isolated temporary working
directory, stdin for the prompt, `--sandbox read-only`, `--ephemeral`,
`--skip-git-repo-check`, `--ignore-user-config`, `--ignore-rules`,
`--output-schema`, `--output-last-message`, and `--color never`.
Tool/plugin/app features and web search are disabled; MCP configuration is empty.
Subprocess arguments are passed as a list without a shell. Temporary structured
responses are bounded and removed, and CLI diagnostics are discarded rather
than exposing authentication details to HTTP callers. An incompatible CLI,
authentication failure, timeout, or invalid response fails explicitly.

The JSON schema requires probabilities for every variable and null placeholders
for absent evidence. Action clamps are restricted to null in the generation
schema. The bridge removes null placeholders and rejects unknown names,
nonfinite/out-of-range probabilities, nonbinary clamps, invalid roles, and
missing or nonliteral provenance. Soft probabilities must be in
`[0.001, 0.999]`; hard evidence is applied separately as exact endpoints.
The prompt asks for candidate facts and a short rationale, with enabled rules
as context. Source quotes are checked as literal substrings, which proves
textual provenance **only**, not entailment or truth. The output remains a
model interpretation for review.

Timeout is 30 seconds by default, configurable with `--timeout` or the API
`timeout` field up to 120 seconds. No silent retry or deterministic substitution
conceals live errors. `backend="deterministic"` explicitly chooses a small
keyword/status parser for tests and offline demonstrations; it does not emulate
a general language model. Explicit `variable=true/false` statements and a narrow
set of observed-status aliases can generate evidence; ambiguous/hypothetical
or conflicting statements remain uncertain.

```sh
wmc-guardrail-studio --domain clinical_prescribing --mode exact_rational \
  --intercept "The patient has severe renal impairment and an acute bacterial infection. The agent proposes ketorolac without an antibiotic."
wmc-guardrail-studio --domain fintech_treasury --backend deterministic \
  --intercept "circuit_breaker_tripped=true; smart_contract_bridge_tx=true"
```

The CLI reports backend/model, rationale, evidence, raw threshold tuple,
violated rules, Z/KL, the full MAP tuple, controllable changes, separate
unobserved-context inference, all prior-to-posterior shifts, and timings.
Changing a MAP safeguard flag proposes a condition to satisfy; it does not
perform verification or obtain an approval.

### Studio and REST integration

The top panel has three editable sample prompts per domain, live/offline backend
selection, an in-flight status, extracted source quotes, and separate model and
posterior/MAP timings. Evidence/safeguard sliders cycle through **Unclamped →
Clamped 0 → Clamped 1 → Unclamped**. A lock disables its slider. Preset/domain
changes clear locks; normal slider and rule edits preserve them. The **Clear
evidence locks** button supports switching to a different observed scenario.
Existing locks are merged into new extractions; conflicting values cause an
error rather than overwriting a known fact. Responses from stale in-flight UI
requests are not applied after inputs change.

```sh
curl -s http://127.0.0.1:8765/api/intercept \
  -H 'Content-Type: application/json' \
  -d '{"domain":"cloud_ops_agent", "mode":"exact_rational",
       "scenario_text":"Production outage. The agent proposes dropping a table. MFA is verified, but peer review is not done.",
       "clamped_evidence":{"prod_environment":1}, "num_samples":5}'
```

`POST /api/intercept` takes the same domain/custom policy and inference options
as `/api/evaluate`, plus `scenario_text`, `backend` (default `codex`), and
`timeout`. It validates policy limits and inference options before invoking the
external process, runs extraction outside the engine lock, then calls cached
inference. Response fields include the normal evaluation payload plus `model`,
`backend`, `llm_ms`, `rationale`, `clamped_evidence`, `evidence_sources`,
`extracted_probs`, `extraction`, and `intercept_total_ms`. `prior_probs` contains
effective clamped priors; `extracted_probs` retains the soft estimates.

`POST /api/evaluate` accepts `clamped_evidence` (or the `clamped_vars` alias;
conflicting aliases are rejected). Diff and batch endpoints also preserve locks.
Policy-diff **masses** are conditional on evidence, while model **counts** and
structural-pruning telemetry remain unconditional policy structure. The UI
labels that distinction. This release does not change the cached CNF for each
observation. A batch whose every variable is clamped contains equivalent
evaluations even though its underlying soft vectors were perturbed.

`eval_ms` measures posterior computation; `map_ms` and `sample_ms` are separate.
`total_ms` covers local policy evaluation, and `intercept_total_ms` additionally
covers extraction. Wall time also includes Python startup and HTTP/rendering.
These are distinct from the prior v3.2 GPU benchmark timings. Neural probabilities
are uncalibrated estimates, not measured operational risks.

### Recorded v3.4 verification

The final documentation-release recheck records [643 passed, 8 skipped
offline](../verification-v3.4-release.xml) and [63 passed with live g18
enabled](../verification-v3.4-live.xml). These cover 644 distinct passing tests;
the live check replaces its offline skip. The seven hardware skips are unchanged.

The complete suite ran with `WMC_LIVE_CODEX=1`: **644 passed, 7 skipped**
(63 new g18 tests). All previous tests remain covered. Native MPS floating-point
inference passed; unavailable CUDA and unsupported exact-integer MPS kernels
account for the seven skips. Without opt-in, the authenticated model test is
skipped so ordinary CI never invokes a paid/authenticated model.

```sh
python -m pytest -q
WMC_LIVE_CODEX=1 python -m pytest -q -m g18
```

The g18 suite covers constrained-policy Hypothesis enumeration, each domain's
presets, exact and floating clamp invariants, immutable batch inputs, zero mass,
role eligibility, missing/invalid JSON, unknown variables, quoted provenance,
hypothetical/conflicting offline statements, subprocess launch isolation and
timeouts, HTTP/CLI integration, request validation before model invocation,
manual-lock conflicts, and cached evaluation.

Two live CLI commands ran on 2026-10-27 against `gpt-6-luna` with
`--mode exact_rational`. Their saved MAP tuples were independently checked
against every domain rule and every extracted clamp:

| Live scenario | Locked facts | Raw rule violations | Extraction | Exact posterior | Exact MAP | Conditional Z |
|---|---:|---:|---:|---:|---:|---:|
| Cloud outage | 8 | 5 | 11.59 s | 0.56 ms | 5.94 ms | 1.09375e-07 |
| Clinical fixture | 7 | 11 | 14.02 s | 0.45 ms | 3.53 ms | 1.2005e-05 |

The cloud scenario requested a table drop and migration during a production
incident, with no reviewed window, no breakglass ticket, and no tested rollback.
The model supplied eight locks. The exact MAP suppressed both requested database
operations while retaining production=1, incident=1, window=0, peer review=0,
breakglass=0, MFA=1, snapshot=1, and tested rollback=0.

```text
Evidence: prod_environment=1, change_window_open=0, incident_sev1_active=1,
          mfa_verified=1, peer_review_approved=0, emergency_breakglass_ticket=0,
          fresh_snapshot_verified=1, rollback_script_tested=0
run_schema_migration: 0.999 -> posterior 0; MAP 1 -> 0
drop_or_truncate_table: 0.999 -> posterior 0; MAP 1 -> 0
Z: 1.09375e-7; locked values preserved; MAP violations: 0
```

The clinical fixture described renal impairment, ulcer history, AFib, and an
infection, while proposing warfarin/ketorolac without an antibiotic and stating
that INR/renal panels had not been ordered. Seven locks were retained. Under
these illustrative rules the MAP suppressed warfarin/ketorolac and selected a
modeled antibiotic. Unobserved risk flags remained uncertain and are listed
separately from action repairs. This is a software fixture, not a prescription.

```text
Evidence: severe_renal_impairment=1, history_peptic_ulcer=1,
          pregnant_or_lactating=0, atrial_fibrillation=1,
          acute_bacterial_infection=1, order_inr_monitoring=0,
          order_renal_panel_stat=0
rx_warfarin: 0.96 -> posterior 0; MAP 1 -> 0
rx_nsaid_ketorolac: 0.96 -> posterior 0; MAP 1 -> 0
Z: 1.2005e-5; locked values preserved; MAP violations: 0
```

[Cloud transcript](live-v3.4/cloud-cli.txt) ·
[Clinical transcript](live-v3.4/clinical-cli.txt) ·
[Structured receipt](../interception-v3.4.json) ·
[Full regression report](../verification-v3.4.xml).

These timings are single observations, not performance medians; the CLI runs
and regression suite overlapped. First compilations were approximately 89 ms;
local evaluation totals were 96.92 ms and 94.50 ms, respectively. Subsequent
queries reuse support. End-to-end interception totals were 11.69 s and 14.11 s.
Model estimates and wording can vary on repetition. During development,
paraphrased quotes and attempts to clamp an action were correctly rejected;
verbatim instructions and role-constrained output schema were then tightened.
No failed output was silently repaired or replaced with the offline parser.

The live browser flow was also exercised: a cloud sample completed Codex
extraction, rendered source quotes and timings, produced a compliant MAP, and
retained manual locks while switching inference modes and editing priors.
