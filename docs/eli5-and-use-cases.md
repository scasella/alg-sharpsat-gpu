# ELI5: from counting switches to practical guardrail recipes

## Part 1: the two-minute explanation

Thirty on/off switches already have over a billion settings. **SAT** asks for
one setting obeying all the rules; **#SAT** asks how many obey them; **weighted
model counting (WMC)** gives settings different odds and adds the odds of the
permitted ones. With 100 switches, even checking a billion settings per second
would take about 40 trillion years—roughly 2,900 universe ages.

Inspired by the October 2026 OpenAI ETH manuscript and Ryan O’Donnell’s
exposition, the project follows four stages:

1. **Cut the problem into zones.** A rule crossing a zone boundary is like a
   wire between rooms. Replace those wires with polynomial expressions: at
   selected dial settings, the rooms can be evaluated separately. Reconstruct
   the polynomial's value at zero to recover the original count.
2. **Find the catch, then test a richer mechanism.** In v1, keeping the same
   zones means cutting more wires also raises polynomial degree: the work
   reappears in reconstruction. V2 packs several coefficients into each dial
   setting, but this static degree-conservation barrier remains. Stage4 tests
   SWAP–Möbius replacements with changing zones and shared measurements. Its
   tiny `N=16` example needs one exceptional measurement (a rank-one correction)
   and verifies locality at **both** evaluation nodes. This is a miniature
   executable audit, not a proof of the full manuscript.
3. **Build a practical sparse funnel.** Each zone first solves its own switches
   (Phase-1 Owned Frontier), shares which boundary settings are possible
   (Master-First Semi-Join Pushdown), then removes pieces that cannot join a
   complete solution (Backward Intersection Pruning). One fixture's joint-key table falls
   `442,784 → 40,303 → 639`. Compile that structure once; attach new probabilities
   to obtain permitted mass, conditional odds, or permitted samples.
4. **Connect language to those explicit rules.** V3.4 uses `gpt-6-luna` through
   `codex exec` to extract candidate facts
   and probabilities from a scenario. Review evidence before locking it: a
   known fact stays fixed; an unknown fact stays uncertain. A source quote
   proves where a claim came from, not whether it is true. MAP chooses the
   permitted assignment with highest prior weight, not the fewest flipped bits.

The finite-size sufficient screen first admits `b=74, Q=978`, giving
`N=2**1956` for `d=2`; the older `Q=968, N=2**1936` fails that screen.
This is neither a universal lower bound nor a guaranteed crossover. The sparse
funnel removes **99.8557% cumulatively**, **98.4145% in the backward pass alone**;
these are joint table keys, not process-memory measurements.

The cloud example produced a rule-compliant MAP tuple with every lock intact,
even though its supplied prior assigned only about one chance in nine million
to compliance. The recorded live examples took 11.59–14.02 seconds for LLM extraction, about
0.45–0.56 ms for the posterior alone, 3.53–5.94 ms for MAP, and about 89 ms for
first compilation. These are individual observations, not service guarantees.
The historical CPU batch fixture's 476,981 vectors/second is likewise a fixture result,
not an SLA. Engineering examples in cloud, prescribing, and treasury are not
certified clinical or financial production systems. Compilation can still grow
exponentially; a GPU is not automatically faster.

See the [technical account](blog-post.md), [API contracts](wmc-guardrail.md),
[research notes](v3-research.md), [live records](../interception-v3.4.json), and
[historical CPU batch report](../benchmark-v3.2-guardrail.json) for evidence and scope.

## Part 2: four practical use cases

From the repository's `alg-sharpsat-gpu` directory, install once with
`python -m pip install -e .`. Each Python block below is independent: save a
block as a `.py` file and run it with that environment's Python. No model or
network access is needed by default.

### 1. Cloud operations: language extraction plus locked evidence

Use this pattern to prevent a proposed cloud action from changing an observed
outage flag just to satisfy a rule. The default extractor is an explicit offline
deterministic parser, useful for reproducible integration tests. It is not an
LLM simulation. Set `LIVE_CODEX=1` to explicitly opt into the actual Codex Luna
extractor; that requires an authenticated `codex` CLI on `PATH`, invokes a model,
and may incur usage. Live failures raise rather than silently falling back.

```python
import os
from alg_sharpsat import VariableSpec, RuleSpec, PolicySpec, CompiledPolicy
from alg_sharpsat.llm_bridge import extract

policy = PolicySpec([
    VariableSpec("outage", "Observed outage", role="evidence"),
    VariableSpec("restart", "Restart service", default_prob="9/10", role="action"),
    VariableSpec("backup", "Verified backup", role="safeguard"),
], [RuleSpec("restart_gate", "Restart requires backup during outage",
             "restart implies (not outage or backup)")])
scenario = "outage=true. backup=false. The operator proposes a restart."
backend = "codex" if os.environ.get("LIVE_CODEX") == "1" else "deterministic"
candidate = extract(policy, scenario, backend=backend)
print("Candidate source quotes:", candidate["evidence_sources"])
# Application review step: these explicit fixture facts have been verified.
# Never promote arbitrary model output to trusted facts automatically.
locks = {"outage": 1, "backup": 0}
engine = CompiledPolicy(policy)
result = engine.evaluate(candidate["probs"], mode="exact_rational",
                         clamped_vars=locks)
action, weight = engine.map_inference(candidate["probs"], clamped_vars=locks)
assert action == (1, 0, 0)
assert result.posterior_marginals[0] == 1
assert result.posterior_marginals[2] == 0
assert not policy.check_violations(dict(zip(policy.variable_names, action)))
print("Permitted MAP:", dict(zip(policy.variable_names, action)))
print("Conditional mass:", result.partition_function)
```

Here `backup=false` is known unavailable, not merely unmentioned. If its status
were unknown, omit that lock and retain its soft prior. Library clamps accept
any named variable; applications must enforce the intended evidence/safeguard
boundary and verification process themselves. The extractor rejects action
clamps. `Z` here is the permitted mass after the locks are applied, not the
probability of observing those locked facts.

### 2. Clinical engineering fixture: exact rare-event odds and samples

Use exact fractions to audit a deliberately simplified contraindication fixture:
ten contraindication flags must all be false before this treatment is permitted.
Each flag independently has prior probability `9/10`, making the permitted event
rare: `Z=10**-10`. This fixture isolates the arithmetic and sampling mechanism;
it is not a prescribing protocol. Conditional sampling traverses the compiled
tree directly, without proposing and rejecting about ten billion prior draws.

```python
from fractions import Fraction
from alg_sharpsat import VariableSpec, RuleSpec, PolicySpec, CompiledPolicy

names = [f"contraindication_{i}" for i in range(10)]
policy = PolicySpec([VariableSpec(n, "Illustrative contraindication", role="evidence")
                     for n in names], [
    RuleSpec(f"exclude_{i}", "Treatment eligibility excludes this flag", f"not {n}")
    for i, n in enumerate(names)
])
engine = CompiledPolicy(policy)
p = [Fraction(9, 10)] * len(names)
r = engine.evaluate(p, mode="exact_rational")
assert r.partition_function == Fraction(1, 10**10)
assert r.posterior_marginals == [Fraction(0)] * len(names)
samples = engine.sample_conditional(p, num_samples=1000, seed=7)
assert samples.shape == (1000, 10)
assert all(not policy.check_violations(dict(zip(names, row))) for row in samples)
print("Z:", r.partition_function, "marginals:", r.posterior_marginals)
print("Validated samples without rejection:", len(samples))
```

A zero-mass policy has no conditional distribution: MAP and sampling raise.
Fractions preserve the arithmetic of these supplied priors.

### 3. Cloud IAM and firewall blast radius: review a policy change before rollout

Compare a public-ingress policy requiring approval with one additionally
requiring an active firewall shield. The same pattern applies to IAM access
grants or financial approval rules. Counts show the logical change; mass shows its effect
under this particular independent prior. Neither is an empirical incident rate.
For old allowed set `A` and new allowed set `B`, the combined rules give
`A ∩ B`; newly exposed settings are `B \ A`, revoked settings are `A \ B`.
Compute `newly_exposed_count = new_allowed_count - intersection_count` and
`revoked_count = old_allowed_count - intersection_count`. The same subtraction
holds for probability mass under a shared prior. Combining old and new rules
constructs the intersection directly, avoiding negation of an entire CNF and
its additional compilation work.

```python
from fractions import Fraction
from alg_sharpsat import VariableSpec, RuleSpec, PolicySpec, diff_policies

variables = [VariableSpec(n, n) for n in ("public_ingress", "approved", "shield")]
old = PolicySpec(variables, [
    RuleSpec("approval", "Approval needed", "public_ingress implies approved")])
new = PolicySpec(variables, [
    RuleSpec("approval", "Approval needed", "public_ingress implies approved"),
    RuleSpec("shield", "Firewall shield needed", "public_ingress implies shield")])
d = diff_policies(old, new, {"public_ingress": Fraction(1, 2),
                           "approved": Fraction(3, 4), "shield": Fraction(1, 4)})
assert (d.old_allowed_count, d.new_allowed_count) == (6, 5)
assert (d.newly_exposed_count, d.revoked_count) == (0, 1)
assert d.revoked_prob == Fraction(9, 32)
print("Newly exposed:", d.newly_exposed_count, "revoked:", d.revoked_count)
print("Revoked prior mass:", d.revoked_prob)
```

The two policies must name the same variable set. This comparison recompiles
both policies and their intersection; it is not a promise of constant-time diffs.

### 4. Batch risk sweeps: compile once, evaluate 256 different priors

Use a batch for RL/neural policy calibration, scenario sweeps, or many requests
sharing one policy. `CompiledPolicy` reuses a cached `CompiledGuardrail`;
raw-CNF callers can use `compile_guardrail` directly.
CPU float64 is the reference below. MPS/CUDA use float32 and are checked against
it with a tolerance. Device availability is reported explicitly; unsupported
accelerator operations fail explicitly and are printed as such.

```python
import numpy as np
import torch
from alg_sharpsat import VariableSpec, RuleSpec, PolicySpec, CompiledPolicy

policy = PolicySpec([VariableSpec(n, n) for n in ("a", "b", "c")], [
    RuleSpec("coverage", "Coverage", "a or b"),
    RuleSpec("dependency", "Dependency", "b implies c")])
engine = CompiledPolicy(policy)  # Only one structural compilation.
batch = np.random.default_rng(7).uniform(.05, .95, (256, 3))
reference = engine.evaluate(batch, mode="float64", device="cpu")
expected = reference.posterior_marginals.detach().cpu().numpy()
assert expected.shape == (256, 3)
for index in (0, 127, 255):
    single = engine.evaluate(batch[index], mode="float64", device="cpu")
    np.testing.assert_allclose(single.posterior_marginals.cpu().numpy(), expected[index])
print("cpu: validated 256-vector float64 batch")
for device, available in (("mps", torch.backends.mps.is_available()),
                          ("cuda", torch.cuda.is_available())):
    if not available:
        print(device + ": unavailable")
        continue
    try:
        result = engine.evaluate(batch, mode="float32", device=device)
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        print(device + ": unsupported: " + str(exc))
        continue
    actual = result.posterior_marginals.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    print(device + ": validated 256-vector float32 batch")
```

This verifies numerical agreement on this fixture. It measures no throughput;
benchmark with compilation, warmup, transfers, hardware, precision, and batch
size reported separately. See the [benchmark workflow](wmc-guardrail.md#reproduction-and-scope).
