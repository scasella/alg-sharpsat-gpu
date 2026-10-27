"""Observed facts remain fixed while proposed decisions are conditioned."""

from fractions import Fraction
from itertools import product
from math import prod

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from alg_sharpsat.domains import get_domains
from alg_sharpsat.policy_dsl import PolicySpec, RuleSpec, VariableSpec, apply_clamps

pytestmark = pytest.mark.g18


@given(
    witness=st.lists(st.integers(0, 1), min_size=4, max_size=4),
    clauses=st.lists(
        st.tuples(
            st.sampled_from(["implies", "xor"]), st.integers(0, 3), st.integers(0, 3)
        ),
        min_size=1,
        max_size=5,
    ),
    selected=st.sets(st.integers(0, 1)),
    numerators=st.lists(st.integers(1, 4), min_size=4, max_size=4),
)
@settings(max_examples=35, deadline=None)
def test_satisfiable_subset_is_fixed_everywhere(witness, clauses, selected, numerators):
    names = [f"v{i}" for i in range(4)]
    rules = []
    for index, (operator, left, right) in enumerate(clauses):
        # Each literal is chosen against a known satisfying witness. Implications
        # have true consequents; XOR has opposite-valued witness literals.
        lhs = names[left] if witness[left] else f"not {names[left]}"
        rhs_true = operator == "implies"
        rhs_positive = bool(witness[right]) == rhs_true
        rhs = names[right] if rhs_positive else f"not {names[right]}"
        rules.append(
            RuleSpec(str(index), "Generated constraint", f"({lhs}) {operator} ({rhs})")
        )
    policy = PolicySpec(
        [
            VariableSpec(name, name, role="evidence" if i < 2 else "action")
            for i, name in enumerate(names)
        ],
        rules,
    )
    engine = policy.compile_guardrail()
    priors = [Fraction(value, 5) for value in numerators]
    clamps = {names[i]: witness[i] for i in selected}
    weights = [
        Fraction(clamps[name]) if name in clamps else prior
        for name, prior in zip(names, priors)
    ]
    allowed = []
    for bits in product((0, 1), repeat=4):
        if not policy.check_violations(dict(zip(names, bits))):
            mass = prod(p if bit else 1 - p for p, bit in zip(weights, bits))
            if mass:
                allowed.append((bits, mass))
    expected_z = sum(mass for _, mass in allowed)
    result = engine.evaluate(priors, mode="exact_rational", clamped_vars=clamps)
    assert result.partition_function == expected_z > 0
    expected_marginals = [
        sum(mass for bits, mass in allowed if bits[i]) / expected_z for i in range(4)
    ]
    assert list(result.posterior_marginals) == expected_marginals
    assignment, mass = engine.map_inference(priors, clamped_vars=clamps)
    assert mass == max(weight for _, weight in allowed)
    assert (tuple(assignment), mass) in allowed
    samples = engine.sample_conditional(priors, num_samples=8, clamped_vars=clamps)
    for sample in samples:
        assert not policy.check_violations(dict(zip(names, sample)))
    for name, value in clamps.items():
        index = names.index(name)
        assert result.posterior_marginals[index] == value
        assert assignment[index] == value
        assert (samples[:, index] == value).all()


def test_contradictory_evidence_has_zero_mass():
    engine = PolicySpec(
        [VariableSpec("fact", "Fact")], [RuleSpec("required", "Required", "fact")]
    ).compile_guardrail()
    assert (
        engine.evaluate(
            clamped_vars={"fact": 0}, mode="exact_rational"
        ).partition_function
        == 0
    )
    with pytest.raises(ValueError):
        engine.map_inference(clamped_vars={"fact": 0})
    with pytest.raises(ValueError):
        engine.sample_conditional(clamped_vars={"fact": 0})


@pytest.mark.parametrize(
    "clamps", [{"missing": 1}, {"fact": 0.5}, {"fact": "1"}, {"fact": 2}, [("fact", 1)]]
)
def test_invalid_clamps_rejected(clamps):
    engine = PolicySpec([VariableSpec("fact", "Fact")], []).compile_guardrail()
    for method in (engine.evaluate, engine.map_inference, engine.sample_conditional):
        with pytest.raises(ValueError):
            method(clamped_vars=clamps)


def test_batch_and_mapping_inputs_unchanged():
    engine = PolicySpec(
        [VariableSpec("fact", "Fact"), VariableSpec("action", "Action")], []
    ).compile_guardrail()
    priors = np.array([[0.2, 0.4], [0.8, 0.6]])
    before = priors.copy()
    result = engine.evaluate(priors, clamped_vars={"fact": 1})
    assert np.array_equal(priors, before)
    assert np.array_equal(result.posterior_marginals[:, 0], [1, 1])
    mapping = {"fact": Fraction(1, 3), "action": Fraction(2, 3)}
    assert apply_clamps(mapping, engine.variable_names, {"fact": 0})["fact"] == 0
    assert mapping["fact"] == Fraction(1, 3)


@pytest.mark.parametrize(
    "pack", list(get_domains().values()), ids=lambda p: p.domain_id
)
def test_domain_evidence_and_metadata(pack):
    assert len(pack.sample_prompts) == 3
    assert all(
        v.role in {"evidence", "action", "safeguard"} for v in pack.policy.variables
    )
    engine = pack.policy.compile_guardrail()
    # Every conflict preset contributes a satisfiable witness. Subsets include
    # verified safeguards, true facts and false facts without inventing evidence.
    for preset in pack.presets.values():
        witness, _ = engine.map_inference(preset)
        eligible = [(i, v) for i, v in enumerate(pack.policy.variables) if v.can_clamp]
        for offset in (0, 1):
            clamps = {v.name: int(witness[i]) for i, v in eligible[offset::2]}
            result = engine.evaluate(preset, mode="exact_rational", clamped_vars=clamps)
            assignment, _ = engine.map_inference(preset, clamped_vars=clamps)
            samples = engine.sample_conditional(
                preset, num_samples=3, clamped_vars=clamps
            )
            assert result.partition_function > 0
            for sample in samples:
                assert not pack.policy.check_violations(
                    dict(zip(engine.variable_names, sample))
                )
            for name, value in clamps.items():
                index = engine.variable_names.index(name)
                assert result.posterior_marginals[index] == value
                assert assignment[index] == value
                assert (samples[:, index] == value).all()


def test_clamping_does_not_hide_invalid_prior():
    engine = PolicySpec([VariableSpec("fact", "Fact")], []).compile_guardrail()
    with pytest.raises(ValueError):
        engine.evaluate([float("nan")], clamped_vars={"fact": 1})
    with pytest.raises(ValueError):
        engine.evaluate([2], mode="exact_rational", clamped_vars={"fact": 0})


def test_explicit_verified_safeguards_are_eligible():
    cloud = get_domains()["cloud_ops_agent"]
    variables = {v.name: v for v in cloud.policy.variables}
    assert variables["prod_environment"].can_clamp
    assert variables["peer_review_approved"].can_clamp
    assert variables["mfa_verified"].can_clamp
    assert variables["fresh_snapshot_verified"].can_clamp
    assert not variables["run_schema_migration"].can_clamp
    assert VariableSpec("legacy", "Legacy", "context").effective_role == "evidence"


@pytest.mark.parametrize("mode", ["exact_rational", "float64"])
def test_default_priors_and_verified_observation_override(mode):
    engine = PolicySpec(
        [
            VariableSpec("fact", "Fact", default_prob=0),
            VariableSpec("action", "Action", default_prob=Fraction(1, 3)),
        ],
        [RuleSpec("dependency", "Dependency", "action -> fact")],
    ).compile_guardrail()
    result = engine.evaluate(mode=mode, clamped_vars={"fact": True})
    assert float(result.partition_function) == pytest.approx(1)
    assert float(result.posterior_marginals[0]) == 1
    assignment, _ = engine.map_inference(mode=mode, clamped_vars={"fact": True})
    assert assignment[0] == 1
    samples = engine.sample_conditional(
        mode=mode, num_samples=4, clamped_vars={"fact": True}
    )
    assert (samples[:, 0] == 1).all()
    assert engine.policy.variables[0].default_prob == 0


@pytest.mark.parametrize("category", ["control", "controls", "safeguard", "safeguards"])
def test_legacy_safeguard_categories_and_explicit_override(category):
    legacy = VariableSpec("verified", "Verified control", category)
    assert legacy.effective_role == "safeguard"
    assert legacy.can_clamp
    explicit = VariableSpec("proposed", "Proposed action", category, role="action")
    assert explicit.effective_role == "action"
    assert not explicit.can_clamp
