from fractions import Fraction
from itertools import product
import pytest
from alg_sharpsat.cnf import dpll_count, CNF, Clause, Literal
from alg_sharpsat.policy_dsl import VariableSpec, RuleSpec, PolicySpec, diff_policies

pytestmark = pytest.mark.g16


def policy(expr, names="abc"):
    return PolicySpec(
        [VariableSpec(n, n) for n in names], [RuleSpec("r", "test", expr)]
    )


@pytest.mark.parametrize(
    "expr,predicate",
    [
        ("a and b", lambda a, b, c: a and b),
        ("a & b", lambda a, b, c: a and b),
        ("a or b", lambda a, b, c: a or b),
        ("a | b", lambda a, b, c: a or b),
        ("not a", lambda a, b, c: not a),
        ("~a", lambda a, b, c: not a),
        ("!a", lambda a, b, c: not a),
        ("a -> b", lambda a, b, c: not a or b),
        ("implies(a,b)", lambda a, b, c: not a or b),
        ("a <-> b", lambda a, b, c: a == b),
        ("iff(a,b)", lambda a, b, c: a == b),
        ("xor(a,b)", lambda a, b, c: a != b),
        ("a xor b", lambda a, b, c: a != b),
        ("at_most_one(a,b,c)", lambda *v: sum(v) <= 1),
        ("at_least_one(a,b,c)", lambda *v: sum(v) >= 1),
        ("exactly_one(a,b,c)", lambda *v: sum(v) == 1),
        ("forbid(a)", lambda a, b, c: not a),
        ("require(a)", lambda a, b, c: a),
        ("a or (b and c)", lambda a, b, c: a or (b and c)),
        ("a -> b -> c", lambda a, b, c: not a or not b or c),
    ],
)
def test_exhaustive(expr, predicate):
    p = policy(expr)
    cnf = p.to_cnf()
    assert cnf.to_dimacs() == p.to_cnf().to_dimacs()
    for bits in product((False, True), repeat=3):
        expected = predicate(*bits)
        assert (not p.check_violations(dict(zip("abc", bits)))) == expected
        fixed = CNF(
            cnf.num_vars,
            cnf.clauses + [Clause((Literal(i, v),)) for i, v in enumerate(bits)],
        )
        assert dpll_count(fixed) == int(expected)  # exactly one auxiliary extension


def test_neutral_weights_and_inference():
    p = policy("a or xor(b,c)")
    g = p.compile_guardrail()
    assert g.cnf.num_vars > 3
    probs = ["1/3", "2/5", "3/7"]
    expected = Fraction(1, 3) + Fraction(2, 3) * (
        Fraction(2, 5) * Fraction(4, 7) + Fraction(3, 5) * Fraction(3, 7)
    )
    exact = g.evaluate(probs, mode="exact_rational")
    assert exact.partition_function == exact.acceptance_prob == expected
    assert len(exact.posterior_marginals) == 3
    assert g.evaluate(probs).acceptance_prob.item() == pytest.approx(float(expected))
    assignment, weight = g.map_inference(probs)
    assert not p.check_violations(dict(zip("abc", assignment)))
    assert isinstance(weight, Fraction)
    samples = g.sample_conditional(probs, 20, seed=7)
    assert samples.shape == (20, 3)
    assert all(not p.check_violations(dict(zip("abc", row))) for row in samples)
    assert g.model_count == 6


def test_diff_reordered_variables():
    old = policy("a", "ab")
    new = policy("b", "ba")
    d = diff_policies(old, new, {"a": "1/3", "b": "1/4"})
    assert (
        d.old_allowed_count,
        d.new_allowed_count,
        d.intersection_count,
        d.newly_exposed_count,
        d.revoked_count,
    ) == (2, 2, 1, 1, 1)
    assert d.old_allowed_prob == Fraction(1, 3)
    assert d.new_allowed_prob == Fraction(1, 4)
    assert d.intersection_prob == Fraction(1, 12)
    assert d.newly_exposed_prob == Fraction(1, 6)
    assert d.revoked_prob == Fraction(1, 4)


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "a.b",
        "a[0]",
        "a + b",
        "unknown",
        "a b",
        "a and",
        "(a",
        "implies(a)",
        "lambda a: a",
    ],
)
def test_restricted_parser(expr):
    with pytest.raises(ValueError):
        policy(expr)


def test_limits_and_empty_cardinality():
    with pytest.raises(ValueError):
        PolicySpec(
            [VariableSpec("a", "a")], [RuleSpec("r", "", "a and a")], max_tokens=1
        )
    with pytest.raises(ValueError):
        PolicySpec(
            [VariableSpec(n, n) for n in "abc"],
            [RuleSpec("r", "", "a or xor(b,c)")],
            max_aux_variables=0,
        ).to_cnf()
    assert dpll_count(policy("at_least_one()").to_cnf()) == 0
    assert dpll_count(policy("at_most_one()").to_cnf()) == 8


@pytest.mark.parametrize(
    "expression", ["a -> (b and c)", "forbid(a,b,c)", "(a or b) -> c", "(a and b) -> c"]
)
def test_common_patterns_have_no_auxiliaries(expression):
    assert policy(expression).to_cnf().num_vars == 3


def test_variadic_helpers_and_rule_objects():
    p = policy("require(a,b,c)")
    assert p.check_violations(dict(a=1, b=0, c=1)) == list(p.rules)
    for bad in ("false", "1", 2, 0.5, None):
        with pytest.raises(ValueError):
            p.check_violations(dict(a=bad, b=1, c=1))
    for bits in product((False, True), repeat=3):
        assert bool(
            policy("forbid(a,b,c)").check_violations(dict(zip("abc", bits)))
        ) == all(bits)


def test_numpy_and_torch_batches():
    import numpy as np
    import torch

    g = policy("a", "ab").compile_guardrail()
    expected = [0.2, 0.8]
    for batch in (
        np.array([[0.2, 0.3], [0.8, 0.7]]),
        torch.tensor([[0.2, 0.3], [0.8, 0.7]], dtype=torch.float64),
    ):
        result = g.evaluate(batch)
        assert result.posterior_marginals.shape == (2, 2)
        assert result.acceptance_prob.tolist() == pytest.approx(expected)


def test_empty_policy_and_zero_mass():
    import math

    g = PolicySpec([], []).compile_guardrail()
    assert g.model_count == 1
    result = g.evaluate([], mode="exact_rational")
    assert result.partition_function == 1
    assert result.posterior_marginals == []
    zero = policy("a and not a", "a").compile_guardrail()
    result = zero.evaluate(["1/2"], mode="exact_rational")
    assert result.partition_function == 0
    assert result.posterior_marginals == [None]
    assert math.isnan(result.kl_divergence_nats)
    assert math.isnan(zero.evaluate([0.5]).kl_divergence_nats.item())


def test_bounded_probability_parsing():
    for value in ("1e-1000000000", "1e1000000000", "1/" + "9" * 5000):
        with pytest.raises(ValueError):
            policy("a", "a").compile_guardrail().evaluate([value])
        with pytest.raises(ValueError):
            PolicySpec([VariableSpec("a", "a", default_prob=value)], [])


def test_numeric_batch_matches_exact_without_rational_conversion(monkeypatch):
    import alg_sharpsat.policy_dsl as dsl
    import numpy as np
    import torch

    g = policy("a or xor(b,c)").compile_guardrail()
    rows = [[0.2, 0.3, 0.4], [0.8, 0.5, 0.1]]
    expected = [g.evaluate(row, mode="exact_rational") for row in rows]

    def reject_fraction_conversion(value):
        raise AssertionError("numeric floating inputs must avoid Fraction parsing")

    monkeypatch.setattr(dsl, "parse_probability", reject_fraction_conversion)
    for data in (rows, np.asarray(rows), torch.tensor(rows, dtype=torch.float64)):
        result = g.evaluate(data)
        assert result.acceptance_prob.tolist() == pytest.approx(
            [float(r.partition_function) for r in expected]
        )
        np.testing.assert_allclose(
            result.posterior_marginals.numpy(),
            [[float(v) for v in r.posterior_marginals] for r in expected],
        )
        assert g.evaluate(
            data, mode="float32"
        ).acceptance_prob.tolist() == pytest.approx(
            [float(r.partition_function) for r in expected], rel=1e-6
        )


@pytest.mark.parametrize(
    "probabilities", [{"a": 0.5}, {"a": 0.5, "b": 0.5, "extra": 0.5}]
)
def test_probability_mapping_requires_exact_keys(probabilities):
    p = policy("a", "ab")
    g = p.compile_guardrail()
    for operation in (g.evaluate, g.map_inference, g.sample_conditional):
        with pytest.raises(ValueError, match="mapping"):
            operation(probabilities)
    with pytest.raises(ValueError, match="mapping"):
        diff_policies(p, p, probabilities)


@pytest.mark.parametrize("values", [[0.5], [0.5, 0.5, 0.5]])
def test_diff_rejects_wrong_vector_length(values):
    p = policy("a", "ab")
    with pytest.raises(ValueError, match="length"):
        diff_policies(p, p, values)


def test_numpy_boolean_diagnostics_and_invalid_floating_values():
    import numpy as np

    p = policy("a", "ab")
    assert p.check_violations({"a": np.bool_(False), "b": np.bool_(True)}) == list(
        p.rules
    )
    g = p.compile_guardrail()
    for invalid in (
        [float("nan"), 0.5],
        [float("inf"), 0.5],
        [-0.1, 0.5],
        [1.1, 0.5],
        [[0.5]],
        np.zeros((1, 1, 2)),
    ):
        with pytest.raises(ValueError):
            g.evaluate(invalid)
    exact = g.evaluate({"a": Fraction(1, 3), "b": "2/7"}, mode="exact_rational")
    assert g.evaluate(
        {"a": Fraction(1, 3), "b": "2/7"}
    ).acceptance_prob.item() == pytest.approx(float(exact.partition_function))
