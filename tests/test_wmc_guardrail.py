"""Independent weighted enumeration oracles for the guardrail inference API."""

from fractions import Fraction
from itertools import product
import math

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.stage5_wmc_guardrail import compile_guardrail


def as_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def formula(n, clauses):
    return CNF(
        n, [Clause(tuple(Literal(abs(x) - 1, x > 0) for x in c)) for c in clauses]
    )


def weighted_models(cnf, probs):
    rows = []
    for bits in product((False, True), repeat=cnf.num_vars):
        if all(
            any(bits[lit.var] == lit.pos for lit in c.literals) for c in cnf.clauses
        ):
            weight = math.prod((p if bit else 1 - p) for p, bit in zip(probs, bits))
            rows.append((bits, weight))
    return rows


def oracle(cnf, probs):
    rows = weighted_models(cnf, probs)
    z = sum((w for _, w in rows), Fraction(0))
    marginals = (
        [
            sum((w for bits, w in rows if bits[i]), Fraction(0)) / z
            for i in range(cnf.num_vars)
        ]
        if z
        else None
    )
    return z, marginals


@st.composite
def rational_cases(draw):
    n = draw(st.integers(0, 14))
    raw = (
        draw(
            st.lists(
                st.lists(
                    st.integers(1, max(1, n)).flatmap(
                        lambda v: st.sampled_from((v, -v))
                    ),
                    max_size=5,
                ),
                max_size=8,
            )
        )
        if n
        else []
    )
    probs = draw(st.lists(st.integers(0, 8), min_size=n, max_size=n))
    return (
        formula(n, raw),
        [Fraction(p, 8) for p in probs],
        draw(st.sampled_from((2, 4))),
    )


@st.composite
def mixed_inference_cases(draw):
    import random
    from alg_sharpsat.benchmarks import coupled_3sat, bipartite_3sat

    family = draw(st.sampled_from(("random", "coupled", "bipartite")))
    n = draw(st.integers(6, 14))
    seed = draw(st.integers(0, 2**16))
    if family == "random":
        rng = random.Random(seed)
        clauses = [
            [
                (v + 1) * (1 if rng.getrandbits(1) else -1)
                for v in rng.sample(range(n), rng.randint(1, 5))
            ]
            for _ in range(rng.randint(2, 6))
        ]
        cnf = formula(n, clauses)
    else:
        cnf = (coupled_3sat if family == "coupled" else bipartite_3sat)(n, seed=seed)
    denominators = draw(st.lists(st.integers(3, 11), min_size=n, max_size=n))
    probs = [Fraction(draw(st.integers(1, d - 1)), d) for d in denominators]
    options = dict(
        num_groups=draw(st.sampled_from((2, 4))),
        max_width=draw(st.sampled_from((None, 3))),
        max_degree=draw(st.sampled_from((None, 2)))
        if family == "random" and n <= 7
        else None,
        semi_join_pushdown=draw(st.booleans()),
    )
    return cnf, probs, options


@pytest.mark.g13
@settings(max_examples=150, deadline=None, derandomize=True)
@given(mixed_inference_cases())
def test_150_exact_rational_partition_and_all_marginals(case):
    cnf, probs, options = case
    compiled = compile_guardrail(cnf, **options)
    result = compiled.evaluate(probs, mode="exact_rational")
    z, marginals = oracle(cnf, probs)
    assert result.partition_function == z
    floating = compiled.evaluate(probs, mode="float64", device="cpu")
    assert float(floating.partition_function) == pytest.approx(
        float(z), rel=1e-11, abs=1e-14
    )
    if z:
        assert list(result.posterior_marginals) == marginals
        np.testing.assert_allclose(
            as_numpy(floating.posterior_marginals),
            np.array(marginals, dtype=float),
            atol=1e-11,
        )
        assert result.acceptance_prob == z
        assert result.log_partition == pytest.approx(math.log(z))
        assert result.kl_divergence_nats == pytest.approx(-math.log(z))
    count = brute_force_count(cnf)
    assert (
        compiled.evaluate(
            weights=[[1, 1]] * cnf.num_vars, mode="exact_rational"
        ).partition_function
        == count
    )
    assert compiled.evaluate(
        [Fraction(1, 2)] * cnf.num_vars, mode="exact_rational"
    ).partition_function == Fraction(count, 2**cnf.num_vars)


@pytest.mark.g13
@pytest.mark.parametrize(
    "cnf, probs",
    [
        (formula(0, []), []),
        (formula(0, [[]]), []),
        (formula(3, [[1, -1]]), [Fraction(0), Fraction(1), Fraction(1, 3)]),
        (formula(2, [[1], [-1]]), [Fraction(1, 2)] * 2),
        (formula(3, [[1], [2, 3]]), [Fraction(0), Fraction(1), Fraction(0)]),
        (formula(4, [[1, 2], [-1, 3], [-1, 4]]), [Fraction(1, 3)] * 4),
    ],
)
def test_exact_edge_cases_and_shared_source_correlations(cnf, probs):
    result = compile_guardrail(cnf, num_groups=4).evaluate(probs, mode="exact_rational")
    z, marginals = oracle(cnf, probs)
    assert result.partition_function == z
    if z:
        assert list(result.posterior_marginals) == marginals


@pytest.mark.g14
@pytest.mark.parametrize("groups", [2, 4])
def test_map_is_global_optimum_and_samples_obey_conditional_law(groups):
    cnf = formula(4, [[1, 2], [-1, 3], [-2, 4], [-3, -4]])
    probs = [Fraction(1, 5), Fraction(2, 3), Fraction(3, 7), Fraction(4, 5)]
    compiled = compile_guardrail(cnf, num_groups=groups)
    rows = weighted_models(cnf, probs)
    assignment, weight = compiled.map_inference(probs, mode="exact_rational")
    assert tuple(assignment) in dict(rows)
    assert weight == max(w for _, w in rows)
    assert dict(rows)[tuple(assignment)] == weight
    samples = np.asarray(
        compiled.sample_conditional(
            probs, num_samples=6000, seed=41, mode="exact_rational"
        )
    )
    assert samples.shape == (6000, 4)
    assert all(tuple(row) in dict(rows) for row in samples)
    assert np.array_equal(
        samples,
        compiled.sample_conditional(
            probs, num_samples=6000, seed=41, mode="exact_rational"
        ),
    )
    _, marginal = oracle(cnf, probs)
    np.testing.assert_allclose(
        samples.mean(axis=0), np.array(marginal, dtype=float), atol=0.035
    )


@pytest.mark.g15
@pytest.mark.parametrize("mode, atol", [("float64", 1e-12), ("float32", 2e-6)])
def test_batch32_matches_independent_exact_results(mode, atol):
    cnf = formula(4, [[1, 2], [-1, 3], [-2, 4]])
    batch = [[Fraction((i + j * 3) % 9, 8) for j in range(4)] for i in range(32)]
    compiled = compile_guardrail(cnf, num_groups=4)
    result = compiled.evaluate(np.array(batch, dtype=float), mode=mode, device="cpu")
    expected = [oracle(cnf, probs) for probs in batch]
    np.testing.assert_allclose(
        result.partition_function, [float(z) for z, _ in expected], atol=atol
    )
    for i, (z, marginals) in enumerate(expected):
        if z:
            np.testing.assert_allclose(
                result.posterior_marginals[i],
                np.array(marginals, dtype=float),
                atol=atol,
            )


@pytest.mark.g13
@pytest.mark.parametrize("degree,width", [(None, 3), (2, 3), (2, 4)])
def test_unique_extension_transforms_preserve_original_weights(degree, width):
    cnf = formula(6, [[1, 2, 3, 4, 5, 6], [1, -2], [1, -3], [1, -4], [-1, 5], [-1, 6]])
    probs = [Fraction(i, 7) for i in range(1, 7)]
    result = compile_guardrail(
        cnf, num_groups=4, max_degree=degree, max_width=width
    ).evaluate(probs, mode="exact_rational")
    z, marginals = oracle(cnf, probs)
    assert result.partition_function == z
    assert list(result.posterior_marginals) == marginals


@pytest.mark.g13
def test_general_weights_and_exact_integer_reconstruction_beyond_90_bits():
    compiled = compile_guardrail(formula(2, [[1, 2]]), num_groups=2)
    big = 2**50 + 27
    weights = [[big, big + 1], [big + 2, big + 3]]
    result = compiled.evaluate(weights=weights, mode="exact_rational")
    expected = big * (big + 3) + (big + 1) * (big + 2) + (big + 1) * (big + 3)
    assert expected > 2**90
    assert result.partition_function == expected
    assert list(result.posterior_marginals) == [
        Fraction((big + 1) * (2 * big + 5), expected),
        Fraction((big + 3) * (2 * big + 1), expected),
    ]


@pytest.mark.g13
def test_log_partition_survives_probability_underflow():
    n = 8
    cnf = formula(n, [[i + 1] for i in range(n)])
    result = compile_guardrail(cnf, num_groups=4).evaluate(
        [1e-50] * n, mode="float64", device="cpu"
    )
    assert math.isfinite(result.log_partition)
    assert result.log_partition == pytest.approx(n * math.log(1e-50))
    assert result.log_partition < -700
    np.testing.assert_allclose(result.posterior_marginals, np.ones(n))


@pytest.mark.g15
@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_available_accelerator_matches_cpu(device):
    available = (
        torch.cuda.is_available()
        if device == "cuda"
        else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} hardware is unavailable")
    compiled = compile_guardrail(formula(4, [[1, 2], [-1, 3], [-2, 4]]), num_groups=4)
    probs = [0.2, 0.4, 0.6, 0.8]
    cpu = compiled.evaluate(probs, mode="float64", device="cpu")
    gpu = compiled.evaluate(probs, mode="float32", device=device)
    np.testing.assert_allclose(
        as_numpy(gpu.partition_function), as_numpy(cpu.partition_function), rtol=2e-5
    )
    np.testing.assert_allclose(
        as_numpy(gpu.posterior_marginals), as_numpy(cpu.posterior_marginals), atol=2e-5
    )


@pytest.mark.g14
@pytest.mark.parametrize(
    "cnf,probs",
    [
        (formula(1, [[1], [-1]]), [Fraction(1, 2)]),
        (formula(1, [[1]]), [Fraction(0)]),
    ],
)
def test_zero_mass_map_and_sampling_report_explicit_failure(cnf, probs):
    compiled = compile_guardrail(cnf)
    for call in (
        lambda: compiled.map_inference(probs, mode="exact_rational"),
        lambda: compiled.sample_conditional(
            probs, num_samples=2, seed=1, mode="exact_rational"
        ),
    ):
        with pytest.raises(
            ValueError, match="Formula is unsatisfiable or has zero probability mass"
        ):
            call()


@pytest.mark.g13
def test_decimal_general_weights_use_neutral_auxiliary_variables():
    from decimal import Decimal

    cnf = formula(5, [[1, 2, 3, 4, 5], [1, -2], [1, -3], [-1, 4]])
    weights = [
        [Decimal(".25"), Decimal(".75")],
        [Decimal(".8"), Decimal(".3")],
        [Decimal(".1"), Decimal(".4")],
        [Decimal(".7"), Decimal(".2")],
        [Decimal(".5"), Decimal(".6")],
    ]
    compiled = compile_guardrail(cnf, num_groups=4, max_width=3, max_degree=1)
    result = compiled.evaluate(weights=weights, mode="exact_rational")
    rows = [
        (bits, math.prod(Fraction(weights[i][int(bit)]) for i, bit in enumerate(bits)))
        for bits, _ in weighted_models(cnf, [Fraction(1, 2)] * 5)
    ]
    z = sum((w for _, w in rows), Fraction(0))
    assert result.partition_function == z
    assert list(result.posterior_marginals) == [
        sum(w for bits, w in rows if bits[i]) / z for i in range(5)
    ]


@pytest.mark.g15
def test_benchmark_cli_smoke(tmp_path, monkeypatch):
    import json
    from alg_sharpsat.guardrail_benchmarks import main

    output = tmp_path / "guardrail-smoke.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "guardrail-benchmarks",
            "--sizes",
            "6",
            "--batch-size",
            "4",
            "--repeats",
            "1",
            "--skip-accelerators",
            "--json",
            str(output),
        ],
    )
    main() == 0
    payload = json.loads(output.read_text())
    assert payload["schema"] == "benchmark-v3.2-guardrail"
    assert payload["cases"]
    assert payload["rare_event"]["conditional_samples"]["count"] > 0
    assert payload["rare_event"]["conditional_samples"]["all_valid"]
    assert 0 < payload["rare_event"]["acceptance_probability"] < 1e-5


@pytest.mark.g13
@pytest.mark.parametrize("n", [10, 12, 14])
def test_nontrivial_coupled_formulas_through_fourteen_variables(n):
    cnf = formula(n, [[i + 1, (i + 1) % n + 1, -((i + 3) % n + 1)] for i in range(n)])
    probs = [Fraction(i % 7 + 1, 9) for i in range(n)]
    z, marginals = oracle(cnf, probs)
    result = compile_guardrail(cnf, num_groups=4).exact_marginals(probs)
    assert result.partition_function == z
    assert list(result.posterior_marginals) == marginals


@pytest.mark.g14
def test_deterministic_and_empty_formula_sampling():
    compiled = compile_guardrail(formula(3, []), num_groups=4)
    probs = [Fraction(0), Fraction(1), Fraction(0)]
    assignment, weight = compiled.map_inference(probs, mode="exact_rational")
    assert tuple(assignment) == (False, True, False)
    assert weight == 1
    np.testing.assert_array_equal(
        compiled.sample_conditional(
            probs, num_samples=20, seed=19, mode="exact_rational"
        ),
        [[0, 1, 0]] * 20,
    )
    empty = compile_guardrail(formula(0, []))
    assert empty.map_inference([], mode="exact_rational") == ((), 1)
    assert np.asarray(
        empty.sample_conditional([], num_samples=3, seed=1, mode="exact_rational")
    ).shape == (3, 0)


@pytest.mark.g14
@settings(max_examples=50, deadline=None, derandomize=True)
@given(rational_cases())
def test_50_map_global_optimality_cases(case):
    cnf, probs, groups = case
    compiled = compile_guardrail(cnf, num_groups=groups)
    rows = weighted_models(cnf, probs)
    best = max((w for _, w in rows), default=Fraction(0))
    if not best:
        with pytest.raises(ValueError, match="zero probability mass"):
            compiled.map_inference(probs, mode="exact_rational")
    else:
        assignment, weight = compiled.map_inference(probs, mode="exact_rational")
        assert dict(rows)[tuple(assignment)] == best == weight


@pytest.mark.g13
def test_rational_pair_inputs_and_huge_free_variable_weights():
    compiled = compile_guardrail(formula(2, [[1]]))
    result = compiled.evaluate([(1, 3), (2, 5)], mode="exact_rational")
    assert result.partition_function == Fraction(1, 3)
    assert list(result.posterior_marginals) == [1, Fraction(2, 5)]
    huge = 2**130 + 17
    result = compiled.evaluate(
        weights=[[(1, 3), (2, 3)], [huge, huge + 1]], mode="exact_rational"
    )
    assert result.partition_function == Fraction(2, 3) * (2 * huge + 1)
    assert list(result.posterior_marginals) == [1, Fraction(huge + 1, 2 * huge + 1)]


@pytest.mark.g13
@pytest.mark.parametrize(
    "probs",
    [
        [-0.1, 0.5],
        [1.1, 0.5],
        [float("nan"), 0.5],
        [float("inf"), 0.5],
        [0.5],
        [[[0.5, 0.5]]],
    ],
)
@pytest.mark.parametrize("mode", ["float64", "exact_rational"])
def test_invalid_probability_inputs_fail(probs, mode):
    with pytest.raises((ValueError, OverflowError)):
        compile_guardrail(formula(2, [[1, 2]])).evaluate(probs, mode=mode)


@pytest.mark.g13
@pytest.mark.parametrize(
    "weights", [[[-1, 1], [1, 1]], [[float("inf"), 1], [1, 1]], [[1, 1, 1]]]
)
def test_invalid_general_weights_fail(weights):
    with pytest.raises(ValueError):
        compile_guardrail(formula(2, [[1, 2]])).evaluate(weights=weights)


@pytest.mark.g15
def test_explicit_resource_limits_fail_instead_of_truncating():
    from alg_sharpsat.stage3_gpu_engine import ResourceLimitError

    cnf = formula(6, [[1, 2, 3], [-1, 4, 5], [-2, -4, 6]])
    with pytest.raises(ResourceLimitError):
        compile_guardrail(cnf, max_memory_bytes=1)
    compiled = compile_guardrail(cnf, max_work=1000)
    with pytest.raises(ResourceLimitError):
        compiled.evaluate(np.full((10000, 6), 0.5))


@pytest.mark.g14
def test_exact_sampling_uses_integer_draws_for_sub_float_precision_mass(monkeypatch):
    import alg_sharpsat._guardrail_inference as inference

    compiled = compile_guardrail(formula(1, []))
    totals = []

    class LastInteger:
        def __init__(self, seed):
            pass

        def randrange(self, total):
            assert isinstance(total, int)
            totals.append(total)
            return total - 1

    monkeypatch.setattr(inference.random, "Random", LastInteger)
    huge = 2**130
    samples = compiled.sample_conditional(
        weights=[[huge, 1]], num_samples=1, seed=0, mode="exact_rational"
    )
    assert samples.tolist() == [[1]]
    assert huge + 1 in totals


@pytest.mark.g14
def test_map_ties_are_stable_and_optimal():
    compiled = compile_guardrail(formula(3, [[1, 2, 3]]), num_groups=4)
    probs = [Fraction(1, 2)] * 3
    first = compiled.map_inference(probs, mode="exact_rational")
    assert first == compiled.map_inference(probs, mode="exact_rational")
    assert any(first[0])
    assert first[1] == Fraction(1, 8)


@pytest.mark.g13
def test_expandable_crt_has_proven_bound_and_more_than_three_primes():
    compiled = compile_guardrail(formula(2, [[1, 2]]), num_groups=4)
    huge = 2**100 + 3
    result = compiled.evaluate(
        weights=[[huge, huge + 1], [huge + 2, huge + 3]], mode="exact_rational"
    )
    expected = huge * (huge + 3) + (huge + 1) * (huge + 2) + (huge + 1) * (huge + 3)
    assert result.partition_function == expected
    info = compiled.last_evaluation
    assert info["arithmetic"] == "expandable_crt"
    assert len(info["primes"]) > 3
    assert math.prod(info["primes"]) > info["integer_bound"] >= expected


@pytest.mark.g13
def test_g4_shared_source_hyperedge_retains_joint_support():
    # x0 couples to each of the other three groups through distinct implications.
    cnf = formula(8, [[-1, 3], [-1, 5], [-1, 7], [1, 2], [-3, 4], [-5, 6], [-7, 8]])
    probs = [Fraction(i + 1, 10) for i in range(8)]
    compiled = compile_guardrail(cnf, num_groups=4, semi_join_pushdown=False)
    result = compiled.evaluate(probs, mode="exact_rational")
    z, marginals = oracle(cnf, probs)
    assert result.partition_function == z
    assert list(result.posterior_marginals) == marginals


@pytest.mark.g13
def test_public_wmc_projection_and_marginal_exports():
    from alg_sharpsat import guardrail_project, exact_marginals, wmc_sharp_sat

    cnf = formula(2, [[1, 2]])
    weights = [[Fraction(2, 3), Fraction(1, 3)], [Fraction(3, 5), Fraction(2, 5)]]
    assert wmc_sharp_sat(cnf, weights) == Fraction(3, 5)
    result = exact_marginals(
        cnf, [Fraction(1, 3), Fraction(2, 5)], mode="exact_rational"
    )
    projected = guardrail_project(
        cnf, [Fraction(1, 3), Fraction(2, 5)], mode="exact_rational"
    )
    assert projected.partition_function == result.partition_function
    assert projected.posterior_marginals == result.posterior_marginals
    rows = dict(weighted_models(cnf, [Fraction(1, 3), Fraction(2, 5)]))
    assert rows[tuple(projected.map_assignment)] == max(rows.values())
    assert projected.map_prob == max(rows.values())


@pytest.mark.g13
@pytest.mark.parametrize("mode,tiny", [("float64", 1e-323), ("float32", 1e-40)])
def test_tiny_rows_accumulate_to_representable_marginal(mode, tiny):
    # x0 can be true with many equally weighted choices of x2..x11.
    # Their separate row masses may underflow even though their sum does not.
    cnf = formula(12, [[1, 2]] + [[2, var + 1] for var in range(2, 12)])
    compiled = compile_guardrail(cnf, num_groups=2)
    probabilities = [tiny] + [0.5] * 11
    exact = compiled.evaluate(probabilities, mode="exact_rational")
    expected = float(exact.posterior_marginals[0])
    batch = np.array([probabilities, [0.25] + [0.5] * 11])
    result = compiled.evaluate(batch, mode=mode, device="cpu")
    actual = result.posterior_marginals[0, 0].item()
    assert actual > 0
    assert actual == pytest.approx(expected, rel=2e-4, abs=0)
    assert torch.isfinite(result.posterior_marginals[1]).all()


@pytest.mark.g14
def test_sampling_work_guard_bounds_large_traversals():
    from alg_sharpsat.stage3_gpu_engine import ResourceLimitError

    compiled = compile_guardrail(formula(3, [[1, 2, 3]]), max_work=1000)
    with pytest.raises(ResourceLimitError, match="max_work"):
        compiled.sample_conditional([0.5] * 3, num_samples=10000)
