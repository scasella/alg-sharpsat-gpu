import csv
import json

import pytest

from alg_sharpsat.benchmarks import (
    bipartite_3sat, coupled_3sat, main, random_3sat, run_benchmark,
)
from alg_sharpsat.cnf import brute_force_count


@pytest.mark.parametrize("generator", [random_3sat, coupled_3sat, bipartite_3sat])
def test_generators_are_reproducible_and_planted(generator):
    cnf = generator(8, seed=729)
    assert cnf == generator(8, seed=729)
    assert cnf != generator(8, seed=730)
    assert all(len(clause.literals) == 3 for clause in cnf.clauses)
    assert all(len({literal.var for literal in clause.literals}) == 3
               for clause in cnf.clauses)
    assert all(any(literal.pos for literal in clause.literals) for clause in cnf.clauses)
    assert brute_force_count(cnf) > 0


def test_coupled_bounds_bridges_and_bipartite_is_cross_clause_stress():
    n, bridges = 10, 4
    cnf = coupled_3sat(n, bridges=bridges)
    assert len(cnf.clauses) == 4 * n + bridges
    crosses = lambda clause: len({literal.var >= n // 2 for literal in clause.literals}) == 2
    assert sum(crosses(clause) for clause in cnf.clauses) == bridges
    assert all(crosses(clause) for clause in bipartite_3sat(n).clauses)
    assert random_3sat(6, num_clauses=0).clauses == []


@pytest.mark.g6
def test_benchmark_smoke_cli_counts_metrics_and_exports(tmp_path, capsys):
    json_path, csv_path = tmp_path / "rows.json", tmp_path / "rows.csv"
    assert main(["--sizes", "6", "8", "--seed", "17", "--device", "cpu",
                 "--json", str(json_path), "--csv", str(csv_path)]) == 0
    output = capsys.readouterr().out
    assert "bounded interfaces" in output
    assert "no general speedup" in output
    assert "product(k+1)" in output
    assert "DPLL_ms" in output and "tensor_ms" in output and "eval/s" in output
    rows = json.loads(json_path.read_text())
    assert [row["n"] for row in rows] == [6, 8]
    for row in rows:
        cnf = coupled_3sat(row["n"], seed=17)
        assert row["count"] == row["reference_count"] == brute_force_count(cnf)
        assert row["tensor_seconds"] > 0 and row["dpll_seconds"] > 0
        assert row["device"] == "cpu"
        assert row["conditioning_states_sum"] >= row["conditioning_states_max"] > 0
        assert row["residual_bits_sum"] >= row["residual_bits_max"] >= 0
        assert row["binary_link_states"] == 2 ** row["boundary_links"]
        assert row["stats"]["evaluations"] > 0
        assert row["evaluations_per_second"] > 0
        assert row["versions"]["python"] and row["versions"]["torch"]
    with csv_path.open(newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2
    assert int(exported[1]["count"]) == rows[1]["count"]


@pytest.mark.g6
def test_random_and_bipartite_smoke():
    for family in ("random", "bipartite"):
        row = run_benchmark(6, family=family, device="cpu", seed=3)
        assert row["count"] == row["reference_count"]


def test_cli_budget_exhaustion_is_explicit(capsys):
    assert main(["--sizes", "6", "--max-work", "1", "--device", "cpu"]) == 2
    assert "failed explicitly" in capsys.readouterr().err


@pytest.mark.parametrize("generator,args", [
    (random_3sat, {"num_vars": 2}),
    (coupled_3sat, {"num_vars": 5}),
    (coupled_3sat, {"num_vars": 6, "bridges": -1}),
    (bipartite_3sat, {"num_vars": 3}),
    (random_3sat, {"num_vars": 6, "num_clauses": -1}),
])
def test_invalid_generator_inputs(generator, args):
    with pytest.raises(ValueError):
        generator(**args)


@pytest.mark.g9
def test_both_engines_report_actual_subset_and_signature_work():
    from math import comb, prod

    row = run_benchmark(8, family="bipartite", device="cpu", seed=7, engine="both")
    assert row["count"] == row["engine_counts"]["lagrange"] == row["engine_counts"]["hermite"]
    assert row["count"] == row["reference_count"]
    baseline = row["engine_stats"]["lagrange"]
    assert row["lagrange_conditioning_states_sum"] == sum(
        prod(states) for states in baseline["conditioning_states"]
    )
    nodes = row["hermite_node_evaluations"]
    assert nodes
    for node in nodes:
        k, q = node["k"], node["q"]
        assert node["processed_subsets"] == sum(comb(k, j) for j in range(min(k, q) + 1))
        assert node["full_set_processed"] == (q >= k)
        assert node["conditioning_states_sum"] == node["processed_subsets"] * prod(node["conditioning_states"])
    assert any(node["q"] >= node["k"] for node in nodes)
    assert row["hermite_subset_conditioning_states_sum"] == sum(node["conditioning_states_sum"] for node in nodes)
    assert row["hermite_processed_subsets"] == sum(node["processed_subsets"] for node in nodes)
    assert row["hermite_over_lagrange_signature_work"] == (
        row["hermite_subset_conditioning_states_sum"] / row["lagrange_conditioning_states_sum"]
    )
    for name in ("lagrange", "hermite"):
        assert row["engine_seconds"][name] > 0
        assert row["preprocessing_source_signature_counts"][name] == row["engine_stats"][name]["source_signature_counts"]
        assert row["preprocessing_incoming_signature_counts"][name] == row["engine_stats"][name]["incoming_signature_counts"]


@pytest.mark.g9
def test_single_node_taylor_reports_full_derivative_subset_enumeration():
    row = run_benchmark(6, device="cpu", seed=17, engine="both",
                        jet_nodes=1, jet_mode="single_node_taylor")
    assert row["boundary_links"] > 0
    assert row["engine_counts"]["lagrange"] == row["engine_counts"]["hermite"] == row["reference_count"]
    for node in row["hermite_node_evaluations"]:
        assert node["k"] == node["q"] == row["boundary_links"]
        assert node["processed_subsets"] == 1 << row["boundary_links"]
        assert node["full_set_processed"]
    # Keep unfavorable work ratios: this is a measurement, not an asserted gain.
    assert row["hermite_over_lagrange_signature_work"] > 0


@pytest.mark.g9
def test_hermite_cli_exports_honest_node_details(tmp_path, capsys):
    json_path, csv_path = tmp_path / "jet.json", tmp_path / "jet.csv"
    assert main(["--sizes", "6", "--device", "cpu", "--engine", "both",
                 "--jet-nodes", "2", "--json", str(json_path), "--csv", str(csv_path)]) == 0
    output = capsys.readouterr().out
    assert "some q>=k" in output
    assert "ratios may exceed one" in output
    assert "exact signature-work" in output and "full-set=" in output
    assert "v1_ms" in output and "v2_ms" in output
    assert "v1_visits" in output and "v2_visits" in output
    assert "prime=" not in output  # Full per-prime traces remain in exports.
    row = json.loads(json_path.read_text())[0]
    assert row["engine_counts"]["lagrange"] == row["engine_counts"]["hermite"]
    assert row["hermite_node_evaluations"]
    with csv_path.open(newline="") as stream:
        exported = next(csv.DictReader(stream))
    assert json.loads(exported["hermite_node_evaluations"]) == row["hermite_node_evaluations"]


@pytest.mark.g9
def test_jet_subset_budget_fails_explicitly(capsys):
    assert main(["--sizes", "6", "--device", "cpu", "--engine", "both",
                 "--max-subset-terms", "1"]) == 2
    assert "failed explicitly" in capsys.readouterr().err


def test_invalid_benchmark_engine():
    with pytest.raises(ValueError, match="engine"):
        run_benchmark(6, engine="unknown")


@pytest.mark.g9
def test_hermite_only_accepts_explicit_orders():
    from alg_sharpsat.stage1_sparsify import partition_and_decouple_boundary

    boundary = len(partition_and_decouple_boundary(bipartite_3sat(8, seed=7)).boundary_links)
    orders = (boundary - 1, 0)
    row = run_benchmark(8, family="bipartite", seed=7, device="cpu",
                        engine="hermite", jet_orders=orders)
    assert row["engine_counts"] == {"hermite": row["reference_count"]}
    assert row["lagrange_conditioning_states_sum"] is None
    assert row["hermite_over_lagrange_signature_work"] is None
    assert {node["node_index"]: node["q"] for node in row["hermite_node_evaluations"]} == dict(enumerate(orders))
