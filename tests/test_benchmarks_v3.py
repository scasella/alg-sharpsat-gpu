import pytest

from alg_sharpsat import benchmarks
from alg_sharpsat.cnf import brute_force_count
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError, UnsupportedDeviceError

pytestmark = pytest.mark.g12


@pytest.mark.parametrize("family", ["coupled", "bipartite"])
def test_v3_counts_and_joint_signature_telemetry(family):
    row = benchmarks.run_v3_benchmark(6, family=family, seed=7, device="cpu", repeats=2)
    generator = benchmarks.coupled_3sat if family == "coupled" else benchmarks.bipartite_3sat
    expected = brute_force_count(generator(6, seed=7))
    assert set(row["engine_runs"]) == {"dpll", "lagrange", "hermite", "sparse_contract"}
    assert row["reference_count"] == row["count"] == expected
    for run in row["engine_runs"].values():
        assert run["status"] == "ok" and run["verified"]
        assert run["count"] == expected
        assert len(run["samples_seconds"]) == run["completed_repeats"] == 2
        assert run["seconds"] > 0
        assert run["failed_attempt_seconds"] is None
    stats = row["engine_stats"]["sparse_contract"]
    assert row["joint_signature_counts"] == stats["joint_signature_counts"]
    assert row["local_assignment_counts"] == stats["local_assignment_counts"]
    assert all(joint <= assignments for joint, assignments in zip(
        row["joint_signature_counts"], row["local_assignment_counts"]))
    assert row["join_pair_counts"] == stats["join_pair_counts"]
    assert row["selected_engine"] == "sparse_contract"
    assert all(value >= 0 for value in row["sparse_phase_seconds"].values())
    swap = row["swap_mobius"]
    assert swap["cost_estimate"]["crossover_established"] is False
    assert "miniature" in swap["runtime_scope"]
    if swap["runtime"]["status"] == "ok":
        assert swap["runtime"]["count"] == expected and swap["runtime"]["verified"]
    else:
        assert swap["runtime"]["status"] == "resource_limit"


def test_v3_resource_failure_does_not_hide_other_engines(monkeypatch):
    def exhausted(*args, **kwargs):
        raise ResourceLimitError("deliberate legacy budget")

    monkeypatch.setattr(benchmarks, "count_with_stats", exhausted)
    row = benchmarks.run_v3_benchmark(8, family="bipartite", device="cpu", seed=7,
                                       max_subset_terms=1)
    assert row["engine_status"]["lagrange"] == "resource_limit"
    assert row["engine_status"]["hermite"] == "resource_limit"
    assert row["engine_status"]["sparse_contract"] == "ok"
    assert row["engine_counts"]["sparse_contract"] == row["reference_count"]
    for engine in ("lagrange", "hermite"):
        run = row["engine_runs"][engine]
        assert run["seconds"] is None and run["failed_attempt_seconds"] > 0
        assert not run["verified"] and run["count"] is None
    assert row["swap_mobius"]["runtime"]["status"] == "not_run"
    assert row["swap_mobius"]["runtime"]["seconds"] is None


def test_measurement_retains_samples_and_uses_median(monkeypatch):
    clock = iter([0.0, 4.0, 10.0, 12.0, 20.0, 23.0])
    monkeypatch.setattr(benchmarks, "perf_counter", lambda: next(clock))
    run = benchmarks._measure_engine(lambda: (9, {"device": "cpu"}), repeats=3,
                                     device="cpu", reference=9)
    assert run["samples_seconds"] == [4.0, 2.0, 3.0]
    assert run["seconds"] == 3.0


@pytest.mark.parametrize("error,status", [
    (ResourceLimitError("budget"), "resource_limit"),
    (UnsupportedDeviceError("device"), "unsupported_device"),
])
def test_failed_measurement_is_not_a_completed_timing(error, status):
    def fail():
        raise error

    run = benchmarks._measure_engine(fail, repeats=2, device="cpu", reference=1)
    assert run["status"] == status
    assert run["seconds"] is None and run["failed_attempt_seconds"] >= 0
    assert run["samples_seconds"] == [] and run["completed_repeats"] == 0


def test_mismatched_count_is_an_error():
    with pytest.raises(AssertionError, match="differs from DPLL"):
        benchmarks._measure_engine(lambda: (2, {}), repeats=1, device="cpu", reference=3)


def test_v3_cli_default_matrix_and_statuses(monkeypatch, capsys):
    seen = []

    def fake_benchmark(size, **options):
        seen.append((size, options))
        ok = {"status": "ok", "seconds": 0.001, "reason": None}
        limited = {"status": "resource_limit", "seconds": None, "reason": "budget"}
        return {"n": size, "count": 1, "joint_signature_counts": [2, 2],
                "join_pair_counts": [1], "peak_intermediate_rows": 1,
                "engine_runs": {"dpll": ok, "lagrange": limited, "hermite": limited,
                                "sparse_contract": ok}}

    monkeypatch.setattr(benchmarks, "run_benchmark", fake_benchmark)
    assert benchmarks.main(["--engine", "v3", "--device", "cpu", "--repeats", "3"]) == 0
    assert [(options["family"], size) for size, options in seen] == [
        (family, n) for family in ("coupled", "bipartite") for n in (18, 22, 26, 30)]
    assert all(options["repeats"] == 3 and options["jet_nodes"] == 6 for _, options in seen)
    output = capsys.readouterr().out
    assert "sparse_ms" in output and "resource_limit" in output
    assert "miniature-only" in output and "cost estimates only" in output


@pytest.mark.parametrize("options", [{"repeats": 0}, {"torch_threads": 0}])
def test_v3_invalid_measurement_configuration(options):
    with pytest.raises(ValueError):
        benchmarks.run_v3_benchmark(6, **options)


@pytest.mark.parametrize('pushdown', [False, True])
def test_v31_benchmark_records_sparse_filter_and_peak_rows(pushdown):
    row = benchmarks.run_v3_benchmark(6, family='coupled', device='cpu',
                                       semi_join_pushdown=pushdown)
    stats = row['engine_stats']['sparse_contract']
    assert row['semi_join_pushdown'] is pushdown
    assert row['engine_options']['semi_join_pushdown'] is pushdown
    assert row['peak_frontier_rows'] == stats['peak_frontier_rows']
    assert row['peak_temporary_rows'] == stats['peak_temporary_rows']
    assert row['peak_joint_rows'] == stats['peak_joint_rows']
    assert row['peak_joint_rows'] >= max(row['joint_signature_counts'], default=0)
    assert row['phase1_assignment_counts'] == stats['phase1_assignment_counts']
    assert row['feasible_master_mask_counts'] == stats['feasible_master_mask_counts']
    assert bool(row['phase1_assignment_counts']) is pushdown
    assert row['engine_counts']['sparse_contract'] == row['reference_count']
