"""Adversarial arithmetic and conditioned/unconditioned tensor parity."""
import pytest
import torch

from alg_sharpsat.cnf import CNF, Clause, Literal
from alg_sharpsat.stage1_sparsify import (
    BoundaryLink, MeasurementBlock, PartitionedCNF, partition_and_decouple_boundary,
)
from alg_sharpsat.stage2_algebraic import SuppressionSchedule, direct_h
from alg_sharpsat.stage3_gpu_engine import AlgebraicCounter, ResourceLimitError, UnsupportedDeviceError
import alg_sharpsat.stage3_gpu_engine as engine


@pytest.mark.parametrize("prime", [998244353, 1004535809, 469762049])
def test_near_modulus_long_dot_matches_python_and_limb_backend(prime):
    partitioned = partition_and_decouple_boundary(CNF(0, []))
    counter = AlgebraicCounter(partitioned, device="cpu")
    inner = 2301  # A direct int64 dot of these products overflows repeatedly.
    left = torch.tensor([[prime - 1 - (k + row) % 13 for k in range(inner)]
                         for row in range(3)], dtype=torch.int64)
    right = torch.tensor([[prime - 1 - (k + col) % 17 for col in range(5)]
                          for k in range(inner)], dtype=torch.int64)
    expected = [[sum(int(left[row, k]) * int(right[k, col]) for k in range(inner)) % prime
                 for col in range(5)] for row in range(3)]
    assert counter._matmul_mod(left, right, prime).tolist() == expected
    counter.matmul_backend = "limbs"
    assert counter._matmul_mod(left, right, prime).tolist() == expected


def _repeated_master_partition():
    # Both masters are linked to two targets. The one block's measurement is
    # 2*x0 + 2*x1, so all-suppressed conditioning merges source rows 01 and 10.
    return PartitionedCNF(
        7, ((0, 1), (2, 3, 4), (5, 6)),
        ((), (Clause((Literal(3, True), Literal(2, True))),
              Clause((Literal(4, True), Literal(2, False)))),
         (Clause((Literal(5, True), Literal(6, True))),)),
        (BoundaryLink(0, 0, 3, 1), BoundaryLink(1, 0, 4, 1),
         BoundaryLink(0, 0, 5, 2), BoundaryLink(1, 0, 6, 2)),
        (MeasurementBlock((0, 1, 2, 3), 0),), 3,
    )


def test_measurement_conditioning_collapses_rows_and_preserves_exact_h():
    partitioned = _repeated_master_partition()
    schedule = SuppressionSchedule((1, 2, 3, 4, 5), ((1,), (1,), (1,), (1,)))
    counter = AlgebraicCounter(partitioned, device="cpu")
    assert len(counter.tables[0].source_bits) == 4
    for t in schedule.nodes:
        conditioned = counter.evaluate(t, 998244353, schedule)
        assert conditioned == counter.evaluate(t, 998244353, schedule, conditioned=False)
        assert conditioned == direct_h(partitioned, t, 998244353, schedule)
    assert counter.stats.conditioning_states[0][0] == 3
    assert counter.stats.residual_active_bits[0][0] == 0
    assert counter.stats.conditioning_states[1][0] == 4
    assert counter.stats.residual_active_bits[1][0] == 2


def test_repeated_master_retained_when_any_of_its_links_is_active():
    partitioned = _repeated_master_partition()
    schedule = SuppressionSchedule((1, 2, 3, 4, 5), ((1,), (1,), (2,), (2,)))
    counter = AlgebraicCounter(partitioned, device="cpu")
    actual = counter.evaluate(1, 998244353, schedule)
    assert actual == direct_h(partitioned, 1, 998244353, schedule)
    assert counter.stats.residual_active_bits[-1][0] == 2
    assert counter.stats.conditioning_states[-1][0] == 4


def test_auto_falls_back_but_explicit_unsupported_device_fails(monkeypatch):
    def mock_probe(name, backend):
        if name != "cpu":
            raise UnsupportedDeviceError(f"{name}: mock missing int64 remainder")
    monkeypatch.setattr(engine, "_probe_device", mock_probe)
    device, reasons = engine.select_device("auto")
    assert device.type == "cpu"
    assert len(reasons) == 2
    with pytest.raises(UnsupportedDeviceError, match="mock missing int64 remainder"):
        engine.select_device("cuda")


def test_signature_encoding_obeys_memory_budget_before_allocation():
    links = tuple(BoundaryLink(i, 0, 12 + i, 1) for i in range(12))
    partitioned = PartitionedCNF(
        24, (tuple(range(12)), tuple(range(12, 24))), ((), ()), links,
        tuple(MeasurementBlock(tuple(range(start, start + 4)), 0)
              for start in range(0, 12, 4)), 12,
    )
    # A single 4096x12 int64 signature matrix would exceed this limit.
    with pytest.raises(ResourceLimitError, match="budget"):
        AlgebraicCounter(partitioned, device="cpu", max_memory_bytes=100_000)


def test_conditioning_workspace_obeys_memory_budget():
    counter = AlgebraicCounter(_repeated_master_partition(), device="cpu")
    counter.max_memory_bytes = counter._persistent_bytes + 10
    schedule = SuppressionSchedule((1, 2, 3, 4, 5), ((1,),) * 4)
    with pytest.raises(ResourceLimitError, match="conditioning workspace"):
        counter.evaluate(1, 998244353, schedule)
