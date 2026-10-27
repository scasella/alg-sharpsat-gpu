"""Boundary key encoding stays exact across host dtype and layout changes."""
import pytest
import torch

from alg_sharpsat.cnf import CNF, Clause, Literal
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF, partition_and_decouple_boundary
from alg_sharpsat.stage3_sparse_contract import SparseCounter, SparseFactor


pytestmark = pytest.mark.g10


@pytest.mark.parametrize("masters,groups,internal", [(32, 2, False), (63, 3, True)])
def test_high_signature_bits_and_internal_position_fallback(masters, groups, internal):
    # A cycle of group-to-group aliases exercises bit31 (uint32 host keys)
    # and bit62 (signed int64). The extra internal variable forces the
    # compact position fallback because 63 signature bits already exist.
    original = masters + int(internal)
    variables = [list(range(g * (masters // groups), (g + 1) * (masters // groups)))
                 for g in range(groups)]
    if internal:
        variables[0].append(masters)
    links = []
    for master in range(masters):
        source = master // (masters // groups)
        target = (source + 1) % groups
        proxy = original + master
        variables[target].append(proxy)
        links.append(BoundaryLink(master, source, proxy, target))
    clauses = tuple(tuple(Clause((Literal(v, True),)) for v in group) for group in variables)
    blocks = tuple(MeasurementBlock(tuple(i for i, link in enumerate(links)
                                         if link.master_group == g), g) for g in range(groups))
    partitioned = PartitionedCNF(original + masters, tuple(map(tuple, variables)), clauses,
                                tuple(links), blocks, original)
    counter = SparseCounter(partitioned, device="cpu")
    assert all(f.keys.tolist() == [f.scope] for f in counter.factors)
    assert counter.count() == 1


@pytest.mark.parametrize("prime", [None, 5])
def test_large_sorted_intersection_retains_weighted_matches(prime):
    counter = SparseCounter(partition_and_decouple_boundary(CNF(0, [])), device="cpu")
    scope = (1 << 14) - 1
    a = torch.arange(0, 10000, 2, dtype=torch.int64)
    b = torch.arange(3, 16000, 3, dtype=torch.int64)
    left = SparseFactor(scope, a, torch.full_like(a, 2))
    right = SparseFactor(scope, b, torch.full_like(b, 3))
    result = counter._join(left, right, 0, prime, False)
    expected = 6 * len(range(6, 10000, 6))
    assert result.keys.tolist() == [0]
    assert result.counts.tolist() == [expected if prime is None else expected % prime]


def test_explicit_unsupported_sparse_kernel_uses_public_device_error(monkeypatch):
    from alg_sharpsat.stage3_gpu_engine import UnsupportedDeviceError
    def unavailable(*args, **kwargs):
        raise NotImplementedError('missing sparse kernel')
    monkeypatch.setattr(torch, 'searchsorted', unavailable)
    with pytest.raises(UnsupportedDeviceError, match='sparse key search'):
        SparseCounter(partition_and_decouple_boundary(CNF(0, [])), device='cpu')
