from math import prod
import pytest
from hypothesis import given, settings, strategies as st
from alg_sharpsat.cnf import CNF, Clause, Literal, brute_force_count
from alg_sharpsat.benchmarks import random_3sat, coupled_3sat, bipartite_3sat
from alg_sharpsat.stage1_sparsify import BoundaryLink, MeasurementBlock, PartitionedCNF
from alg_sharpsat.stage3_sparse_contract import SparseCounter, contract_sharp_sat, sparse_count_with_stats, auto_count_with_stats
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError

pytestmark = pytest.mark.g10

@pytest.mark.parametrize('groups', [2, 4])
@settings(max_examples=150, deadline=None, derandomize=True)
@given(n=st.integers(6, 12), seed=st.integers(0, 2**20), family=st.sampled_from(['random','coupled','bipartite']), planted=st.booleans())
def test_sparse_differential(groups, n, seed, family, planted):
    cnf = {'random': random_3sat, 'coupled': coupled_3sat, 'bipartite': bipartite_3sat}[family](n, seed=seed, planted=planted)
    assert contract_sharp_sat(cnf, num_groups=groups, device='cpu') == brute_force_count(cnf)

@pytest.mark.parametrize('groups', [1,2,4,8])
@pytest.mark.parametrize('cnf,expected', [(CNF(0,[]),1),(CNF(0,[Clause(())]),0),(CNF(100,[]),1<<100),(CNF(3,[Clause((Literal(0,True),Literal(0,False)))]),8)])
def test_empty_free_tautological(groups, cnf, expected):
    assert contract_sharp_sat(cnf, num_groups=groups, device='cpu') == expected

def test_master_shared_by_three_factors_survives_intermediate_join():
    p=PartitionedCNF(5, ((0,), (1,2), (3,), (4,)),
        ((), (Clause((Literal(1,True),Literal(2,True))),),
         (Clause((Literal(3,False),)),), (Clause((Literal(4,True),)),)),
        (BoundaryLink(0,0,2,1),BoundaryLink(0,0,3,2),BoundaryLink(1,1,4,3)),
        (MeasurementBlock((0,1),0),MeasurementBlock((2,),1)),2)
    counter=SparseCounter(p,device='cpu')
    assert counter.count()==1
    assert len(counter.stats.join_pair_counts)==3
    assert counter.stats.joint_signature_counts[1]==3

def test_joint_signatures_not_incoming_marginals():
    p=PartitionedCNF(4, ((0,3),(1,2)),
        ((Clause((Literal(0,False),Literal(3,True))),Clause((Literal(0,True),Literal(3,False)))),
         (Clause((Literal(1,False),Literal(2,False))),Clause((Literal(1,True),Literal(2,True))))),
        (BoundaryLink(0,0,2,1),BoundaryLink(1,1,3,0)),
        (MeasurementBlock((0,),0),MeasurementBlock((1,),1)),2)
    assert SparseCounter(p,device='cpu').count()==0

def test_safe_crt_path_for_more_than_62_active_local_variables():
    cnf=CNF(70,[Clause((Literal(i,True),)) for i in range(70)])
    value,stats=sparse_count_with_stats(cnf,num_groups=4,device='cpu')
    assert value==1 and stats.arithmetic=='modular_crt'
    assert prod(stats.primes)>1<<70

def test_auto_routes_large_boundary_to_sparse():
    cnf=bipartite_3sat(12,seed=0)
    count,stats=auto_count_with_stats(cnf,device='cpu')
    assert stats.boundary_links>4
    assert stats.selected_engine=='sparse_contract' and 'B>4' in stats.selection_reason
    assert count==brute_force_count(cnf)

def test_join_and_local_budgets_are_explicit():
    with pytest.raises(ResourceLimitError):
        contract_sharp_sat(bipartite_3sat(10),device='cpu',max_local_assignments=1)
    with pytest.raises(ResourceLimitError):
        contract_sharp_sat(bipartite_3sat(10),device='cpu',max_memory_bytes=1)

def test_no_reference_counter_fallback(monkeypatch):
    import alg_sharpsat.cnf as module
    def forbidden(*args,**kwargs): pytest.fail('reference counter called')
    monkeypatch.setattr(module,'dpll_count',forbidden)
    monkeypatch.setattr(module,'brute_force_count',forbidden)
    assert contract_sharp_sat(CNF(2,[Clause((Literal(0,True),Literal(1,True)))]),device='cpu')==3


def test_crt_contracts_large_partial_counts_with_four_primes():
    # 50 independent two-variable OR factors: 3**50 models. No enumeration
    # of the global Cartesian product, and no free-variable shortcut.
    groups = tuple((2*g, 2*g+1) for g in range(50))
    clauses = tuple((Clause((Literal(a, True), Literal(b, True))),) for a,b in groups)
    partitioned = PartitionedCNF(100, groups, clauses, (), (), 100)
    counter = SparseCounter(partitioned, device='cpu')
    assert counter.count() == 3**50
    assert counter.stats.arithmetic == 'modular_crt'
    assert len(counter.stats.primes) >= 4
    assert all(rows == 1 for rows in counter.stats.intermediate_rows)


@pytest.mark.parametrize('device', ['cuda', 'mps'])
def test_sparse_accelerator_when_available(device):
    import torch
    from alg_sharpsat.stage3_gpu_engine import UnsupportedDeviceError
    available = torch.cuda.is_available() if device == 'cuda' else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f'{device} unavailable')
    f = bipartite_3sat(10, seed=17)
    try:
        actual = contract_sharp_sat(f, device=device, num_groups=4)
    except UnsupportedDeviceError as exc:
        pytest.skip(str(exc))
    assert actual == brute_force_count(f)
