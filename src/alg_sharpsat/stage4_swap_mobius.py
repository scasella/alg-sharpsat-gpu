"""Executable miniature SWAP, spectral measurements, and Boolean Mobius jets.

The default miniature keeps clause occurrences coherent and evaluates original
clause factors through shared exact measurements. The older multilinear
expansion counter remains available as an independent finite reference. Neither
path implements the manuscript's asymptotic parameter selection or reductions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from itertools import product, combinations
from math import comb, prod
from time import perf_counter
from typing import Sequence

from .cnf import CNF
from .stage2_algebraic import crt_reconstruct, primes_for_bound, validate_prime
from .stage2_hermite import interpolate_hermite_at_zero
from .stage3_gpu_engine import ResourceLimitError


def _mul(a, b):
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i+j] += x*y
    return out


@dataclass(frozen=True)
class ParityPolynomial:
    coefficients: tuple[Fraction, ...]
    window: tuple[int, ...]

    def __call__(self, z):
        value = Fraction(0)
        for coefficient in reversed(self.coefficients):
            value = value*z + coefficient
        return value


def typical_weight_polynomial(L: int, b: int, p_prob: float = 0.25) -> ParityPolynomial:
    if type(L) is not int or type(b) is not int or L < 2 or not 1 <= b < L:
        raise ValueError('require L>=2 and 1<=b<L')
    probability = Fraction(str(p_prob)) if not isinstance(p_prob, Fraction) else p_prob
    if not 0 <= probability <= 1:
        raise ValueError('p_prob must lie in [0,1]')
    center = L*probability - Fraction(b,2)
    # Nearest feasible interval midpoint; lower start wins exact ties.
    a = min(range(L-b+1), key=lambda start: (abs(Fraction(start)-center),start))
    result = [Fraction(0)]*(b+1)
    binomial = [Fraction(1)]
    for j in range(b+1):
        scale = (-1)**a * (-2)**j
        for i, value in enumerate(binomial):
            result[i] += scale*value
        binomial = [v/Fraction(j+1) for v in _mul(binomial, [Fraction(-a-j),Fraction(1)])]
    return ParityPolynomial(tuple(result), tuple(range(a,a+b+1)))


@dataclass(frozen=True)
class SwapSystem:
    d: int
    L: int
    b: int
    states: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    M: tuple[tuple[Fraction, ...], ...]
    C: tuple[tuple[int, ...], ...]
    eigenvectors: tuple[tuple[int, ...], ...]
    eigenvalues: tuple[Fraction, ...]
    norms: tuple[int, ...]
    error_indices: tuple[int, ...]
    polynomial: ParityPolynomial

    @property
    def N(self):
        return len(self.states)

    @property
    def s0(self):
        return len(self.C)

    def group(self, node: int, label: int) -> tuple[int, ...]:
        if not 1 <= node <= self.L or not 0 <= label < self.d:
            raise ValueError('invalid node or group label')
        return tuple(i for i,(alpha,_) in enumerate(self.states) if alpha[node-1] == label)

    def mismatch(self, omega: int, eta: int) -> tuple[int, ...]:
        return tuple(t+1 for t in range(self.L)
                     if self.states[eta][0][t] != self.states[omega][1][t])


@lru_cache(maxsize=16)
def build_swap_matrix(d: int, L: int, b: int) -> SwapSystem:
    if type(d) is not int or d < 2:
        raise ValueError('d must be an integer >=2')
    if type(L) is not int or type(b) is not int or L < 2 or not 1 <= b < L:
        raise ValueError('require L>=2 and 1<=b<L')
    # Bound dimension before constructing either huge integers or polynomials.
    if L > 4 or d > 4 or d**(2*L) > 256:
        raise ResourceLimitError('explicit SWAP reference supports N<=256')
    N=d**(2*L)
    polynomial = typical_weight_polynomial(L,b,Fraction(d-1,2*d))
    words=tuple(product(range(d),repeat=L))
    states=tuple(product(words,words))
    indices={state:i for i,state in enumerate(states)}
    # Group algebra of commuting involutions: S_a S_b = S_(a xor b).
    W={0:Fraction(L,2), **{1<<t:Fraction(-1,2) for t in range(L)}}
    power={0:Fraction(1)}
    operator={}
    for coefficient in polynomial.coefficients:
        for mask,value in power.items():
            operator[mask]=operator.get(mask,Fraction(0))+coefficient*value
        next_power={}
        for a,x in power.items():
            for c,y in W.items():
                next_power[a^c]=next_power.get(a^c,Fraction(0))+x*y
        power=next_power
    full=(1<<L)-1
    operator={mask^full:value for mask,value in operator.items() if value}
    matrix=[[Fraction(0)]*N for _ in range(N)]
    for column,(alpha,beta) in enumerate(states):
        for mask,weight in operator.items():
            a=tuple(beta[t] if mask>>t&1 else alpha[t] for t in range(L))
            c=tuple(alpha[t] if mask>>t&1 else beta[t] for t in range(L))
            matrix[indices[a,c]][column]+=weight
    modes=[({(a,a):1},0) for a in range(d)]
    for a in range(d):
        for c in range(a+1,d):
            modes.extend([({(a,c):1,(c,a):1},0),({(a,c):1,(c,a):-1},1)])
    vectors,eigenvalues,norms=[],[],[]
    for factors in product(modes,repeat=L):
        vector=[0]*N
        for entries in product(*(tuple(mode.items()) for mode,_ in factors)):
            alpha=tuple(entry[0][0] for entry in entries)
            beta=tuple(entry[0][1] for entry in entries)
            vector[indices[alpha,beta]]=prod(entry[1] for entry in entries)
        k=sum(sign for _,sign in factors)
        vectors.append(tuple(vector))
        eigenvalues.append((-1)**k*polynomial(k))
        norms.append(sum(value*value for value in vector))
    errors=tuple(i for i,value in enumerate(eigenvalues) if value!=1)
    return SwapSystem(d,L,b,states,tuple(map(tuple,matrix)),tuple(vectors[i] for i in errors),
                      tuple(vectors),tuple(eigenvalues),tuple(norms),errors,polynomial)


def _field(value, prime):
    value=Fraction(value)
    if value.denominator % prime == 0:
        raise ValueError('rational coefficient denominator is zero modulo prime')
    return value.numerator * pow(value.denominator,-1,prime) % prime


@lru_cache(maxsize=4096)
def _replacement_row(d: int,L: int,b: int,omega: int):
    system=build_swap_matrix(d,L,b)
    if type(omega) is not int or not 0<=omega<system.N:
        raise ValueError('omega must be a basis-state index')
    coefficients=[[Fraction(0)]*system.N for _ in range(b+1)]
    coefficients[0][omega]=Fraction(1)
    # K(T) = I + sum M_omegaeta*(phi_omegaeta(T)-1).
    for eta,weight in enumerate(system.M[omega]):
        if not weight:
            continue
        phi=[Fraction(1)]
        for node in system.mismatch(omega,eta):
            phi=_mul(phi,[Fraction(1),Fraction(-1,node)])
        if len(phi)>b+1:
            raise ArithmeticError('SWAP support violated the degree bound')
        for degree in range(1,len(phi)):
            coefficients[degree][eta]=weight*phi[degree]
    return tuple(map(tuple,coefficients))


def replacement_coefficients(system: SwapSystem, omega: int, prime: int | None=None):
    coefficients=_replacement_row(system.d,system.L,system.b,omega)
    if prime is None:
        return coefficients
    validate_prime(prime)
    return tuple(tuple(_field(value,prime) for value in row) for row in coefficients)


def eval_replacement_form(system: SwapSystem, omega: int, T, z: Sequence,
                          prime: int | None=None):
    if len(z)!=system.N:
        raise ValueError('z must have one entry for every Omega coordinate')
    coefficients=replacement_coefficients(system,omega,prime)
    total=0
    for row in reversed(coefficients):
        total=total*T+sum(a*x for a,x in zip(row,z))
        if prime is not None:
            total%=prime
    return total


def error_from_measurements(system: SwapSystem, measurements: Sequence,
                            prime: int | None=None):
    if len(measurements)!=system.s0:
        raise ValueError('measurement dimension mismatch')
    out=[Fraction(0)]*system.N if prime is None else [0]*system.N
    for row,index,value in zip(system.C,system.error_indices,measurements):
        scale=(1-system.eigenvalues[index])/system.norms[index]
        scale=scale if prime is None else _field(scale,prime)
        for omega,entry in enumerate(row):
            out[omega]+=scale*entry*value
            if prime is not None:
                out[omega]%=prime
    return out


def _boolean_product(left: dict[int,int],right: dict[int,int]):
    out={}
    for a,x in left.items():
        for b,y in right.items():
            out[a|b]=out.get(a|b,0)+x*y
    return {mask:value for mask,value in out.items() if value}


def cnf_indicator_polynomial(cnf: CNF, max_terms: int=4096) -> dict[int,int]:
    """Exact multilinear Boolean indicator, with explicit exponential storage."""
    cnf.validate()
    terms={0:1}
    for clause in cnf.clauses:
        unsatisfied={0:1}
        for literal in clause.literals:
            bit=1<<literal.var
            factor={0:1,bit:-1} if literal.pos else {bit:1}
            unsatisfied=_boolean_product(unsatisfied,factor)
            if len(unsatisfied)>max_terms:
                raise ResourceLimitError('clause indicator exceeds max_terms')
        factor={mask:-value for mask,value in unsatisfied.items()}
        factor[0]=factor.get(0,0)+1
        terms=_boolean_product(terms,{mask:value for mask,value in factor.items() if value})
        if len(terms)>max_terms:
            raise ResourceLimitError('multilinear CNF indicator exceeds max_terms')
    return terms


def _jet_mul(left,right,q,prime):
    out=[0]*(q+1)
    for i,a in enumerate(left):
        for j in range(min(len(right),q-i+1)):
            out[i+j]=(out[i+j]+a*right[j])%prime
    return out


def mobius_diagonal_jet(num_vars: int,q: int,prime: int,evaluator):
    """General Lemma 7.1 identity; evaluator(x_mask,y_mask) returns a jet.

    This independent diagnostic enumerates x. The model counter below uses
    group-separated measurement buckets instead of this diagnostic routine.
    """
    result=[0]*(q+1)
    for size in range(min(q,num_vars)+1):
        for indices in combinations(range(num_vars),size):
            A=sum(1<<i for i in indices)
            B=A
            while True:
                sign=(-1)**(size-B.bit_count())
                for x in range(1<<num_vars):
                    if x&A==A:
                        jet=evaluator(x,B)
                        for j,value in enumerate(jet[:q+1]):
                            result[j]=(result[j]+sign*value)%prime
                if B==0: break
                B=(B-1)&A
    return result


@dataclass
class SwapStats:
    d: int
    L: int
    b: int
    dimension: int
    measurement_rows: int
    variables: int
    polynomial_terms: int
    degree_bound: int
    nodes: tuple[int,...]
    orders: tuple[int,...]
    embedding: tuple[int,...]
    primes: tuple[int,...]=()
    node_evaluations: list[dict]=field(default_factory=list)
    local_assignment_visits: int=0
    inner_evaluations: int=0
    build_seconds: float=0.0
    evaluation_seconds: float=0.0


class SwapMobiusCounter:
    """Miniature group-separated diagonal-sum reference (default n<=6).

    CNF variables occupy an injective subset of Omega; all remaining coordinates
    are fixed zero. The multilinear indicator expansion, measurement grids and
    Mobius subsets are explicit costs, not hidden in a brute-force fallback.
    """
    def __init__(self,cnf: CNF,*,d: int=2,L: int=2,b: int=1,
                 embedding: Sequence[int] | None=None,max_variables: int=6,
                 max_terms: int=4096,max_work: int=2_000_000):
        cnf.validate()
        if any(type(v) is not int or v<=0 for v in (max_variables,max_terms,max_work)):
            raise ValueError('resource bounds must be positive integers')
        if cnf.num_vars>max_variables:
            raise ResourceLimitError('SWAP-Mobius miniature exceeds max_variables')
        start=perf_counter()
        self.system=build_swap_matrix(d,L,b)
        self.n=cnf.num_vars
        if self.n>self.system.N:
            raise ValueError('Omega cannot hold all CNF variables')
        step=2*d**L-1
        self.embedding=tuple((i*step)%self.system.N for i in range(self.n)) if embedding is None else tuple(embedding)
        if len(self.embedding)!=self.n or len(set(self.embedding))!=self.n or any(type(i) is not int or not 0<=i<self.system.N for i in self.embedding):
            raise ValueError('embedding must inject each variable into Omega')
        self.terms=cnf_indicator_polynomial(cnf,max_terms)
        self.degree=b*max((mask.bit_count() for mask in self.terms),default=0)
        # There can be more data than necessary when D+1<L; use a prefix of
        # the design nodes, while replacement locality holds at ALL L nodes.
        count=min(L,self.degree+1)
        nodes=tuple(range(1,count+1))
        capacity,extra=divmod(self.degree+1,count)
        orders=tuple(capacity-1+(i<extra) for i in range(count))
        self.max_work=max_work
        self.stats=SwapStats(d,L,b,self.system.N,self.system.s0,self.n,len(self.terms),
                             self.degree,nodes,orders,self.embedding)
        self.stats.build_seconds=perf_counter()-start

    def _tick(self,trace):
        trace['local_assignment_visits']+=1
        if trace['local_assignment_visits']>self.max_work:
            raise ResourceLimitError('SWAP-Mobius local partition work exceeds max_work')

    def evaluate_jet(self,node: int,q: int,prime: int) -> list[int]:
        validate_prime(prime)
        if type(node) is not int or not 1<=node<=self.system.L or prime<=self.system.L:
            raise ValueError('node must be in1..L and prime>L')
        if type(q) is not int or q<0:
            raise ValueError('q must be a nonnegative integer')
        system=self.system
        if (q+1)*max(1,self.n*self.n) > self.max_work:
            raise ResourceLimitError('SWAP-Mobius jet storage exceeds max_work')
        trace=dict(node=node,q=q,prime=prime,local_assignment_visits=0,inner_evaluations=0,
                   measurement_tuples=0,forced_subsets=0,local_group_sizes=[],status='pending')
        self.stats.node_evaluations.append(trace)
        groups=[tuple(i for i,omega in enumerate(self.embedding) if system.states[omega][0][node-1]==g)
                for g in range(system.d)]
        if sum(1 << len(group) for group in groups) > self.max_work:
            trace['status']='resource_limit'
            raise ResourceLimitError('SWAP-Mobius local assignment storage exceeds max_work')
        if sum(comb(self.n,j)*(1 << j) for j in range(min(q,self.n)+1)) > self.max_work:
            trace['status']='resource_limit'
            raise ResourceLimitError('SWAP-Mobius subset enumeration exceeds max_work')
        targets=[system.states[omega][1][node-1] for omega in self.embedding]
        trace['local_group_sizes']=list(map(len,groups))
        # Taylor coefficient matrices restricted to active input coordinates.
        jets=[]
        for omega in self.embedding:
            coefficients=replacement_coefficients(system,omega,prime)
            jets.append([[sum(comb(m,j)*pow(node,m-j,prime)*coefficients[m][eta]
                              for m in range(j,system.b+1))%prime if j<=system.b else 0
                          for eta in self.embedding] for j in range(q+1)])
        error_coeff=[]
        for omega in self.embedding:
            error_coeff.append([_field((1-system.eigenvalues[index])/system.norms[index],prime)*row[omega]%prime
                                for row,index in zip(system.C,system.error_indices)])
        local_constant=[]
        for i,omega in enumerate(self.embedding):
            row=[(jets[i][0][u] - _field(int(omega==eta)-system.M[omega][eta],prime))%prime
                 for u,eta in enumerate(self.embedding)]
            if any(value and u not in groups[targets[i]] for u,value in enumerate(row)):
                raise ArithmeticError('replacement form has nonzero residual outside its target group')
            local_constant.append(row)
        term_factors={mask: [[i for i in range(self.n) if mask>>i&1 and targets[i]==g]
                             for g in range(system.d)] for mask in self.terms}
        # Integer measurement tuples, never finite-field hashes.
        assignments=[]
        for variables in groups:
            rows=[]
            for bits in product((0,1),repeat=len(variables)):
                mask=sum(bit<<v for v,bit in zip(variables,bits))
                measurement=tuple(sum(row[self.embedding[v]]*bit for v,bit in zip(variables,bits)) for row in system.C)
                rows.append((mask,measurement))
            assignments.append(rows)
        trace['local_assignment_rows']=list(map(len,assignments))
        result=[0]*(q+1)
        try:
            for size in range(min(q,self.n)+1):
                for selected in combinations(range(self.n),size):
                    A=sum(1<<v for v in selected)
                    trace['forced_subsets']+=1
                    buckets=[]
                    for variables,rows in zip(groups,assignments):
                        forced=A&sum(1<<v for v in variables)
                        group_buckets={}
                        for mask,measurement in rows:
                            if mask&forced==forced:
                                group_buckets.setdefault(measurement,[]).append(mask)
                        buckets.append(group_buckets)
                    B=A
                    while True:
                        sign=(-1)**(size-B.bit_count())
                        increment=[[0]+[sum(jets[i][j][u] for u in range(self.n) if B>>u&1)%prime
                                         for j in range(1,q+1)] for i in range(self.n)]
                        trace['inner_evaluations']+=1
                        for keys in product(*(tuple(bucket) for bucket in buckets)):
                            trace['measurement_tuples']+=1
                            measurement=tuple(sum(parts) for parts in zip(*keys))
                            errors=[sum(a*m for a,m in zip(row,measurement))%prime for row in error_coeff]
                            for monomial,weight in self.terms.items():
                                combined=[1]+[0]*q
                                for g,variables in enumerate(groups):
                                    local=[0]*(q+1)
                                    factors=term_factors[monomial][g]
                                    for mask in buckets[g][keys[g]]:
                                        self._tick(trace)
                                        value=[1]+[0]*q
                                        for i in factors:
                                            base=(errors[i]+sum(local_constant[i][u] for u in variables if mask>>u&1))%prime
                                            value=_jet_mul(value,[base]+increment[i][1:],q,prime)
                                        local=[(a+c)%prime for a,c in zip(local,value)]
                                    combined=_jet_mul(combined,local,q,prime)
                                result=[(a+sign*weight*c)%prime for a,c in zip(result,combined)]
                        if B==0: break
                        B=(B-1)&A
            trace['status']='ok'
        except ResourceLimitError:
            trace['status']='resource_limit'
            raise
        self.stats.local_assignment_visits+=trace['local_assignment_visits']
        self.stats.inner_evaluations+=trace['inner_evaluations']
        return result

    def count(self) -> int:
        start=perf_counter()
        self.stats.primes=primes_for_bound(1<<self.n)
        residues=[]
        for prime in self.stats.primes:
            jets=[self.evaluate_jet(t,q,prime) for t,q in zip(self.stats.nodes,self.stats.orders)]
            residues.append(interpolate_hermite_at_zero(jets,list(self.stats.nodes),list(self.stats.orders),prime))
        count=crt_reconstruct(residues,self.stats.primes)
        if count>1<<self.n:
            raise ArithmeticError('SWAP-Mobius reconstruction exceeds model-count bound')
        self.stats.evaluation_seconds=perf_counter()-start
        return count


@dataclass(frozen=True)
class SwapOccurrence:
    block: int
    omega: int
    var: int
    pos: bool


@dataclass
class OccurrenceSwapStats(SwapStats):
    block_count: int=0
    clause_count: int=0
    occurrence_count: int=0
    raw_measurement_rows: int=0
    representation: str='occurrence_coherent_clause_factors'


class OccurrenceSwapCounter:
    """Bounded occurrence-specific SWAP design and separated clause evaluator.

    Every occurrence of a variable has the same first register word. Every
    occurrence within a clause has the same second register word. Conflicting
    auxiliary cells go in separate blocks; occupied cells always read their
    original Boolean variable. No global CNF polynomial expansion is formed.
    ``max_terms`` is a compatibility alias for the clause-count ``max_factors``
    budget; an explicit ``max_factors`` takes precedence.
    """
    def __init__(self,cnf: CNF,*,d: int=2,L: int=2,b: int=1,
                 max_variables: int=6,max_work: int=2_000_000,
                 max_terms: int=4096,max_factors: int | None=None):
        cnf.validate()
        if any(type(v) is not int or v<=0 for v in (max_variables,max_work,max_terms)):
            raise ValueError('resource bounds must be positive integers')
        factor_limit=max_terms if max_factors is None else max_factors
        if type(factor_limit) is not int or factor_limit<=0:
            raise ValueError('max_factors must be a positive integer')
        if cnf.num_vars>max_variables:
            raise ResourceLimitError('SWAP-Mobius miniature exceeds max_variables')
        if len(cnf.clauses)>factor_limit:
            raise ResourceLimitError('SWAP occurrence clause count exceeds max_factors')
        start=perf_counter()
        self.system=build_swap_matrix(d,L,b)
        self.n,self.max_work=cnf.num_vars,max_work
        words=tuple(product(range(d),repeat=L))
        width=len(words)
        self.alpha=tuple(words[i%width] for i in range(self.n))
        # Boolean-equivalent clause simplification does not expand the product.
        self.clauses=[]
        for clause in cnf.clauses:
            literals=sorted(set((lit.var,lit.pos) for lit in clause.literals))
            if not any((v,not sign) in literals for v,sign in literals):
                self.clauses.append(tuple(literals))
        self.beta=tuple(words[c%width] for c in range(len(self.clauses)))
        occurrences=sum(map(len,self.clauses))
        if occurrences*max(1,self.n)*(b+1)>max_work:
            raise ResourceLimitError('SWAP occurrence replacement storage exceeds max_work')
        self.blocks=[]
        self.occurrences=[]
        for c,clause in enumerate(self.clauses):
            placed=[]
            for var,pos in clause:
                omega=(var%width)*width+c%width
                block=next((j for j,cells in enumerate(self.blocks)
                            if omega not in cells or cells[omega]==var),len(self.blocks))
                if block==len(self.blocks):
                    self.blocks.append({})
                self.blocks[block][omega]=var
                placed.append(SwapOccurrence(block,omega,var,pos))
            self.occurrences.append(tuple(placed))
        if len(self.blocks)*self.system.s0*max(1,self.n)>max_work:
            raise ResourceLimitError('SWAP expanded measurement storage exceeds max_work')
        self.measurement_rows=[]
        self.measurement_indices={}
        for block,cells in enumerate(self.blocks):
            for index,row in enumerate(self.system.C):
                expanded=[0]*self.n
                for eta,var in cells.items():
                    expanded[var]+=row[eta]
                if any(expanded):
                    self.measurement_indices[block,index]=len(self.measurement_rows)
                    self.measurement_rows.append(tuple(expanded))
        if len(self.measurement_rows)*max(1,self.n)>max_work:
            raise ResourceLimitError('SWAP expanded measurement storage exceeds max_work')
        self.degree=b*occurrences
        count=min(L,self.degree+1)
        capacity,extra=divmod(self.degree+1,count)
        nodes=tuple(range(1,count+1))
        orders=tuple(capacity-1+(i<extra) for i in range(count))
        self.stats=OccurrenceSwapStats(
            d,L,b,self.system.N,len(self.measurement_rows),self.n,0,self.degree,
            nodes,orders,(),block_count=len(self.blocks),clause_count=len(self.clauses),
            occurrence_count=occurrences,raw_measurement_rows=len(self.blocks)*self.system.s0)
        self.stats.build_seconds=perf_counter()-start

    def occurrence_coefficients(self,occurrence: SwapOccurrence,prime: int | None=None):
        """Expand one block replacement into original-variable coordinates."""
        rows=replacement_coefficients(self.system,occurrence.omega,prime)
        expanded=[[0]*self.n for _ in rows]
        for degree,row in enumerate(rows):
            for eta,var in self.blocks[occurrence.block].items():
                expanded[degree][var]+=row[eta]
                if prime is not None:
                    expanded[degree][var]%=prime
        return tuple(map(tuple,expanded))

    def _multiply(self,left,right,q,prime,trace):
        size=min(q+1,len(left)+len(right)-1)
        trace['arithmetic_work']+=sum(min(len(right),size-i) for i in range(min(size,len(left))))
        if trace['arithmetic_work']>self.max_work:
            raise ResourceLimitError('SWAP occurrence arithmetic exceeds max_work')
        out=[0]*size
        for i,a in enumerate(left):
            for j in range(min(len(right),size-i)):
                out[i+j]=(out[i+j]+a*right[j])%prime
        while len(out)>1 and out[-1]==0:
            out.pop()
        return out

    def evaluate_jet(self,node: int,q: int,prime: int) -> list[int]:
        validate_prime(prime)
        if type(node) is not int or not 1<=node<=self.system.L or prime<=self.system.L:
            raise ValueError('node must be in 1..L and prime>L')
        if type(q) is not int or q<0:
            raise ValueError('q must be a nonnegative integer')
        total_occurrences=self.stats.occurrence_count
        jet_cells=(q+1)*max(1,total_occurrences*self.n)
        error_cells=total_occurrences*len(self.measurement_rows)
        local_cells=total_occurrences*self.n
        if jet_cells+error_cells+local_cells>self.max_work:
            raise ResourceLimitError('SWAP occurrence jet storage exceeds max_work')
        groups=[tuple(v for v,alpha in enumerate(self.alpha) if alpha[node-1]==g)
                for g in range(self.system.d)]
        if sum(1<<len(group) for group in groups)>self.max_work:
            raise ResourceLimitError('SWAP occurrence local assignment storage exceeds max_work')
        pairs=sum(comb(self.n,j)*(1<<j) for j in range(min(q,self.n)+1))
        if pairs>self.max_work:
            raise ResourceLimitError('SWAP occurrence subset enumeration exceeds max_work')
        trace=dict(node=node,q=q,prime=prime,status='pending',local_assignment_visits=0,
                   arithmetic_work=0,inner_evaluations=0,forced_subsets=0,
                   measurement_tuples=0,local_group_sizes=list(map(len,groups)),
                   group_assignment_visits=[0]*self.system.d,
                   local_assignment_rows=[1<<len(group) for group in groups])
        self.stats.node_evaluations.append(trace)
        targets=[beta[node-1] for beta in self.beta]
        prepared=[]
        for c,clause in enumerate(self.occurrences):
            prepared_clause=[]
            for occurrence in clause:
                coefficients=self.occurrence_coefficients(occurrence,prime)
                jets=[[sum(comb(m,j)*pow(node,m-j,prime)*coefficients[m][v]
                           for m in range(j,self.system.b+1))%prime
                       for v in range(self.n)] for j in range(min(q,self.system.b)+1)]
                error=[0]*len(self.measurement_rows)
                for cindex,index in enumerate(self.system.error_indices):
                    measurement_index=self.measurement_indices.get((occurrence.block,cindex))
                    if measurement_index is not None:
                        error[measurement_index]=_field((1-self.system.eigenvalues[index])/
                            self.system.norms[index],prime)*self.system.C[cindex][occurrence.omega]%prime
                local=list(jets[0])
                for v in range(self.n):
                    local[v]=(local[v]-sum(a*row[v] for a,row in zip(error,self.measurement_rows)))%prime
                target=targets[c]
                if any(value and v not in groups[target] for v,value in enumerate(local)):
                    raise ArithmeticError('occurrence replacement violates assigned clause locality')
                prepared_clause.append((occurrence.pos,jets,error,local))
            prepared.append(prepared_clause)
        assignments=[]
        for variables in groups:
            rows=[]
            for bits in product((0,1),repeat=len(variables)):
                mask=sum(bit<<v for v,bit in zip(variables,bits))
                measurement=tuple(sum(row[v]*bit for v,bit in zip(variables,bits))
                                  for row in self.measurement_rows)
                rows.append((mask,measurement))
            assignments.append(rows)
        result=[0]*(q+1)
        try:
            for size in range(min(q,self.n)+1):
                for selected in combinations(range(self.n),size):
                    A=sum(1<<v for v in selected)
                    trace['forced_subsets']+=1
                    buckets=[]
                    for g,rows in enumerate(assignments):
                        bucket={}
                        forced=A&sum(1<<v for v in groups[g])
                        for mask,measurement in rows:
                            if mask&forced==forced:
                                bucket.setdefault(measurement,[]).append(mask)
                        buckets.append(bucket)
                    B=A
                    while True:
                        trace['inner_evaluations']+=1
                        increments=[[[0]+[sum(row[v] for v in range(self.n) if B>>v&1)%prime
                                          for row in jets[1:]] for _,jets,_,_ in clause]
                                    for clause in prepared]
                        for keys in product(*(tuple(bucket) for bucket in buckets)):
                            trace['measurement_tuples']+=1
                            measurement=tuple(sum(parts) for parts in zip(*keys))
                            combined=[1]
                            for g,variables in enumerate(groups):
                                local_sum=[0]*(q+1)
                                for mask in buckets[g][keys[g]]:
                                    trace['local_assignment_visits']+=1
                                    trace['group_assignment_visits'][g]+=1
                                    if trace['local_assignment_visits']>self.max_work:
                                        raise ResourceLimitError('SWAP occurrence local visits exceed max_work')
                                    value=[1]
                                    for c,clause in enumerate(prepared):
                                        if targets[c]!=g:
                                            continue
                                        unsatisfied=[1]
                                        for i,(pos,_,error,local) in enumerate(clause):
                                            jet=list(increments[c][i])
                                            jet[0]=(sum(a*m for a,m in zip(error,measurement))+
                                                    sum(local[v] for v in variables if mask>>v&1))%prime
                                            # Unsatisfied positive literal is 1-K; negative is K.
                                            if pos:
                                                jet=[(-a)%prime for a in jet]
                                                jet[0]=(jet[0]+1)%prime
                                            unsatisfied=self._multiply(unsatisfied,jet,q,prime,trace)
                                        factor=[(-a)%prime for a in unsatisfied]
                                        factor[0]=(factor[0]+1)%prime
                                        value=self._multiply(value,factor,q,prime,trace)
                                    for j,a in enumerate(value):
                                        local_sum[j]=(local_sum[j]+a)%prime
                                combined=self._multiply(combined,local_sum,q,prime,trace)
                            sign=(-1)**(size-B.bit_count())
                            for j,a in enumerate(combined):
                                result[j]=(result[j]+sign*a)%prime
                        if B==0:
                            break
                        B=(B-1)&A
            trace['status']='ok'
        except ResourceLimitError:
            trace['status']='resource_limit'
            raise
        self.stats.local_assignment_visits+=trace['local_assignment_visits']
        self.stats.inner_evaluations+=trace['inner_evaluations']
        return result

    def count(self) -> int:
        start=perf_counter()
        size=self.degree+1
        if size*size>self.max_work:
            raise ResourceLimitError('SWAP occurrence interpolation storage exceeds max_work')
        if size**3>self.max_work:
            raise ResourceLimitError('SWAP occurrence interpolation work exceeds max_work')
        self.stats.primes=primes_for_bound(1<<self.n)
        residues=[]
        for prime in self.stats.primes:
            jets=[self.evaluate_jet(t,q,prime) for t,q in zip(self.stats.nodes,self.stats.orders)]
            residues.append(interpolate_hermite_at_zero(jets,list(self.stats.nodes),list(self.stats.orders),prime))
        count=crt_reconstruct(residues,self.stats.primes)
        if count>1<<self.n:
            raise ArithmeticError('SWAP occurrence reconstruction exceeds model-count bound')
        self.stats.evaluation_seconds=perf_counter()-start
        return count


def swap_mobius_count(cnf: CNF,**options) -> int:
    return swap_mobius_count_with_stats(cnf,**options)[0]


def swap_mobius_count_with_stats(cnf: CNF,**options):
    counter=(SwapMobiusCounter(cnf,**options) if 'embedding' in options
             else OccurrenceSwapCounter(cnf,**options))
    return counter.count(),counter.stats


def swap_crossover_estimate(num_vars: int,*,d: int=2,L: int=2,b: int=1) -> dict:
    """Legacy multilinear-reference costs; not the default occurrence path."""
    if type(num_vars) is not int or num_vars < 0:
        raise ValueError('num_vars must be a nonnegative integer')
    system=build_swap_matrix(d,L,b)
    D=b*num_vars
    q=D//L
    return dict(representation='legacy_multilinear_reference',
                d=d,L=L,b=b,dimension=system.N,measurement_rows=system.s0,
                degree_upper_bound=D,jet_order_upper_bound=q,
                mobius_inner_terms_per_node_upper_bound=sum(comb(num_vars,j)*(1<<j) for j in range(min(q,num_vars)+1)),
                multilinear_terms_upper_bound=1<<num_vars,
                embedding_fits=num_vars<=system.N,
                executable_under_default_limit=num_vars<=6,
                crossover_established=False,
                status='combinatorial estimate only; no runtime crossover established')


def occurrence_cost_estimate(cnf: CNF,*,d: int=2,L: int=2,b: int=1) -> dict:
    """Count actual design metadata without enumerating Boolean assignments.

    The miniature's fixed parameters need not satisfy the asymptotic balancing
    or concentration assumptions. These are combinatorial sizes, not predicted
    seconds or evidence of a runtime crossover.
    """
    cnf.validate()
    counter=OccurrenceSwapCounter(cnf,d=d,L=L,b=b,max_variables=max(1,cnf.num_vars))
    q=counter.degree//L
    node_costs=[]
    for node in range(1,L+1):
        groups=[tuple(v for v,alpha in enumerate(counter.alpha) if alpha[node-1]==g)
                for g in range(d)]
        restricted=[tuple(row[v] for v in variables)
                    for variables in groups for row in counter.measurement_rows
                    if any(row[v] for v in variables)]
        norm=max((sum(map(abs,row)) for row in restricted),default=0)
        node_costs.append(dict(node=node,group_sizes=list(map(len,groups)),
                               local_assignment_rows=[1<<len(group) for group in groups],
                               group_split_measurement_rows=len(restricted),
                               measurement_l1_max=norm,
                               measurement_grid_tuples_upper_bound=(2*norm+1)**len(restricted)))
    return dict(representation='occurrence_coherent_clause_factors',d=d,L=L,b=b,
                variables=counter.n,clauses=counter.stats.clause_count,
                literal_occurrences=counter.stats.occurrence_count,
                dimension=counter.system.N,block_count=len(counter.blocks),
                raw_measurement_rows=counter.stats.raw_measurement_rows,
                measurement_rows=len(counter.measurement_rows),
                degree_upper_bound=counter.degree,jet_order_upper_bound=q,
                interpolation_matrix_entries_upper_bound=(counter.degree+1)**2,
                interpolation_work_upper_bound=(counter.degree+1)**3,
                mobius_inner_terms_per_node_upper_bound=sum(
                    comb(counter.n,j)*(1<<j) for j in range(min(q,counter.n)+1)),
                node_costs=node_costs,executable_under_default_limit=counter.n<=6,
                asymptotic_parameter_guarantees_applied=False,crossover_established=False,
                status='combinatorial estimate only; no runtime crossover established')
