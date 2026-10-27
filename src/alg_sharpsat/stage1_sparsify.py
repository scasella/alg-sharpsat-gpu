"""Deterministic, model-count preserving sparsification and partitioning.

Width reduction gives every original assignment a unique auxiliary extension.
Degree splitting caps *original-clause incidence* per copy; equality-cycle
clauses add at most two further incidences, so total incidence <= max_degree+2.
Repeated literals in one clause count as one incidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .cnf import CNF, Clause, Literal, _integer


def reduce_clause_width_deterministic(cnf: CNF, max_width: int = 3) -> CNF:
    """Replace prefixes of wide clauses with uniquely determined OR gates.

    For each gate y = (a OR b), emit (!a OR y), (!b OR y),
    (a OR b OR !y). This retains the model count, unlike an existential-only
    clause splitting gadget, even with repeated or opposite literals.
    """
    cnf.validate()
    _integer(max_width, "max_width", 3)
    clauses: list[Clause] = []
    next_var = cnf.num_vars
    for clause in cnf.clauses:
        literals = list(clause.literals)
        while len(literals) > max_width:
            a, b = literals[:2]
            y = Literal(next_var, True)
            next_var += 1
            clauses.extend((Clause((-a, y)), Clause((-b, y)), Clause((a, b, -y))))
            literals = [y, *literals[2:]]
        clauses.append(Clause(tuple(literals)))
    return CNF(next_var, clauses)


reduce_clause_width = reduce_clause_width_deterministic


def split_high_degree_variables(cnf: CNF, max_degree: int = 4) -> CNF:
    """Split a variable used in m clauses into ceil(m/max_degree) copies.

    The first copy reuses the original ID. Each copy appears in at most
    max_degree input clauses. A directed implication cycle forces all copies
    equal and contributes two distinct clauses per copy (including the
    two-copy case). Hence total clause incidence is at most max_degree+2.
    Duplicated literal occurrences within an input clause stay on one copy.
    """
    cnf.validate()
    _integer(max_degree, "max_degree", 1)
    incidences: dict[int, list[int]] = {}
    for ci, clause in enumerate(cnf.clauses):
        for variable in sorted({l.var for l in clause.literals}):
            incidences.setdefault(variable, []).append(ci)
    replacements: dict[tuple[int, int], int] = {}
    equality_clauses: list[Clause] = []
    next_var = cnf.num_vars
    for variable, clause_indices in sorted(incidences.items()):
        copies_needed = (len(clause_indices) + max_degree - 1) // max_degree
        if copies_needed <= 1:
            continue
        copies = [variable]
        for _ in range(copies_needed - 1):
            copies.append(next_var)
            next_var += 1
        for i, ci in enumerate(clause_indices):
            replacements[(ci, variable)] = copies[i // max_degree]
        for i, copy in enumerate(copies):
            successor = copies[(i + 1) % len(copies)]
            equality_clauses.append(Clause((Literal(copy, False), Literal(successor, True))))
    clauses = [
        Clause(tuple(Literal(replacements.get((ci, l.var), l.var), l.pos) for l in c.literals))
        for ci, c in enumerate(cnf.clauses)
    ]
    return CNF(next_var, clauses + equality_clauses)


@dataclass(frozen=True)
class BoundaryLink:
    master_var: int
    master_group: int
    proxy_var: int
    proxy_group: int

    def __post_init__(self):
        _integer(self.master_var, "master_var")
        _integer(self.proxy_var, "proxy_var")
        _integer(self.master_group, "master_group")
        _integer(self.proxy_group, "proxy_group")
        if self.master_group == self.proxy_group or self.master_var == self.proxy_var:
            raise ValueError("a boundary link must connect distinct variables and groups")


@dataclass(frozen=True)
class MeasurementBlock:
    link_indices: tuple[int, ...]
    master_group: int

    def __post_init__(self):
        _integer(self.master_group, "master_group")
        if not isinstance(self.link_indices, tuple) or not self.link_indices:
            raise ValueError("measurement blocks require a nonempty tuple of links")
        for i in self.link_indices:
            _integer(i, "link index")
        if len(set(self.link_indices)) != len(self.link_indices):
            raise ValueError("duplicate link index in measurement block")


@dataclass(frozen=True)
class PartitionedCNF:
    num_vars: int
    group_vars: tuple[tuple[int, ...], ...]
    group_clauses: tuple[tuple[Clause, ...], ...]
    boundary_links: tuple[BoundaryLink, ...]
    measurement_blocks: tuple[MeasurementBlock, ...]
    original_num_vars: int

    def __post_init__(self):
        _integer(self.num_vars, "num_vars")
        _integer(self.original_num_vars, "original_num_vars")
        if self.original_num_vars > self.num_vars:
            raise ValueError("original_num_vars exceeds num_vars")
        if not isinstance(self.group_vars, tuple) or not self.group_vars:
            raise ValueError("group_vars must be a nonempty tuple")
        if not isinstance(self.group_clauses, tuple) or len(self.group_vars) != len(self.group_clauses):
            raise ValueError("group_vars and group_clauses must have equal length")
        owner = {}
        for gi, variables in enumerate(self.group_vars):
            if not isinstance(variables, tuple):
                raise ValueError("group variable lists must be tuples")
            for var in variables:
                _integer(var, "group variable")
                if var in owner or var >= self.num_vars:
                    raise ValueError("group variables must be disjoint and in range")
                owner[var] = gi
            if not isinstance(self.group_clauses[gi], tuple):
                raise ValueError("group clause lists must be tuples")
            for clause in self.group_clauses[gi]:
                if not isinstance(clause, Clause) or any(l.var not in variables for l in clause.literals):
                    raise ValueError("every clause must use only variables owned by its group")
        if len(owner) != self.num_vars:
            raise ValueError("every declared variable must belong to exactly one group")
        if not isinstance(self.boundary_links, tuple) or not isinstance(self.measurement_blocks, tuple):
            raise ValueError("links and blocks must be tuples")
        proxies = set()
        pairs = set()
        for link in self.boundary_links:
            if not isinstance(link, BoundaryLink):
                raise ValueError("invalid boundary link")
            if owner.get(link.master_var) != link.master_group or owner.get(link.proxy_var) != link.proxy_group:
                raise ValueError("boundary link ownership mismatch")
            if link.master_var >= self.original_num_vars or link.proxy_var < self.original_num_vars:
                raise ValueError("boundary link must connect an input variable to a new proxy")
            pair = (link.master_var, link.proxy_group)
            if link.proxy_var in proxies or pair in pairs:
                raise ValueError("proxies must be unique and reused per master/target group")
            proxies.add(link.proxy_var)
            pairs.add(pair)
        if proxies != set(range(self.original_num_vars, self.num_vars)):
            raise ValueError("each new variable must be a boundary proxy")
        covered = []
        for block in self.measurement_blocks:
            if not isinstance(block, MeasurementBlock):
                raise ValueError("invalid measurement block")
            for index in block.link_indices:
                if index >= len(self.boundary_links) or self.boundary_links[index].master_group != block.master_group:
                    raise ValueError("measurement block must contain links with one master group")
                covered.append(index)
        if sorted(covered) != list(range(len(self.boundary_links))):
            raise ValueError("each boundary link must occur in exactly one measurement block")


def _spectral_groups(cnf: CNF, num_groups: int) -> tuple[tuple[int, ...], ...]:
    """Balanced recursive spectral bisection with deterministic disconnected fallback."""
    n = cnf.num_vars
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for clause in cnf.clauses:
        variables = sorted({l.var for l in clause.literals})
        for i, a in enumerate(variables):
            for b in variables[i + 1:]:
                graph.add_edge(a, b, weight=graph.get_edge_data(a, b, {}).get("weight", 0) + 1)
    adjacency = nx.to_numpy_array(graph, nodelist=list(range(n)), dtype=float)
    quotas = [n // num_groups + (i < n % num_groups) for i in range(num_groups)]

    def order(nodes):
        if len(nodes) <= 1:
            return nodes
        # Explicit components avoid arbitrary bases for a degenerate nullspace.
        remaining = set(nodes)
        components = []
        while remaining:
            stack = [min(remaining)]
            remaining.remove(stack[0])
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                neighbors = sorted(v for v in remaining if adjacency[node, v] > 0)
                remaining.difference_update(neighbors)
                stack.extend(reversed(neighbors))
            components.append(sorted(component))
        if len(components) > 1:
            return [v for component in components for v in component]
        index = np.asarray(nodes, dtype=int)
        subgraph = adjacency[np.ix_(index, index)]
        laplacian = np.diag(subgraph.sum(axis=1)) - subgraph
        _, vectors = np.linalg.eigh(laplacian)
        fiedler = vectors[:, 1]
        nonzero = np.flatnonzero(np.abs(fiedler) > 1e-12)
        if nonzero.size and fiedler[nonzero[0]] < 0:
            fiedler = -fiedler
        return [nodes[i] for i in sorted(range(len(nodes)), key=lambda i: (round(float(fiedler[i]), 12), nodes[i]))]

    def bisect(nodes, sizes):
        if len(sizes) == 1:
            return [tuple(sorted(nodes))]
        midpoint = len(sizes) // 2
        left_count = sum(sizes[:midpoint])
        ordered = order(nodes)
        return bisect(ordered[:left_count], sizes[:midpoint]) + bisect(ordered[left_count:], sizes[midpoint:])

    return tuple(bisect(list(range(n)), quotas))


def partition_and_decouple_boundary(cnf: CNF, num_groups: int = 2, block_size: int = 4) -> PartitionedCNF:
    """Make local formulas, sharing one proxy per (master, destination group).

    Every declared input variable is retained. Each clause is owned by the
    group containing the most distinct variables, with ties broken by group ID.
    Empty clauses belong to group zero.
    Equality constraints are represented by boundary links, not local clauses.
    Blocks have one master group but may have several destinations or repeated
    masters; their population count therefore counts links with multiplicity.
    """
    cnf.validate()
    _integer(num_groups, "num_groups", 1)
    _integer(block_size, "block_size", 1)
    groups = [list(v) for v in _spectral_groups(cnf, num_groups)]
    owners = {v: i for i, variables in enumerate(groups) for v in variables}
    clauses: list[list[Clause]] = [[] for _ in groups]
    proxies: dict[tuple[int, int], int] = {}
    links = []
    next_var = cnf.num_vars
    for clause in cnf.clauses:
        votes = Counter(owners[v] for v in {l.var for l in clause.literals})
        group = min(votes, key=lambda gi: (-votes[gi], gi)) if votes else 0
        local_literals = []
        for literal in clause.literals:
            master = literal.var
            if owners[master] == group:
                local_literals.append(literal)
                continue
            key = (master, group)
            if key not in proxies:
                proxies[key] = next_var
                groups[group].append(next_var)
                links.append(BoundaryLink(master, owners[master], next_var, group))
                next_var += 1
            local_literals.append(Literal(proxies[key], literal.pos))
        clauses[group].append(Clause(tuple(local_literals)))
    blocks = []
    for group in range(num_groups):
        indices = [i for i, link in enumerate(links) if link.master_group == group]
        blocks.extend(MeasurementBlock(tuple(indices[i:i + block_size]), group) for i in range(0, len(indices), block_size))
    return PartitionedCNF(next_var, tuple(tuple(v) for v in groups), tuple(tuple(c) for c in clauses), tuple(links), tuple(blocks), cnf.num_vars)
