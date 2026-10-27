"""Validated CNF data and exact reference model counters.

Variable identifiers are zero based; DIMACS uses signed one-based IDs. Empty
clauses are false; an empty formula is true, including all its explicitly
declared, unconstrained variables.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def _integer(value: object, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, order=True)
class Literal:
    var: int
    pos: bool

    def __post_init__(self) -> None:
        _integer(self.var, "var")
        if type(self.pos) is not bool:
            raise ValueError("pos must be a bool")

    def __neg__(self) -> Literal:
        return Literal(self.var, not self.pos)

    @property
    def signed(self) -> int:
        return self.var + 1 if self.pos else -(self.var + 1)


@dataclass(frozen=True)
class Clause:
    literals: tuple[Literal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.literals, tuple) or any(
            not isinstance(lit, Literal) for lit in self.literals
        ):
            raise ValueError("literals must be a tuple of Literal objects")


@dataclass
class CNF:
    num_vars: int
    clauses: list[Clause]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _integer(self.num_vars, "num_vars")
        if not isinstance(self.clauses, list) or any(
            not isinstance(clause, Clause) for clause in self.clauses
        ):
            raise ValueError("clauses must be a list of Clause objects")
        if any(lit.var >= self.num_vars for c in self.clauses for lit in c.literals):
            raise ValueError("literal variable exceeds num_vars")

    @classmethod
    def from_dimacs(cls, text: str) -> CNF:
        return parse_dimacs(text)

    @classmethod
    def parse_dimacs(cls, text: str) -> CNF:
        return parse_dimacs(text)

    def to_dimacs(self) -> str:
        return to_dimacs(self)


def parse_dimacs(text: str) -> CNF:
    """Parse DIMACS, accepting clauses spanning lines and multiple per line.

    A single accurate ``p cnf`` header is required. Comments begin a line with
    ``c``; malformed tokens, missing terminators and out-of-range IDs fail.
    """
    if not isinstance(text, str):
        raise ValueError("DIMACS input must be text")
    header: tuple[int, int] | None = None
    clauses: list[Clause] = []
    pending: list[Literal] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        tokens = line.split()
        if tokens[0] == "p":
            if header is not None or len(tokens) != 4 or tokens[1] != "cnf":
                raise ValueError(f"invalid DIMACS header on line {line_number}")
            try:
                n, m = int(tokens[2]), int(tokens[3])
            except ValueError as exc:
                raise ValueError("noninteger DIMACS header") from exc
            _integer(n, "DIMACS variable count")
            _integer(m, "DIMACS clause count")
            header = (n, m)
            continue
        if header is None:
            raise ValueError("DIMACS clauses must follow a p cnf header")
        for token in tokens:
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"invalid DIMACS literal {token!r}") from exc
            if value == 0:
                clauses.append(Clause(tuple(pending)))
                pending = []
            else:
                if abs(value) > header[0]:
                    raise ValueError("DIMACS literal exceeds declared variable count")
                pending.append(Literal(abs(value) - 1, value > 0))
    if header is None:
        raise ValueError("missing DIMACS p cnf header")
    if pending:
        raise ValueError("unterminated DIMACS clause")
    if len(clauses) != header[1]:
        raise ValueError("DIMACS clause count does not match header")
    return CNF(header[0], clauses)


def to_dimacs(cnf: CNF) -> str:
    cnf.validate()
    lines = [f"p cnf {cnf.num_vars} {len(cnf.clauses)}"]
    lines.extend(" ".join([*(str(l.signed) for l in c.literals), "0"]) for c in cnf.clauses)
    return "\n".join(lines) + "\n"


def brute_force_count(cnf: CNF, chunk_size: int = 65536) -> int:
    """Count with vectorized NumPy batches and bounded assignment storage.

    Unused variables are factored out exactly. The returned Python integer does
    not overflow, including for formulas with many unconstrained variables.
    """
    cnf.validate()
    _integer(chunk_size, "chunk_size", 1)
    if any(not c.literals for c in cnf.clauses):
        return 0
    active = sorted({l.var for c in cnf.clauses for l in c.literals})
    positions = {var: i for i, var in enumerate(active)}
    total = 1 << len(active)
    count = 0
    for start in range(0, total, chunk_size):
        size = min(chunk_size, total - start)
        # uint64 wraps the low bits correctly; larger positions use Python ints.
        ids = np.arange(size, dtype=np.uint64) + np.uint64(start % (1 << 64))
        values = [
            ((ids >> np.uint64(i)) & 1).astype(bool)
            if i < 64 else np.fromiter(
                (((start + j) >> i) & 1 for j in range(size)), dtype=bool, count=size
            )
            for i in range(len(active))
        ]
        satisfies = np.ones(size, dtype=bool)
        for clause in cnf.clauses:
            clause_value = np.zeros(size, dtype=bool)
            for lit in clause.literals:
                v = values[positions[lit.var]]
                clause_value |= v if lit.pos else ~v
            satisfies &= clause_value
            if not satisfies.any():
                break
        count += int(np.count_nonzero(satisfies))
    return count << (cnf.num_vars - len(active))


def dpll_count(cnf: CNF) -> int:
    """Exact DPLL counter with unit propagation and unused-variable accounting."""
    cnf.validate()

    def canonical(clauses):
        return tuple(sorted(set(tuple(sorted(set(c))) for c in clauses)))

    clauses = canonical(
        tuple(l.signed for l in c.literals)
        for c in cnf.clauses
        if not any(-l.signed in {x.signed for x in c.literals} for l in c.literals)
    )

    def assign(current, literal):
        return canonical(
            tuple(v for v in clause if v != -literal)
            for clause in current if literal not in clause
        )

    @lru_cache(maxsize=None)
    def solve(current, remaining):
        # Propagate a full unit chain iteratively, avoiding Python recursion
        # limits on formulas with thousands of unit clauses.
        while current:
            if not current[0]:
                return 0
            unit = next((c[0] for c in current if len(c) == 1), None)
            if unit is None:
                break
            current = assign(current, unit)
            remaining -= 1
        if not current:
            return 1 << remaining
        frequency = Counter(abs(lit) for clause in current for lit in clause)
        variable = min(frequency, key=lambda v: (-frequency[v], v))
        return solve(assign(current, variable), remaining - 1) + solve(
            assign(current, -variable), remaining - 1
        )

    return solve(clauses, cnf.num_vars)
