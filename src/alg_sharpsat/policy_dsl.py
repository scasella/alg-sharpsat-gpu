"""Restricted symbolic policies with uniquely determined, neutral-weight auxiliaries."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Mapping, Sequence, TYPE_CHECKING
import numpy as np
import numbers
import re
from .cnf import CNF, Clause, Literal

if TYPE_CHECKING:
    from .stage5_wmc_guardrail import WMCResult


def parse_probability(value: object) -> Fraction:
    """Parse a bounded rational scalar before allocating exponent-sized integers."""
    if isinstance(value, Fraction):
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 16384:
            raise ValueError("probability exceeds rational size limit")
        result = value
    else:
        text = str(value)
        if len(text) > 4096:
            raise ValueError("probability exceeds text size limit")
        exponent = re.search(r"[eE]([+-]?\d+)$", text)
        if exponent and (
            len(exponent[1].lstrip("+-")) > 4 or abs(int(exponent[1])) > 4096
        ):
            raise ValueError("probability exponent exceeds size limit")
        try:
            result = Fraction(text)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError("invalid rational probability") from exc
    if not 0 <= result <= 1:
        raise ValueError("probabilities must lie in [0,1]")
    return result


@dataclass(frozen=True)
class VariableSpec:
    name: str
    description: str
    category: str = "general"
    default_prob: object = 0.5
    role: str | None = None

    def __post_init__(self):
        if self.role not in (None, "evidence", "action", "safeguard"):
            raise ValueError("variable role must be evidence, action, or safeguard")

    @property
    def effective_role(self) -> str:
        if self.role is not None:
            return self.role
        if self.category in {"context", "evidence", "patient_state"}:
            return "evidence"
        if self.category in {"control", "controls", "safeguard", "safeguards"}:
            return "safeguard"
        return "action"

    @property
    def can_clamp(self) -> bool:
        """Boundary eligibility; safeguards still require verified provenance."""
        return self.effective_role in {"evidence", "safeguard"}


def validate_clamps(clamped_vars, variable_names) -> dict[str, int]:
    """Validate explicit binary observations without coercing strings or floats."""
    if clamped_vars is None:
        return {}
    if not isinstance(clamped_vars, Mapping):
        raise ValueError("clamped_vars must be a mapping")
    if any(name not in variable_names for name in clamped_vars):
        raise ValueError("clamped_vars contains unknown variable names")
    if any(
        not isinstance(value, (numbers.Integral, np.bool_)) or value not in (0, 1)
        for value in clamped_vars.values()
    ):
        raise ValueError("clamped_vars values must be binary booleans or integers")
    return {name: int(value) for name, value in clamped_vars.items()}


def apply_clamps(probs, variable_names, clamped_vars=None):
    """Copy a probability mapping/vector/batch and replace observed priors."""
    clamps = validate_clamps(clamped_vars, variable_names)
    if isinstance(probs, Mapping):
        if set(probs) != set(variable_names):
            raise ValueError(
                "probability mapping must contain exactly the policy variables"
            )
        return {name: clamps.get(name, probs[name]) for name in variable_names}
    if hasattr(probs, "detach"):
        probs = probs.detach().cpu().numpy()
    values = np.array(probs, copy=True)
    if values.ndim not in (1, 2) or values.shape[-1] != len(variable_names):
        raise ValueError("probabilities must have shape (n,) or (K,n)")
    for name, value in clamps.items():
        values[..., variable_names.index(name)] = value
    return values


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    expr: str
    enabled: bool = True


def _not(a):
    return a[1] if a[0] == "not" else ("not", a)


def _join(op, args):
    args = tuple(args)
    return args[0] if len(args) == 1 else (op, *args)


class _Parser:
    def __init__(self, text, names, max_tokens, max_depth):
        if not isinstance(text, str):
            raise ValueError("expression must be text")
        self.tokens = re.findall(r"<->|->|[A-Za-z_][A-Za-z_0-9]*|[()!,&|~^]|\S", text)
        if len(self.tokens) > max_tokens:
            raise ValueError("expression exceeds max_tokens")
        self.i, self.names, self.max_depth = 0, names, max_depth
        self.max_tokens = max_tokens

    def take(self):
        if self.i == len(self.tokens):
            raise ValueError("unexpected end of expression")
        t = self.tokens[self.i]
        self.i += 1
        return t

    def parse(self, minimum=0, depth=0):
        if depth > self.max_depth:
            raise ValueError("expression exceeds max_depth")
        token = self.take()
        if token in ("not", "~", "!"):
            left = _not(self.parse(6, depth + 1))
        elif token == "(":
            left = self.parse(0, depth + 1)
            if self.take() != ")":
                raise ValueError("expected closing parenthesis")
        elif self.i < len(self.tokens) and self.tokens[self.i] == "(":
            self.i += 1
            args = []
            if self.i < len(self.tokens) and self.tokens[self.i] != ")":
                while True:
                    args.append(self.parse(0, depth + 1))
                    sep = self.take()
                    if sep == ")":
                        break
                    if sep != ",":
                        raise ValueError("expected comma")
            else:
                self.take()
            if token in ("implies", "iff", "xor") and len(args) == 2:
                left = (token, *args)
            elif token in ("require", "forbid"):
                conjunction = _join("and", args)
                left = conjunction if token == "require" else _not(conjunction)
            elif token in ("at_least_one", "at_most_one", "exactly_one"):
                if len(args) * (len(args) - 1) // 2 > self.max_tokens:
                    raise ValueError("cardinality helper expansion exceeds max_tokens")
                atleast = _join("or", args)
                atmost = _join(
                    "and",
                    (
                        (_join("or", (_not(a), _not(b))))
                        for a, b in combinations(args, 2)
                    ),
                )
                left = (
                    atleast
                    if token == "at_least_one"
                    else atmost
                    if token == "at_most_one"
                    else ("and", atleast, atmost)
                )
            else:
                raise ValueError("unknown helper or invalid helper arity")
        elif token in self.names:
            left = ("var", token)
        else:
            raise ValueError(f"unknown variable or token {token!r}")
        operators = {
            "<->": (1, "iff"),
            "iff": (1, "iff"),
            "->": (2, "implies"),
            "implies": (2, "implies"),
            "or": (3, "or"),
            "|": (3, "or"),
            "xor": (4, "xor"),
            "^": (4, "xor"),
            "and": (5, "and"),
            "&": (5, "and"),
        }
        while self.i < len(self.tokens) and self.tokens[self.i] in operators:
            priority, op = operators[self.tokens[self.i]]
            if priority < minimum:
                break
            self.i += 1
            right = self.parse(priority if op == "implies" else priority + 1, depth + 1)
            left = (op, left, right)
        return left


def _value(node, values):
    op, *args = node
    if op == "var":
        return bool(values[args[0]])
    vals = [_value(a, values) for a in args]
    if op == "not":
        return not vals[0]
    if op == "and":
        return all(vals)
    if op == "or":
        return any(vals)
    if op == "implies":
        return not vals[0] or vals[1]
    if op == "iff":
        return vals[0] == vals[1]
    return vals[0] != vals[1]


class PolicySpec:
    def __init__(
        self,
        variables: Sequence[VariableSpec],
        rules: Sequence[RuleSpec],
        *,
        max_tokens=10000,
        max_depth=128,
        max_variables=10000,
        max_clauses=100000,
        max_aux_variables=100000,
    ):
        self.variables, self.rules = tuple(variables), tuple(rules)
        self.variable_names = tuple(v.name for v in self.variables)
        if len(set(self.variable_names)) != len(self.variables) or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", n)
            or n in {"and", "or", "not", "implies", "iff", "xor"}
            for n in self.variable_names
        ):
            raise ValueError(
                "variable names must be unique identifiers and not operators"
            )
        if len(self.variables) > max_variables:
            raise ValueError("policy exceeds max_variables")
        if len({r.rule_id for r in self.rules}) != len(self.rules):
            raise ValueError("duplicate rule_id")
        for v in self.variables:
            parse_probability(v.default_prob)
        self._trees = []
        for r in self.rules:
            p = _Parser(r.expr, set(self.variable_names), max_tokens, max_depth)
            tree = p.parse()
            if p.i != len(p.tokens):
                raise ValueError("unexpected trailing token")
            pending = [(tree, 0)]
            while pending:
                node, level = pending.pop()
                if level > max_depth:
                    raise ValueError("expression exceeds max_depth")
                if node[0] != "var":
                    pending.extend((child, level + 1) for child in node[1:])
            self._trees.append(tree)
        self.max_clauses, self.max_aux_variables = max_clauses, max_aux_variables

    def check_violations(self, values: Mapping[str, bool | int]) -> list[RuleSpec]:
        if any(n not in values for n in self.variable_names):
            raise ValueError("assignment is missing variables")
        if any(
            not isinstance(values[n], (numbers.Integral, np.bool_))
            or values[n] not in (0, 1)
            for n in self.variable_names
        ):
            raise ValueError("assignment values must be binary booleans or integers")
        return [
            r
            for r, t in zip(self.rules, self._trees)
            if r.enabled and not _value(t, values)
        ]

    def to_cnf(self) -> CNF:
        names = {n: i + 1 for i, n in enumerate(self.variable_names)}
        clauses, cache = [], {}
        nextvar = len(names)

        def add(c):
            if len(clauses) >= self.max_clauses:
                raise ValueError("policy exceeds max_clauses")
            clauses.append(Clause(tuple(Literal(abs(x) - 1, x > 0) for x in c)))

        def lit(t):
            nonlocal nextvar
            op, *args = t
            if op == "var":
                return names[args[0]]
            if op == "not":
                return -lit(args[0])
            if t in cache:
                return cache[t]
            a = [lit(x) for x in args]
            nextvar += 1
            if nextvar - len(names) > self.max_aux_variables:
                raise ValueError("policy exceeds max_aux_variables")
            z = nextvar
            cache[t] = z
            if op == "and":
                for x in a:
                    add([-z, x])
                add([z, *(-x for x in a)])
            elif op == "or":
                for x in a:
                    add([z, -x])
                add([-z, *a])
            elif op == "implies":
                x, y = a
                add([-z, -x, y])
                add([z, x])
                add([z, -y])
            else:
                x, y = a
                for vx in (False, True):
                    for vy in (False, True):
                        result = (vx == vy) if op == "iff" else (vx != vy)
                        add([-x if vx else x, -y if vy else y, z if result else -z])
            return z

        def direct(t, negated=False):
            """Return small direct CNF, or None when a Tseitin seam is needed."""
            op, *args = t
            if op == "var":
                value = names[args[0]]
                return [[-value if negated else value]]
            if op == "not":
                return direct(args[0], not negated)
            if op == "implies":
                return direct(("or", _not(args[0]), args[1]), negated)
            if op not in ("and", "or"):
                return None
            join = ("or" if op == "and" else "and") if negated else op
            pieces = [direct(a, negated) for a in args]
            if any(p is None for p in pieces):
                return None
            if join == "and":
                if sum(map(len, pieces)) > 256:
                    return None
                return [c for p in pieces for c in p]
            result = [[]]
            for piece in pieces:
                if len(result) * len(piece) > 256:
                    return None
                result = [a + b for a in result for b in piece]
            return result

        def enforce(t):
            simple = direct(t)
            if simple is not None:
                for clause in simple:
                    add(clause)
            elif t[0] == "and":
                for a in t[1:]:
                    enforce(a)
            elif t[0] in ("iff", "xor"):
                a, b = lit(t[1]), lit(t[2])
                if t[0] == "iff":
                    add([-a, b])
                    add([a, -b])
                else:
                    add([a, b])
                    add([-a, -b])
            else:
                add([lit(t)])

        for r, t in zip(self.rules, self._trees):
            if r.enabled:
                enforce(t)
        return CNF(nextvar, clauses)

    compile = to_cnf

    def compile_guardrail(self, **opts) -> "CompiledPolicy":
        return CompiledPolicy(self, **opts)


class CompiledPolicy:
    def __init__(self, policy, **opts):
        from .stage5_wmc_guardrail import compile_guardrail

        self.policy = policy
        self.cnf = policy.to_cnf()
        self.compiled = compile_guardrail(self.cnf, **opts)
        self.variable_names = policy.variable_names

    @property
    def model_count(self) -> int:
        result = self.compiled.evaluate(
            weights=np.ones((self.cnf.num_vars, 2), dtype=object), mode="exact_rational"
        )
        return int(result.partition_function)

    def _weights(self, probs, *, floating: bool = False, clamped_vars=None):
        if probs is None:
            probs = [v.default_prob for v in self.policy.variables]
        if clamped_vars is not None:
            # Validate original priors before overriding evidence.
            self._weights(probs, floating=floating)
            probs = apply_clamps(probs, self.variable_names, clamped_vars)
        if isinstance(probs, Mapping):
            if set(probs) != set(self.variable_names):
                raise ValueError(
                    "probability mapping must contain exactly the policy variables"
                )
            probs = [probs[n] for n in self.variable_names]
        if hasattr(probs, "detach"):
            probs = probs.detach().cpu().numpy()
        if floating:
            try:
                array = np.asarray(probs)
            except (ValueError, TypeError):
                array = None
            if array is not None and array.dtype.kind in "buif":
                n = len(self.variable_names)
                if array.ndim not in (1, 2) or array.shape[-1] != n:
                    raise ValueError("probabilities must have shape (n,) or (K,n)")
                if array.ndim == 2 and array.shape[0] == 0:
                    raise ValueError(
                        "batch must contain at least one probability vector"
                    )
                array = array.astype(np.float64, copy=False)
                if (
                    not np.isfinite(array).all()
                    or (array < 0).any()
                    or (array > 1).any()
                ):
                    raise ValueError("probabilities must be finite and lie in [0,1]")
                weights = np.ones(
                    (*array.shape[:-1], self.cnf.num_vars, 2), dtype=np.float64
                )
                weights[..., :n, 0] = 1 - array
                weights[..., :n, 1] = array
                return weights
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        probs = list(probs)
        if probs and isinstance(probs[0], (list, tuple, dict)):
            return [self._weights(p) for p in probs]
        if len(probs) != len(self.variable_names):
            raise ValueError("wrong probability vector length")
        p = [parse_probability(x) for x in probs]
        if any(x < 0 or x > 1 for x in p):
            raise ValueError("probabilities must lie in [0,1]")
        weights = [(1 - x, x) for x in p] + [(Fraction(1), Fraction(1))] * (
            self.cnf.num_vars - len(p)
        )
        return weights if weights else np.empty((0, 2), dtype=object)

    def evaluate(
        self, probs=None, mode: str = "float64", *, clamped_vars=None, **kwargs
    ) -> "WMCResult":
        """Evaluate a vector or floating batch; use Fraction/strings for rationals.

        Nested sequences always denote batches, never numerator/denominator pairs.
        Exact arithmetic accepts a single probability vector. ``clamped_vars``
        accepts any named library variable; extraction and HTTP/UI boundaries
        restrict eligibility to evidence and verified safeguards.
        """
        result = self.compiled.evaluate(
            weights=self._weights(
                probs, floating=mode != "exact_rational", clamped_vars=clamped_vars
            ),
            mode=mode,
            **kwargs,
        )
        result.posterior_marginals = (
            result.posterior_marginals[..., : len(self.variable_names)]
            if hasattr(result.posterior_marginals, "shape")
            else result.posterior_marginals[: len(self.variable_names)]
        )
        result.acceptance_prob = result.partition_function
        if mode == "exact_rational":
            result.kl_divergence_nats = (
                -result.log_partition if result.partition_function else float("nan")
            )
        else:
            import torch

            result.kl_divergence_nats = torch.where(
                ~torch.isneginf(result.log_partition),
                -result.log_partition,
                torch.full_like(result.log_partition, float("nan")),
            )
        return result

    def map_inference(
        self, probs=None, mode: str = "exact_rational", *, clamped_vars=None
    ) -> tuple[tuple[int, ...] | None, Fraction | float]:
        assignment, weight = self.compiled.map_inference(
            weights=self._weights(
                probs, floating=mode != "exact_rational", clamped_vars=clamped_vars
            ),
            mode=mode,
        )
        return (
            None if assignment is None else assignment[: len(self.variable_names)]
        ), weight

    def sample_conditional(
        self,
        probs=None,
        num_samples: int = 1,
        seed: int = 0,
        mode: str = "exact_rational",
        *,
        clamped_vars=None,
    ) -> np.ndarray:
        return self.compiled.sample_conditional(
            weights=self._weights(
                probs, floating=mode != "exact_rational", clamped_vars=clamped_vars
            ),
            num_samples=num_samples,
            seed=seed,
            mode=mode,
        )[:, : len(self.variable_names)]


@dataclass(frozen=True)
class PolicyDiff:
    old_allowed_count: int
    new_allowed_count: int
    intersection_count: int
    newly_exposed_count: int
    revoked_count: int
    old_allowed_prob: Fraction | None = None
    new_allowed_prob: Fraction | None = None
    intersection_prob: Fraction | None = None
    newly_exposed_prob: Fraction | None = None
    revoked_prob: Fraction | None = None


def diff_policies(old: PolicySpec, new: PolicySpec, probs=None) -> PolicyDiff:
    if set(old.variable_names) != set(new.variable_names):
        raise ValueError("policies must use the same variable set")
    combined = PolicySpec(
        old.variables,
        [
            RuleSpec("old_" + str(i), r.description, r.expr, r.enabled)
            for i, r in enumerate(old.rules)
        ]
        + [
            RuleSpec("new_" + str(i), r.description, r.expr, r.enabled)
            for i, r in enumerate(new.rules)
        ],
    )
    engines = [p.compile_guardrail() for p in (old, new, combined)]
    a, b, c = (engine.model_count for engine in engines)
    if probs is None:
        return PolicyDiff(a, b, c, b - c, a - c)
    if isinstance(probs, Mapping):
        if set(probs) != set(old.variable_names):
            raise ValueError(
                "probability mapping must contain exactly the policy variables"
            )
        aligned = probs
    else:
        probabilities = list(probs)
        if len(probabilities) != len(old.variable_names):
            raise ValueError("wrong probability vector length")
        aligned = dict(zip(old.variable_names, probabilities))
    x, y, z = (
        engine.evaluate(aligned, mode="exact_rational").partition_function
        for engine in engines
    )
    return PolicyDiff(a, b, c, b - c, a - c, x, y, z, y - z, x - z)
