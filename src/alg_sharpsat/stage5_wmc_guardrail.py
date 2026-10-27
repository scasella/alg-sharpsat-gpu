"""Compiled sparse weighted counting. Undefined zero-mass posteriors are NaN."""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from decimal import Decimal
from math import lcm, log
from time import perf_counter
from copy import deepcopy
import numpy as np
import torch
from .cnf import CNF
from .stage3_sparse_contract import SparseCounter
from .stage3_gpu_engine import ResourceLimitError, UnsupportedDeviceError
from .stage1_sparsify import (
    partition_and_decouple_boundary,
    reduce_clause_width_deterministic,
    split_high_degree_variables,
)


@dataclass
class WMCResult:
    partition_function: object
    posterior_marginals: object
    log_partition: object
    acceptance_prob: object
    kl_divergence_nats: object

    def __iter__(self):
        yield self.partition_function
        yield self.posterior_marginals


@dataclass
class GuardrailResult(WMCResult):
    """Weighted posterior projection and its most likely accepted assignment."""

    map_assignment: object
    map_prob: object

    @property
    def map_weight(self):
        """Alias for the unnormalized MAP assignment weight."""
        return self.map_prob


@dataclass
class PlanNode:
    scope: int
    keys: np.ndarray
    left: int = -1
    right: int = -1
    pair_left: object = None
    pair_right: object = None
    out_idx: object = None
    original_vars: tuple = ()
    assignments: object = None
    key_idx: object = None

    @property
    def key_count(self):
        return len(self.keys)


class _Compiler(SparseCounter):
    def _finish(self, group, frontier):
        layout = self._layouts[group]
        keys = self._signature_keys(frontier, layout).astype(np.int64)
        unique, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
        owned = tuple(
            v
            for v in layout["order"]
            if v < self.actual_original_num_vars and v not in self.alias
        )
        self._budget(len(frontier), max(32, len(owned) + 16))
        assignments = np.array(
            [((frontier >> layout["positions"][v]) & 1) for v in owned], dtype=bool
        ).T
        if not owned:
            assignments = np.zeros((len(frontier), 0), dtype=bool)
        self._budget(len(frontier), max(32, len(owned) + 16))
        self.leaves.append(
            PlanNode(
                layout["scope"],
                unique,
                original_vars=owned,
                assignments=assignments,
                key_idx=inv,
            )
        )
        return super()._finish(group, frontier)


def _fraction(x):
    if isinstance(x, tuple) and len(x) == 2:
        return Fraction(*x)
    if isinstance(x, (Fraction, Decimal)):
        return Fraction(x)
    if isinstance(x, np.generic):
        x = x.item()
    return Fraction(str(x))


class CompiledGuardrail:
    """Immutable CNF support and cached sparse upward/downward contraction plan.

    Structure is compiled on CPU. Queries attach weights only to original
    variables; auxiliaries/proxies are neutral. ``raw_sparse_stats`` preserves
    construction counts; ``stats.retained_assignment_counts`` describes the
    support after backward pruning. Device caches are filled on first use.
    """

    def __init__(
        self,
        cnf: CNF,
        *,
        num_groups: int = 2,
        block_size: int = 4,
        max_degree: int | None = None,
        max_width: int | None = None,
        device: str = "auto",
        **options,
    ):
        started = perf_counter()
        self.original_num_vars = cnf.num_vars
        self.device = device
        self.max_memory_bytes = options.get("max_memory_bytes", 512 * 1024 * 1024)
        self.max_work = options.get("max_work", 2_000_000_000)
        c = _Compiler.__new__(_Compiler)
        c.leaves = []
        c.actual_original_num_vars = cnf.num_vars
        transformed = (
            reduce_clause_width_deterministic(cnf, max_width)
            if max_width is not None
            else cnf
        )
        if max_degree is not None:
            transformed = split_high_degree_variables(transformed, max_degree)
        partitioned = partition_and_decouple_boundary(
            transformed, num_groups, block_size
        )
        _Compiler.__init__(c, partitioned, "cpu", **options)
        self.stats = c.stats
        self.leaves = c.leaves
        self.nodes = list(self.leaves)
        owned = {v for leaf in self.leaves for v in leaf.original_vars}
        self.free_vars = tuple(v for v in range(cnf.num_vars) if v not in owned)
        active = list(range(len(self.nodes)))
        work = 0
        while len(active) > 1:
            a, b = min(
                ((a, b) for ii, a in enumerate(active) for b in active[ii + 1 :]),
                key=lambda ab: (
                    -(self.nodes[ab[0]].scope & self.nodes[ab[1]].scope).bit_count(),
                    self.nodes[ab[0]].key_count * self.nodes[ab[1]].key_count,
                    ab,
                ),
            )
            left, right = self.nodes[a], self.nodes[b]
            shared = left.scope & right.scope
            rp = right.keys & shared
            order = np.argsort(rp)
            rp = rp[order]
            lo = np.searchsorted(rp, left.keys & shared)
            hi = np.searchsorted(rp, left.keys & shared, side="right")
            lengths = hi - lo
            count = int(lengths.sum())
            work += count
            if count > c.max_join_pairs:
                raise ResourceLimitError("compiled sparse join exceeds max_join_pairs")
            c._budget(count, 96)
            if work > self.max_work:
                raise ResourceLimitError("compiled contraction exceeds max_work")
            li = np.repeat(np.arange(left.key_count), lengths)
            ri = order[
                lo[li]
                + np.arange(count)
                - np.repeat(np.cumsum(lengths) - lengths, lengths)
            ]
            remaining = [i for i in active if i not in (a, b)]
            needed = 0
            for i in remaining:
                needed |= self.nodes[i].scope
            keep = (left.scope | right.scope) & needed
            keys, oi = np.unique(
                (left.keys[li] | right.keys[ri]) & keep, return_inverse=True
            )
            self.nodes.append(PlanNode(keep, keys, a, b, li, ri, oi))
            active = remaining + [len(self.nodes) - 1]
            if (
                sum(
                    n.keys.nbytes
                    + sum(
                        getattr(n, k).nbytes
                        for k in (
                            "pair_left",
                            "pair_right",
                            "out_idx",
                            "assignments",
                            "key_idx",
                        )
                        if getattr(n, k) is not None
                    )
                    for n in self.nodes
                )
                > self.max_memory_bytes
            ):
                raise ResourceLimitError("compiled plan exceeds max_memory_bytes")
        if not active:
            self.nodes.append(
                PlanNode(
                    0,
                    np.array([0]),
                    original_vars=(),
                    assignments=np.zeros((1, 0), bool),
                    key_idx=np.array([0]),
                )
            )
            self.leaves = [self.nodes[-1]]
            active = [0]
        self.root = active[0]
        self.stats.join_pair_counts = [
            len(n.pair_left) for n in self.nodes if n.left >= 0
        ]
        self.stats.intermediate_rows = [n.key_count for n in self.nodes if n.left >= 0]
        self.stats.peak_intermediate_rows = max(self.stats.intermediate_rows, default=0)
        self.stats.selected_engine = "wmc_guardrail"
        self.raw_sparse_stats = deepcopy(self.stats)
        self._prune_unreachable()
        self._tensor_cache = {}
        self._probe_cache = {}
        self.compile_seconds = perf_counter() - started
        self.stats.preprocessing_seconds = (
            self.compile_seconds - self.stats.build_seconds
        )
        self.stats.compile_seconds = self.compile_seconds
        self.stats.retained_assignment_counts = [len(n.key_idx) for n in self.leaves]
        self.stats.retained_join_pair_counts = [
            len(n.pair_left) for n in self.nodes if n.left >= 0
        ]

    def _prune_unreachable(self):
        """Remove signatures and local rows which cannot reach the root support."""
        allowed = [np.zeros(n.key_count, dtype=bool) for n in self.nodes]
        allowed[self.root][:] = True
        pair_masks = {}
        for i in range(len(self.nodes) - 1, -1, -1):
            node = self.nodes[i]
            if node.left >= 0:
                mask = allowed[i][node.out_idx]
                pair_masks[i] = mask
                allowed[node.left][node.pair_left[mask]] = True
                allowed[node.right][node.pair_right[mask]] = True
        remaps = []
        for node, mask in zip(self.nodes, allowed):
            mapping = np.full(node.key_count, -1, dtype=np.int64)
            mapping[mask] = np.arange(int(mask.sum()))
            remaps.append(mapping)
        for i, node in enumerate(self.nodes):
            node.keys = node.keys[allowed[i]]
            if node.left < 0:
                rows = allowed[i][node.key_idx]
                node.assignments = node.assignments[rows]
                node.key_idx = remaps[i][node.key_idx[rows]]
            else:
                mask = pair_masks[i]
                node.pair_left = remaps[node.left][node.pair_left[mask]]
                node.pair_right = remaps[node.right][node.pair_right[mask]]
                node.out_idx = remaps[i][node.out_idx[mask]]

    def _cached_tensors(self, target, dtype):
        """Materialize immutable plan indices once per device and arithmetic type."""
        key = (str(target), dtype)
        if key not in self._tensor_cache:
            result = []
            for node in self.nodes:
                arrays = {}
                for name in ("key_idx", "pair_left", "pair_right", "out_idx"):
                    value = getattr(node, name)
                    if value is not None:
                        arrays[name] = torch.as_tensor(
                            value, dtype=torch.long, device=target
                        )
                if node.left < 0:
                    arrays["row_bits"] = tuple(
                        torch.as_tensor(
                            node.assignments[:, j].astype(np.int64), device=target
                        )
                        for j in range(len(node.original_vars))
                    )
                    arrays["owned_bits"] = torch.as_tensor(
                        node.assignments, dtype=dtype, device=target
                    )
                    arrays["original_vars"] = torch.tensor(
                        node.original_vars, dtype=torch.long, device=target
                    )
                result.append(arrays)
            self._tensor_cache[key] = result
        return self._tensor_cache[key]

    def _inputs(self, inputs=None, weights=None, input_kind="probabilities"):
        if weights is not None:
            if inputs is not None:
                raise ValueError("supply probabilities or weights, not both")
            inputs = weights
            input_kind = "weights"
        if inputs is None:
            inputs = [Fraction(1, 2)] * self.original_num_vars
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.detach().cpu().numpy()

        def normalize(value, depth=0):
            if (
                depth >= (2 if input_kind == "weights" else 1)
                and isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(x, (int, np.integer)) for x in value)
            ):
                return Fraction(int(value[0]), int(value[1]))
            if isinstance(value, (list, tuple)):
                return [normalize(x, depth + 1) for x in value]
            return value

        arr = np.asarray(normalize(inputs), dtype=object)
        n = self.original_num_vars
        if input_kind == "weights":
            if arr.shape == (n, 2):
                return arr[None], False
            if arr.ndim == 3 and arr.shape[1:] == (n, 2):
                return arr, True
            raise ValueError("weights must have shape (n,2) or (K,n,2)")
        if input_kind != "probabilities":
            raise ValueError("input_kind must be probabilities or weights")
        if arr.shape == (n,):
            arr = arr[None]
            batch = False
        elif arr.ndim == 2 and arr.shape[1] == n:
            batch = True
        else:
            raise ValueError("probabilities must have shape (n,) or (K,n)")
        out = np.empty((*arr.shape, 2), object)
        for idx in np.ndindex(arr.shape):
            p = _fraction(arr[idx])
            if not 0 <= p <= 1:
                raise ValueError("probabilities must be in [0,1]")
            out[idx + (0,)] = 1 - p
            out[idx + (1,)] = p
        return out, batch

    def _prepare_exact(self, inputs=None, weights=None, input_kind="probabilities"):
        arr, batch = self._inputs(inputs, weights, input_kind)
        if batch:
            raise ValueError("exact_rational currently accepts one weight vector")
        pairs = []
        den = 1
        for row in arr[0]:
            p = tuple(map(_fraction, row))
            if min(p) < 0:
                raise ValueError("weights must be nonnegative")
            d = lcm(*(x.denominator for x in p))
            pairs.append(tuple(int(x * d) for x in p))
            den *= d
        return pairs, den

    def _prepare_logs(self, inputs=None, weights=None, input_kind="probabilities"):
        source = weights if weights is not None else inputs
        kind = "weights" if weights is not None else input_kind
        raw = (
            source.detach().cpu().numpy()
            if isinstance(source, torch.Tensor)
            else np.asarray(source)
        )
        n = self.original_num_vars
        if raw.size * 24 > self.max_memory_bytes:
            raise ResourceLimitError("input batch exceeds max_memory_bytes")

        def has_rational_pair(value, depth=0):
            if (
                depth >= (2 if kind == "weights" else 1)
                and isinstance(value, tuple)
                and len(value) == 2
                and all(isinstance(x, (int, np.integer)) for x in value)
            ):
                return True
            return isinstance(value, (list, tuple)) and any(
                has_rational_pair(x, depth + 1) for x in value
            )

        if (
            raw.dtype.kind in "fiu"
            and source is not None
            and not has_rational_pair(source)
        ):
            raw = raw.astype(np.float64, copy=False)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("weights must be finite and nonnegative")
            if kind == "probabilities":
                if (raw > 1).any():
                    raise ValueError("probabilities must be in [0,1]")
                if raw.shape == (n,):
                    raw = raw[None]
                    batch = False
                elif raw.ndim == 2 and raw.shape[1] == n:
                    batch = True
                else:
                    raise ValueError("probabilities must have shape (n,) or (K,n)")
                raw = np.stack((1 - raw, raw), axis=-1)
            elif kind == "weights":
                if raw.shape == (n, 2):
                    raw = raw[None]
                    batch = False
                elif raw.ndim == 3 and raw.shape[1:] == (n, 2):
                    batch = True
                else:
                    raise ValueError("weights must have shape (n,2) or (K,n,2)")
            else:
                raise ValueError("input_kind must be probabilities or weights")
            if weights is not None and inputs is not None:
                raise ValueError("supply probabilities or weights, not both")
            with np.errstate(divide="ignore"):
                return np.log(raw), batch
        arr, batch = self._inputs(inputs, weights, input_kind)
        logs = np.empty(arr.shape, float)
        for idx in np.ndindex(arr.shape):
            x = _fraction(arr[idx])
            if x < 0:
                raise ValueError("weights must be finite and nonnegative")
            logs[idx] = -np.inf if x == 0 else log(x.numerator) - log(x.denominator)
        return logs, batch

    def _evaluate_floating(self, logs, batched, target, dtype):
        w = torch.as_tensor(logs, dtype=dtype, device=target)
        batch = len(w)
        tensors = self._cached_tensors(target, dtype)

        def segment(values, indices, size):
            indices = indices.expand(batch, -1)
            maxima = torch.full((batch, size), -torch.inf, device=target, dtype=dtype)
            maxima.scatter_reduce_(1, indices, values, reduce="amax", include_self=True)
            shifts = maxima.gather(1, indices)
            scaled = torch.where(
                torch.isfinite(shifts),
                torch.exp(values - shifts),
                torch.zeros_like(values),
            )
            sums = torch.zeros_like(maxima)
            sums.scatter_add_(1, indices, scaled)
            return maxima + torch.log(sums)

        up = []
        row_logs = {}
        for i, node in enumerate(self.nodes):
            cache = tensors[i]
            if node.left < 0:
                values = torch.zeros(
                    (batch, len(node.key_idx)), device=target, dtype=dtype
                )
                for j, var in enumerate(node.original_vars):
                    values += w[:, var, cache["row_bits"][j]]
                row_logs[i] = values
                up.append(segment(values, cache["key_idx"], node.key_count))
            else:
                values = (
                    up[node.left][:, cache["pair_left"]]
                    + up[node.right][:, cache["pair_right"]]
                )
                up.append(segment(values, cache["out_idx"], node.key_count))
        down = [torch.full_like(value, -torch.inf) for value in up]
        down[self.root][:] = 0
        for i in range(len(self.nodes) - 1, -1, -1):
            node, cache = self.nodes[i], tensors[i]
            if node.left >= 0:
                down[node.left] = segment(
                    down[i][:, cache["out_idx"]]
                    + up[node.right][:, cache["pair_right"]],
                    cache["pair_left"],
                    self.nodes[node.left].key_count,
                )
                down[node.right] = segment(
                    down[i][:, cache["out_idx"]] + up[node.left][:, cache["pair_left"]],
                    cache["pair_right"],
                    self.nodes[node.right].key_count,
                )
        core = torch.logsumexp(up[self.root], 1)
        totals = torch.logsumexp(w, 2)
        log_partition = core + (
            totals[:, list(self.free_vars)].sum(1) if self.free_vars else 0
        )
        marginal = torch.full(
            (batch, self.original_num_vars), torch.nan, device=target, dtype=dtype
        )
        for i, leaf in enumerate(self.leaves):
            cache = tensors[i]
            if leaf.original_vars:
                normalized_logs = (
                    row_logs[i] + down[i][:, cache["key_idx"]] - core[:, None]
                )
                probabilities = torch.exp(normalized_logs)
                marginal[:, cache["original_vars"]] = (
                    probabilities @ cache["owned_bits"]
                )
                # Many individually tiny row masses can sum to a representable
                # marginal. Keep those rare reductions in log space rather
                # than rounding every row to zero before matrix multiplication.
                floor = -700.0 if dtype == torch.float64 else -80.0
                rare = (
                    torch.isfinite(normalized_logs) & (normalized_logs < floor)
                ).any(dim=1)
                if rare.any().item():
                    rare_logs = normalized_logs[rare]
                    for j, var in enumerate(leaf.original_vars):
                        true_rows = cache["row_bits"][j].bool()
                        marginal[rare, var] = torch.exp(
                            torch.logsumexp(rare_logs[:, true_rows], dim=1)
                        )
        for var in self.free_vars:
            marginal[:, var] = torch.exp(w[:, var, 1] - totals[:, var])
        marginal = torch.where(
            torch.isfinite(log_partition[:, None]), marginal, torch.nan
        )
        prior = totals.sum(1)
        kl = torch.where(
            torch.isfinite(log_partition), prior - log_partition, torch.nan
        )
        values = [
            torch.exp(log_partition),
            marginal,
            log_partition,
            torch.exp(log_partition - prior),
            kl,
        ]
        return WMCResult(*(value if batched else value[0] for value in values))

    def _floating_device(self, choice):
        """Probe the actual floating operators, independently of modular int64 support."""
        choices = ["cuda", "mps", "cpu"] if choice == "auto" else [choice]
        failures = []
        for candidate in choices:
            if candidate not in self._probe_cache:
                try:
                    target = torch.device(candidate)
                    value = torch.tensor([0.0, 1.0], device=target)
                    probe = torch.logsumexp(value, 0)
                    acc = torch.full((1,), -torch.inf, device=target)
                    indices = torch.tensor([0, 0], device=target)
                    acc.scatter_reduce_(
                        0, indices, value, reduce="amax", include_self=True
                    )
                    acc.scatter_add_(0, indices, torch.exp(value - value.max()))
                    matrix = torch.ones((1, 2), device=target) @ torch.ones(
                        (2, 1), device=target
                    )
                    if not torch.isfinite(probe).item() or matrix.item() != 2:
                        raise RuntimeError(
                            "floating device probe returned an invalid result"
                        )
                    self._probe_cache[candidate] = None
                except Exception as exc:
                    self._probe_cache[candidate] = str(exc)
            failure = self._probe_cache[candidate]
            if failure is None:
                self.stats.device = str(candidate)
                self.stats.device_fallbacks = failures
                return torch.device(candidate)
            failures.append(f"{candidate}: {failure}")
        raise UnsupportedDeviceError(
            "float32 operations unavailable: " + "; ".join(failures)
        )

    def evaluate(
        self,
        inputs=None,
        *,
        weights=None,
        input_kind="probabilities",
        mode="float64",
        device=None,
    ) -> WMCResult:
        """Evaluate one vector or a floating batch on the cached contraction tree.

        Floating zero-mass posteriors are NaN; exact undefined entries are None.
        ``log_partition`` stays meaningful when floating partition mass underflows.
        """
        if mode == "exact_rational":
            from ._guardrail_exact import evaluate_exact

            w, den = self._prepare_exact(inputs, weights, input_kind)
            return evaluate_exact(self, w, den)
        if mode in ("mps", "cuda", "cpu"):
            device, mode = mode, "float32"
        if mode not in ("float64", "float32"):
            raise ValueError(
                "mode must be exact_rational, float64, float32, mps, cuda, or cpu"
            )
        if mode == "float64" and device not in (None, "cpu", "auto"):
            raise ValueError("float64 evaluation requires the CPU device")
        target = (
            torch.device("cpu")
            if mode == "float64"
            else self._floating_device(device or self.device)
        )
        logs, batched = self._prepare_logs(inputs, weights, input_kind)
        if len(logs) == 0:
            raise ValueError("batch must contain at least one weight vector")
        dtype = torch.float32 if mode == "float32" else torch.float64
        rows = (
            sum(
                n.key_count * 3
                + (
                    len(n.pair_left) * 4
                    if n.pair_left is not None
                    else len(n.key_idx) * 3
                )
                for n in self.nodes
            )
            + self.original_num_vars * 4
        )
        bytes_per = max(1, rows * (4 if mode == "float32" else 8))
        if bytes_per > self.max_memory_bytes:
            raise ResourceLimitError("evaluation workspace exceeds max_memory_bytes")
        if rows * len(logs) > self.max_work:
            raise ResourceLimitError("evaluation batch exceeds max_work")
        chunk = max(1, self.max_memory_bytes // bytes_per)
        if not batched:
            return self._evaluate_floating(logs, False, target, dtype)
        pieces = [
            self._evaluate_floating(logs[i : i + chunk], True, target, dtype)
            for i in range(0, len(logs), chunk)
        ]
        return WMCResult(
            *(
                torch.cat([getattr(p, name) for p in pieces], 0)
                for name in (
                    "partition_function",
                    "posterior_marginals",
                    "log_partition",
                    "acceptance_prob",
                    "kl_divergence_nats",
                )
            )
        )

    def map_inference(self, *args, **kwargs):
        return self.map(*args, **kwargs)

    def sample_conditional(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def exact_marginals(self, inputs=None, **kwargs):
        return self.evaluate(inputs, mode="exact_rational", **kwargs)

    def map(self, *args, **kwargs):
        from ._guardrail_inference import map_inference

        return map_inference(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        from ._guardrail_inference import sample_conditional

        return sample_conditional(self, *args, **kwargs)


def compile_guardrail(cnf: CNF, **options) -> CompiledGuardrail:
    """Compile CNF support on CPU once for repeated weighted evaluations.

    ``max_width`` and ``max_degree`` optionally introduce neutral auxiliary
    variables. Only the original variables receive user weights.
    """
    return CompiledGuardrail(cnf, **options)


def wmc_sharp_sat(
    cnf: CNF,
    inputs=None,
    *,
    weights=None,
    mode="exact_rational",
    input_kind="weights",
    **options,
):
    """Return WMC; positional inputs are explicit false/true weight pairs.

    Use ``input_kind='probabilities'`` for Bernoulli probability vectors.
    Floating modes return tensors; the default returns an exact Fraction.
    """
    compiled = compile_guardrail(cnf, **options)
    return compiled.evaluate(
        inputs, weights=weights, mode=mode, input_kind=input_kind
    ).partition_function


def exact_marginals(
    cnf: CNF,
    inputs=None,
    *,
    weights=None,
    input_kind="probabilities",
    mode="float64",
    device=None,
    **options,
) -> WMCResult:
    """Return all posterior marginals, using the requested arithmetic precision."""
    return compile_guardrail(cnf, **options).evaluate(
        inputs, weights=weights, input_kind=input_kind, mode=mode, device=device
    )


def guardrail_project(
    cnf: CNF,
    inputs=None,
    *,
    weights=None,
    input_kind="probabilities",
    mode="float64",
    device=None,
    **options,
) -> GuardrailResult:
    """Return a posterior projection plus MAP using one compiled support tree.

    Exact mode accepts one vector and returns Fraction values. Floating modes
    return evaluation tensors and use exact rational CPU MAP. Batches are intended
    for ``CompiledGuardrail.evaluate``; projection accepts one vector.
    """
    compiled = compile_guardrail(cnf, **options)
    result = compiled.evaluate(
        inputs, weights=weights, input_kind=input_kind, mode=mode, device=device
    )
    map_mode = "exact_rational"
    assignment, weight = compiled.map_inference(
        inputs, weights=weights, input_kind=input_kind, mode=map_mode
    )
    return GuardrailResult(
        result.partition_function,
        result.posterior_marginals,
        result.log_partition,
        result.acceptance_prob,
        result.kl_divergence_nats,
        assignment,
        weight,
    )
