"""Max-product backtracking and rejection-free sampling on a compiled join tree.

Integer-mode draws use randrange(total), so neither huge integer weights nor
rare events are rounded to a floating-point categorical distribution.
"""

from __future__ import annotations

from bisect import bisect_right
from fractions import Fraction
from math import exp, log
import random

import numpy as np

ZERO_MASS = "Formula is unsatisfiable or has zero probability mass"


def _prepare(compiled, inputs, weights, input_kind, mode):
    if mode == "exact_rational":
        values, denominator = compiled._prepare_exact(inputs, weights, input_kind)
        return values, denominator, True
    if mode not in ("float64", "float32", "mps"):
        raise ValueError("mode must be exact_rational, float64, float32, or mps")
    values, batched = compiled._prepare_logs(inputs, weights, input_kind)
    if batched:
        raise ValueError("MAP and sampling accept one weight vector at a time")
    return values[0], 1, False


def _inference_budget(compiled, weights, exact, num_samples=0):
    """Bound cached candidate storage and traversal work before allocation."""
    from .stage3_gpu_engine import ResourceLimitError

    entries = sum(
        n.key_count + (len(n.key_idx) if n.left < 0 else len(n.pair_left))
        for n in compiled.nodes
    )
    bits = 64
    if exact:
        bits = 1 + sum(max(a, b, 1).bit_length() for a, b in weights)
        bits += sum(max(1, len(n.key_idx)).bit_length() for n in compiled.leaves)
    if entries * (128 + (bits + 7) // 8) > compiled.max_memory_bytes:
        raise ResourceLimitError("inference messages exceed max_memory_bytes")
    work = sum(
        len(n.key_idx) * (len(n.original_vars) + 2)
        if n.left < 0
        else len(n.pair_left) * 3
        for n in compiled.nodes
    )
    work += num_samples * (len(compiled.nodes) + len(compiled.free_vars))
    if work * max(1, bits // 64) > compiled.max_work:
        raise ResourceLimitError("inference traversal exceeds max_work")


def _row_values(node, weights, exact):
    if exact:
        values = [1] * len(node.key_idx)
        for j, var in enumerate(node.original_vars):
            a, b = weights[var]
            for row, bit in enumerate(node.assignments[:, j]):
                values[row] *= b if bit else a
        return values
    values = np.zeros(len(node.key_idx))
    for j, var in enumerate(node.original_vars):
        values += weights[var, node.assignments[:, j].astype(int)]
    return values


def _buckets(compiled):
    """Compile inverse maps once, independent of all subsequent weights."""
    if not hasattr(compiled, "_sampling_buckets"):
        buckets = []
        for node in compiled.nodes:
            index = node.key_idx if node.left < 0 else node.out_idx
            groups = [[] for _ in range(node.key_count)]
            for i, key in enumerate(index):
                groups[int(key)].append(i)
            buckets.append(groups)
        compiled._sampling_buckets = buckets
    return compiled._sampling_buckets


def map_inference(
    compiled,
    inputs=None,
    *,
    weights=None,
    input_kind="probabilities",
    mode="exact_rational",
):
    """Return a safe maximum-weight assignment, with stable first-choice ties.

    The default returns an exact Fraction. Floating modes maximize representable
    log weights on CPU and return a float; the returned weight can underflow.
    """
    w, denominator, exact = _prepare(compiled, inputs, weights, input_kind, mode)
    _inference_budget(compiled, w, exact)
    found = _float_guided_map(compiled, w, denominator, exact)
    if found is not None:
        return found
    up, back = [], []
    for node in compiled.nodes:
        values = (
            _row_values(node, w, exact)
            if node.left < 0
            else [
                up[node.left][int(l)] * up[node.right][int(r)]
                if exact
                else up[node.left][int(l)] + up[node.right][int(r)]
                for l, r in zip(node.pair_left, node.pair_right)
            ]
        )
        best = [-1 if exact else -np.inf] * node.key_count
        choices = [-1] * node.key_count
        index = node.key_idx if node.left < 0 else node.out_idx
        for i, key in enumerate(index):
            key = int(key)
            if choices[key] < 0 or values[i] > best[key]:
                best[key], choices[key] = values[i], i
        up.append(best)
        back.append(choices)
    root_values = up[compiled.root]
    if not root_values:
        raise ValueError(ZERO_MASS)
    key = max(range(len(root_values)), key=lambda i: root_values[i])
    score = root_values[key]
    assignment = np.zeros(compiled.original_num_vars, dtype=np.int8)
    for var in compiled.free_vars:
        bit = int(w[var][1] > w[var][0])
        assignment[var] = bit
        score = score * w[var][bit] if exact else score + w[var][bit]
    if (exact and score <= 0) or (not exact and not np.isfinite(score)):
        raise ValueError(ZERO_MASS)

    def descend(index, output):
        node = compiled.nodes[index]
        choice = back[index][output]
        if node.left < 0:
            assignment[list(node.original_vars)] = node.assignments[choice]
        else:
            descend(node.left, int(node.pair_left[choice]))
            descend(node.right, int(node.pair_right[choice]))

    descend(compiled.root, key)
    if exact:
        result_weight = Fraction(score, denominator)
    else:
        # exp(log weight) may legitimately be outside float64's dynamic range.
        with np.errstate(over="ignore", under="ignore"):
            result_weight = float(np.exp(score))
    return tuple(map(int, assignment)), result_weight


def _float_guided_map(compiled, w, denominator, exact):
    """Max-product in float64 when every choice on the path wins by a margin.

    A winning margin above the rounding bound makes the float argmax the exact
    argmax, so the result equals the integer pass below. Ties and near ties
    return None and are resolved by that pass, which keeps first-choice ties.
    """
    if exact:
        scores = np.array(
            [[log(a) if a else -np.inf, log(b) if b else -np.inf] for a, b in w]
        ).reshape(len(w), 2)
    else:
        scores = np.asarray(w, dtype=np.float64).reshape(len(w), 2)
    plan = compiled._fast()
    choices = plan.map_search(scores)
    if not isinstance(choices, dict):
        return None
    assignment = np.zeros(compiled.original_num_vars, dtype=np.int8)
    for index, row in choices.items():
        item = plan.items[index]
        assignment[item["vars"]] = item["bits"][row]
    for var in compiled.free_vars:
        assignment[var] = int(w[var][1] > w[var][0])
    if exact:
        score = 1
        for var, bit in enumerate(assignment.tolist()):
            score *= w[var][bit]
        if score <= 0:
            raise ValueError(ZERO_MASS)
        return tuple(assignment.tolist()), Fraction(score, denominator)
    score = float(scores[np.arange(len(assignment)), assignment].sum())
    if not np.isfinite(score):
        raise ValueError(ZERO_MASS)
    with np.errstate(over="ignore", under="ignore"):
        return tuple(assignment.tolist()), float(np.exp(score))


def sample_conditional(
    compiled,
    inputs=None,
    num_samples: int = 1,
    seed: int = 0,
    *,
    weights=None,
    input_kind="probabilities",
    mode="exact_rational",
) -> np.ndarray:
    """Draw independent valid models without rejection.

    Exact mode (the default) performs integer categorical draws. Floating modes
    use log-scaled distributions with ordinary finite-precision sampling error.
    A fixed seed gives the same prefix regardless of num_samples.
    """
    if type(num_samples) is not int or num_samples < 0:
        raise ValueError("num_samples must be a nonnegative integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if num_samples * max(1, compiled.original_num_vars) > compiled.max_memory_bytes:
        from .stage3_gpu_engine import ResourceLimitError

        raise ResourceLimitError("sample output exceeds max_memory_bytes")
    w, _, exact = _prepare(compiled, inputs, weights, input_kind, mode)
    _inference_budget(compiled, w, exact, num_samples)
    up, candidate_values = [], []
    buckets = _buckets(compiled)
    for node, groups in zip(compiled.nodes, buckets):
        values = (
            _row_values(node, w, exact)
            if node.left < 0
            else [
                up[node.left][int(l)] * up[node.right][int(r)]
                if exact
                else up[node.left][int(l)] + up[node.right][int(r)]
                for l, r in zip(node.pair_left, node.pair_right)
            ]
        )
        totals = []
        for group in groups:
            vals = [values[i] for i in group]
            totals.append(
                sum(vals) if exact else np.logaddexp.reduce(vals, initial=-np.inf)
            )
        up.append(totals)
        candidate_values.append(values)
    root_values = up[compiled.root]
    valid = bool(root_values) and (
        sum(root_values) > 0
        if exact
        else np.isfinite(np.logaddexp.reduce(root_values, initial=-np.inf))
    )
    valid = valid and all(
        sum(w[v]) > 0 if exact else np.isfinite(np.logaddexp(*w[v]))
        for v in compiled.free_vars
    )
    if not valid:
        raise ValueError(ZERO_MASS)
    rng = random.Random(seed)
    distributions = {}

    def draw(cache_key, candidates=None, values=None):
        if cache_key not in distributions:
            if exact:
                cumulative, total = [], 0
                for value in values:
                    total += value
                    cumulative.append(total)
            else:
                shift = max(values)
                scaled = [exp(value - shift) for value in values]
                cumulative = np.cumsum(scaled).tolist()
                total = cumulative[-1]
            distributions[cache_key] = candidates, cumulative, total
        candidates, cumulative, total = distributions[cache_key]
        pick = rng.randrange(total) if exact else rng.random() * total
        return candidates[min(bisect_right(cumulative, pick), len(candidates) - 1)]

    result = np.zeros((num_samples, compiled.original_num_vars), dtype=np.int8)

    def descend(index, output, sample):
        node = compiled.nodes[index]
        choices = buckets[index][output]
        cache_key = (index, output)
        if cache_key in distributions:
            choice = draw(cache_key)
        else:
            choice = draw(
                cache_key, choices, [candidate_values[index][i] for i in choices]
            )
        if node.left < 0:
            result[sample, list(node.original_vars)] = node.assignments[choice]
        else:
            descend(node.left, int(node.pair_left[choice]), sample)
            descend(node.right, int(node.pair_right[choice]), sample)

    for sample in range(num_samples):
        key = draw(("root",), list(range(len(root_values))), root_values)
        descend(compiled.root, key, sample)
        for var in compiled.free_vars:
            result[sample, var] = draw(("free", var), [0, 1], w[var])
    return result
