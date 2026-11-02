"""Vectorized float64 plan evaluation and float-guided exact MAP search.

Leaf rows are sorted by signature once, so every segment sum is one
``reduceat``. Each leaf's row log-weights are shifted by their per-query
maximum before ``exp``: a retained row then lies in ``[exp(-R), 1]``, where R
is that leaf's log range. If the ranges of all leaves sum to at most
``SAFE_RANGE`` nats, no product of rows from different leaves can underflow;
sums cannot overflow while the plan's row-count product stays below
``exp(SAFE_RANGE)``. Queries outside either bound are evaluated by the
log-domain tensor path instead, which keeps rare-row reductions exact.
"""

from __future__ import annotations

import ctypes
from math import log

import numpy as np
from scipy import sparse

SAFE_RANGE = 600.0


def _segments(index, size):
    """Stable order, reduceat starts and the present keys of an index array."""
    order = np.argsort(index, kind="stable")
    ordered = index[order]
    if len(ordered):
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    else:
        starts = np.zeros(0, dtype=np.int64)
    present = ordered[starts]
    return order, starts, None if len(present) == size else present


def _segment_max(values, starts, present, size):
    if not len(starts):
        return np.full(size, -np.inf)
    best = np.maximum.reduceat(values, starts)
    if present is None:
        return best
    out = np.full(size, -np.inf)
    out[present] = best
    return out


def _indicator(index, size):
    """Sparse (size, len(index)) matrix summing entries that share a key."""
    count = len(index)
    return sparse.csr_matrix(
        (np.ones(count), (index, np.arange(count))), shape=(size, count)
    )


class FastPlan:
    """Immutable sorted index arrays derived from a pruned compiled plan."""

    def __init__(self, compiled):
        self.root = compiled.root
        self.items = []
        self.log_rows = 0.0
        for node in compiled.nodes:
            item = {"size": node.key_count}
            if node.left < 0:
                order, starts, present = _segments(node.key_idx, node.key_count)
                bits = node.assignments[order]
                item.update(
                    leaf=True,
                    vars=np.asarray(node.original_vars, dtype=np.int64),
                    keys=node.key_idx[order],
                    starts=starts,
                    present=present,
                    bits=bits,
                    A=np.ascontiguousarray(bits, dtype=np.float64),
                    AT=np.ascontiguousarray(bits.T, dtype=np.float64),
                )
                self.log_rows += log(max(1, len(order)))
            else:
                order, starts, present = _segments(node.out_idx, node.key_count)
                left = node.pair_left[order]
                right = node.pair_right[order]
                item.update(
                    leaf=False,
                    left=node.left,
                    right=node.right,
                    pl=left,
                    pr=right,
                    out=node.out_idx[order],
                    starts=starts,
                    present=present,
                    left_size=compiled.nodes[node.left].key_count,
                    right_size=compiled.nodes[node.right].key_count,
                )
            self.items.append(item)
        self.owned = np.concatenate(
            [it["vars"] for it in self.items if it.get("leaf")] or [np.zeros(0, np.int64)]
        )
        self.free = np.asarray(compiled.free_vars, dtype=np.int64)
        self._sums = False

    def _batch_sums(self):
        """Sparse segment-sum matrices, built on the first batched query."""
        if not self._sums:
            for item in self.items:
                if item["leaf"]:
                    item["sum"] = _indicator(item["keys"], item["size"])
                else:
                    item["sum"] = _indicator(item["out"], item["size"])
                    item["to_left"] = _indicator(item["pl"], item["left_size"])
                    item["to_right"] = _indicator(item["pr"], item["right_size"])
            self._sums = True

    def _leaf_logs(self, item, logs):
        """Row log-weights (rows, K) with -inf rows for zero-weight literals."""
        sub = logs[:, item["vars"], :]
        neg = np.isneginf(sub)
        if neg.any():
            finite = np.where(neg, 0.0, sub)
            delta = (finite[:, :, 1] - finite[:, :, 0]).T
            values = item["A"] @ delta + finite[:, :, 0].sum(1)
            zeros = neg.astype(np.float64)
            hits = item["A"] @ (zeros[:, :, 1] - zeros[:, :, 0]).T + zeros[:, :, 0].sum(1)
            values[hits > 0.5] = -np.inf
            return values
        return item["A"] @ (sub[:, :, 1] - sub[:, :, 0]).T + sub[:, :, 0].sum(1)

    def evaluate_one(self, logs):
        """One (n,2) log-weight vector with 1-D arrays and ``bincount`` sums."""
        neg = np.isneginf(logs)
        zeros = bool(neg.any())
        if zeros:
            finite = np.where(neg, 0.0, logs)
            delta, base = finite[:, 1] - finite[:, 0], finite[:, 0]
            zero_delta = neg[:, 1].astype(np.float64) - neg[:, 0]
            zero_base = neg[:, 0].astype(np.float64)
        else:
            delta, base = logs[:, 1] - logs[:, 0], logs[:, 0]
        up, rows, gathered = {}, {}, {}
        shift = spread = 0.0
        for i, item in enumerate(self.items):
            if item["leaf"]:
                variables = item["vars"]
                values = item["A"] @ delta[variables] + base[variables].sum()
                if zeros:
                    hits = item["A"] @ zero_delta[variables] + zero_base[variables].sum()
                    values[hits > 0.5] = -np.inf
                top = values.max() if len(values) else -np.inf
                if np.isfinite(top):
                    low = values[np.isfinite(values)].min() if zeros else values.min()
                    spread += top - low
                else:
                    top = 0.0
                shift += top
                rows[i] = weights = np.exp(values - top)
                up[i] = np.bincount(item["keys"], weights, item["size"])
            else:
                left = up[item["left"]][item["pl"]]
                right = up[item["right"]][item["pr"]]
                gathered[i] = left, right
                up[i] = np.bincount(item["out"], left * right, item["size"])
        core = up[self.root].sum()
        down = {self.root: np.ones(len(up[self.root]))}
        for i in range(len(self.items) - 1, -1, -1):
            item = self.items[i]
            if item["leaf"]:
                continue
            cavity = down.pop(i)[item["out"]]
            left, right = gathered[i]
            down[item["left"]] = np.bincount(item["pl"], cavity * right, item["left_size"])
            down[item["right"]] = np.bincount(item["pr"], cavity * left, item["right_size"])
        numerators = [
            item["AT"] @ (rows[i] * down[i][item["keys"]])
            for i, item in enumerate(self.items)
            if item["leaf"] and len(item["vars"])
        ]
        owned = np.concatenate(numerators) if numerators else np.zeros(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (
                np.array([np.log(core) + shift]),
                (owned / core)[None],
                np.array([spread <= SAFE_RANGE]),
            )

    def evaluate(self, logs):
        """Return (log_core, owned marginals (K,len(owned)), safe mask) or None.

        ``None`` means the plan's row-count product exceeds the overflow bound.
        """
        if self.log_rows > SAFE_RANGE:
            return None
        if logs.shape[0] == 1:
            return self.evaluate_one(logs[0])
        self._batch_sums()
        batch = logs.shape[0]
        up, rows, gathered = {}, {}, {}
        shift_total, spread = np.zeros(batch), np.zeros(batch)
        zeros = bool(np.isneginf(logs).any())
        for i, item in enumerate(self.items):
            if item["leaf"]:
                values = self._leaf_logs(item, logs)
                if values.shape[0]:
                    top = values.max(0)
                    if zeros:
                        alive = np.isfinite(top)
                        top = np.where(alive, top, 0.0)
                        low = np.where(np.isfinite(values), values, np.inf).min(0)
                        spread += np.where(alive, top - low, 0.0)
                    else:
                        spread += top - values.min(0)
                    weights = np.exp(values - top)
                else:
                    top = np.zeros(batch)
                    weights = values
                shift_total += top
                rows[i] = weights
                up[i] = item["sum"] @ weights
            else:
                left = up[item["left"]][item["pl"]]
                right = up[item["right"]][item["pr"]]
                gathered[i] = left, right
                up[i] = item["sum"] @ (left * right)
        core = up[self.root].sum(0)
        down = {self.root: np.ones_like(up[self.root])}
        for i in range(len(self.items) - 1, -1, -1):
            item = self.items[i]
            if item["leaf"]:
                continue
            cavity = down.pop(i)[item["out"]]
            left, right = gathered[i]
            down[item["left"]] = item["to_left"] @ (cavity * right)
            down[item["right"]] = item["to_right"] @ (cavity * left)
        numerators = []
        for i, item in enumerate(self.items):
            if item["leaf"] and len(item["vars"]):
                numerators.append(item["AT"] @ (rows[i] * down[i][item["keys"]]))
        owned = (
            np.concatenate(numerators, 0).T if numerators else np.zeros((batch, 0))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            log_core = np.log(core) + shift_total
            marginals = owned / core[:, None]
        safe = spread <= SAFE_RANGE
        return log_core, marginals, safe

    def map_search(self, scores_by_var):
        """Float max-product over the plan; return leaf row choices or None.

        ``scores_by_var`` is (n,2) float log-weights. Returns ``None`` when any
        comparison on the chosen path is within the rounding tolerance, so the
        caller can resolve ties and near ties with exact integer arithmetic.
        """
        # Every candidate score is a sum of at most 2n+L rounded terms whose
        # magnitudes are bounded by the per-variable maxima; leave a margin.
        finite = np.where(np.isfinite(scores_by_var), np.abs(scores_by_var), 0.0)
        terms = 4 * len(scores_by_var) + 4 * len(self.items) + 16
        tolerance = 1e-9 + terms * np.finfo(np.float64).eps * (float(finite.max(1).sum()) + 1)
        logs = scores_by_var[None]
        up, rows = {}, {}
        for i, item in enumerate(self.items):
            if item["leaf"]:
                values = self._leaf_logs(item, logs)[:, 0]
                rows[i] = values
            else:
                values = up[item["left"]][item["pl"]] + up[item["right"]][item["pr"]]
                rows[i] = values
            up[i] = _segment_max(values, item["starts"], item["present"], item["size"])
        root = up[self.root]
        if not len(root) or not np.isfinite(root.max()):
            return "zero"

        def pick(values):
            best = int(np.argmax(values))
            if len(values) > 1:
                top2 = np.partition(values, len(values) - 2)[-2:]
                if top2[1] - top2[0] <= tolerance:
                    return None
            return best

        key = pick(root)
        if key is None:
            return None
        choices = {}
        stack = [(self.root, key)]
        while stack:
            i, key = stack.pop()
            item = self.items[i]
            if item["present"] is None:
                segment = key
            else:
                segment = int(np.searchsorted(item["present"], key))
            start = item["starts"][segment]
            stop = item["starts"][segment + 1] if segment + 1 < len(item["starts"]) else len(rows[i])
            local = pick(rows[i][start:stop])
            if local is None:
                return None
            choice = start + local
            if item["leaf"]:
                choices[i] = choice
            else:
                stack.append((item["left"], int(item["pl"][choice])))
                stack.append((item["right"], int(item["pr"][choice])))
        return choices


# A flat table holds at most this many (combination, leaf) entries, and the
# owned-bit matrix at most this many (variable, row) entries. Without the C
# kernel, plans whose padded row gather exceeds FUSED_NUMPY_MAX_GATHER entries
# keep the general path, which is faster there.
FUSED_MAX_ENTRIES = 1 << 16
FUSED_MAX_BITS = 1 << 20
FUSED_NUMPY_MAX_GATHER = 1 << 16


class FusedPlan:
    """One-query float64 evaluation over a flat table of signature combinations.

    Each root-consistent choice of one signature per leaf is listed once, so
    the upward pass is one product per combination and the downward pass one
    segment sum per signature: a fixed sequence of about a dozen NumPy calls,
    however many nodes the plan has. Literal weights are used unscaled. With
    every nonzero literal in ``[low, top]``, each nonzero product of owned
    literals lies in ``[low**n, top**n]`` and each sum is at most the number
    of row tuples times ``top**n``; ``safe`` keeps both inside
    ``exp(±SAFE_RANGE)``, so no intermediate is subnormal or infinite.
    """

    def __init__(self, compiled, leaves, rows):
        nodes = compiled.nodes
        n = compiled.original_num_vars
        offsets, start = {}, 0
        for i in leaves:
            offsets[i] = start
            start += nodes[i].key_count
        self.keys = start
        width = max(1, max(len(nodes[i].original_vars) for i in leaves))
        # Row r multiplies literals[gather[:, r]]; index 2n holds the constant 1.
        self.gather = np.full((width, rows), 2 * n, dtype=np.int64)
        self.row_key = np.empty(rows, dtype=np.int64)
        self.bits = np.zeros((n, rows))
        r = 0
        for i in leaves:
            node = nodes[i]
            count = len(node.key_idx)
            variables = np.asarray(node.original_vars, dtype=np.int64)
            bits = node.assignments.T
            self.gather[: len(variables), r : r + count] = 2 * variables[:, None] + bits
            self.row_key[r : r + count] = offsets[i] + node.key_idx
            self.bits[variables, r : r + count] = bits
            r += count
        # Expand combinations bottom-up; each node's table is grouped by its key.
        tables = {}
        for i, node in enumerate(nodes):
            if node.left < 0:
                table = np.full((len(leaves), node.key_count), -1, dtype=np.int64)
                table[leaves.index(i)] = offsets[i] + np.arange(node.key_count)
                tables[i] = table, np.arange(node.key_count)
                continue
            (left, left_key), (right, right_key) = tables.pop(node.left), tables.pop(node.right)
            left_count = np.bincount(left_key, minlength=nodes[node.left].key_count)
            right_count = np.bincount(right_key, minlength=nodes[node.right].key_count)
            left_start = np.cumsum(left_count) - left_count
            right_start = np.cumsum(right_count) - right_count
            sizes = left_count[node.pair_left] * right_count[node.pair_right]
            pair = np.repeat(np.arange(len(sizes)), sizes)
            offset = np.arange(len(pair)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
            across = right_count[node.pair_right][pair]
            li = left_start[node.pair_left][pair] + offset // across
            ri = right_start[node.pair_right][pair] + offset % across
            key = node.out_idx[pair]
            order = np.argsort(key, kind="stable")
            table = np.maximum(left[:, li], right[:, ri])[:, order]
            tables[i] = np.ascontiguousarray(table), key[order]
        self.combos = tables[compiled.root][0]
        # Signature masses are segment sums of combination products; every
        # retained signature occurs in at least one combination.
        order, self.starts, self.missing = _segments(self.combos.ravel(), self.keys)
        self.order = order % self.combos.shape[1]
        self.free = np.asarray(compiled.free_vars, dtype=np.int64)
        self.owned = n - len(self.free)

    @classmethod
    def build(cls, compiled):
        """Return a fused plan, or None when its tables would be too large."""
        nodes = compiled.nodes
        leaves = [i for i, node in enumerate(nodes) if node.left < 0]
        rows = sum(len(nodes[i].key_idx) for i in leaves)
        log_rows = sum(log(max(1, len(nodes[i].key_idx))) for i in leaves)
        combinations = []
        for node in nodes:
            if node.left < 0:
                combinations.append(np.ones(node.key_count))
            else:
                products = combinations[node.left][node.pair_left] * combinations[node.right][node.pair_right]
                combinations.append(np.bincount(node.out_idx, products, node.key_count))
        entries = float(combinations[compiled.root].sum()) * len(leaves)
        if (
            not 0 < entries <= FUSED_MAX_ENTRIES
            or compiled.original_num_vars * rows > FUSED_MAX_BITS
            or log_rows > SAFE_RANGE
        ):
            return None
        plan = cls(compiled, leaves, rows)
        plan.log_rows = log_rows
        if plan.missing is not None:
            return None
        plan._attach_native()
        return plan

    def _attach_native(self):
        """Use the C kernel when it builds and reproduces this plan's results."""
        from ._guardrail_native import Plan, kernel

        self.native = None
        query = kernel()
        if query is None:
            return
        n = self.bits.shape[0]
        self.free = np.ascontiguousarray(self.free, dtype=np.int64)
        # Compressed rows: each row's literal indices without padding, and
        # the variables it sets to true.
        literals = self.gather.T
        valid = literals != 2 * n
        ones = valid & (literals & 1).astype(bool)
        self.native_arrays = (
            np.concatenate(([0], np.cumsum(valid.sum(1)))),
            literals[valid],
            np.concatenate(([0], np.cumsum(ones.sum(1)))),
            literals[ones] >> 1,
        )
        self.native_plan = Plan(
            n, self.gather.shape[1], self.keys, self.combos.shape[0],
            self.combos.shape[1], len(self.free), self.owned, self.log_rows, SAFE_RANGE,
            *(array.ctypes.data for array in self.native_arrays),
            self.row_key.ctypes.data, self.combos.ctypes.data, self.free.ctypes.data,
        )
        self.native_address = ctypes.addressof(self.native_plan)
        # A self-check guards against a stale or mismatched shared library.
        weights = np.tile([0.375, 0.625], (n, 1))
        out = np.empty(n + 4)
        core, numerators = self.evaluate(np.append(weights.ravel(), 1.0), False)
        expected = numerators / core
        expected[self.free] = 0.625
        if not query(self.native_address, weights.ctypes.data, 0, out.ctypes.data) and (
            np.allclose(out[:n], expected, rtol=0, atol=1e-12)
            and abs(out[n + 1] - log(core)) <= 1e-12 * max(1.0, abs(log(core)))
        ):
            self.native = query

    def safe(self, low, top):
        """Whether nonzero literals in ``[low, top]`` keep every value normal."""
        return (
            self.owned * max(0.0, -log(low)) <= SAFE_RANGE
            and self.log_rows + self.owned * max(0.0, log(top)) <= SAFE_RANGE
        )

    @property
    def numpy_ok(self):
        return self.gather.size <= FUSED_NUMPY_MAX_GATHER

    def evaluate(self, literals, zeros):
        """Return (core, numerators) for flat literal weights with a final 1."""
        rows = literals[self.gather].prod(0)
        up = np.bincount(self.row_key, rows, self.keys)
        products = up[self.combos].prod(0)
        mass = np.add.reduceat(products[self.order], self.starts)
        if zeros:
            down = np.divide(mass, up, out=np.zeros(self.keys), where=up > 0)
        else:
            down = mass / up
        return float(products.sum()), self.bits @ (rows * down[self.row_key])
