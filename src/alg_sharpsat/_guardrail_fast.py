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
