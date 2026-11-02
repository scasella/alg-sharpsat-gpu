"""Optional C kernel for fused single float64 queries.

The kernel runs ``FusedPlan.evaluate`` together with the input checks and
result assembly of ``CompiledGuardrail._evaluate_single`` in one call, which
removes about twenty NumPy dispatches per query. It is compiled with the
system C compiler the first time a fused plan is built and cached in a private
per-user directory, keyed by a hash of the source, flags and platform. Any
failure (no compiler, a sandbox, an unwritable or shared cache) leaves the
NumPy path in use. Set ``ALG_SHARPSAT_NATIVE=0`` to disable the kernel and
``ALG_SHARPSAT_CACHE`` to choose the cache directory.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = r"""
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Rows are compressed: literal indices 2v + bit in lit_idx[lit_ptr[r]:
   lit_ptr[r + 1]], and the variables whose bit is 1 in one_var likewise. */
typedef struct {
    int64_t n, rows, keys, groups, count, nfree, owned;
    double log_rows, safe_range;
    const int64_t *lit_ptr, *lit_idx, *one_ptr, *one_var;
    const int64_t *row_key, *combos, *free_vars;
} plan_t;

/* Mirrors FusedPlan.evaluate and CompiledGuardrail._evaluate_single.
   values holds n probabilities or n (false, true) weight pairs. out receives
   n marginals, then Z, log Z, acceptance and KL divergence. Returns 1 when
   the caller must use the general path: an invalid weight, a variable whose
   two weights are zero, or literal bounds outside the safe range. */
int alg_fused_query(const plan_t *p, const double *values,
                    int64_t probabilities, double *out)
{
    const int64_t n = p->n, rows = p->rows, keys = p->keys, count = p->count;
    const int64_t groups = p->groups;
    double low = INFINITY, top = 0.0, prior = 0.0;
    double *lit = malloc(sizeof(double) * (size_t)(2 * n + rows + 2 * keys));
    if (!lit) return 1;
    double *w = lit + 2 * n, *up = w + rows, *mass = up + keys;
    for (int64_t v = 0; v < n; v++) {
        double a, b;
        if (probabilities) { b = values[v]; a = 1.0 - b; }
        else { a = values[2 * v]; b = values[2 * v + 1]; }
        if (!(a >= 0.0 && b >= 0.0 && a < INFINITY && b < INFINITY && a + b > 0.0)) {
            free(lit);
            return 1;
        }
        if (a > 0.0 && a < low) low = a;
        if (b > 0.0 && b < low) low = b;
        if (a > top) top = a;
        if (b > top) top = b;
        lit[2 * v] = a;
        lit[2 * v + 1] = b;
        prior += log(a + b);
    }
    if ((double)p->owned * fmax(0.0, -log(low)) > p->safe_range
        || p->log_rows + (double)p->owned * fmax(0.0, log(top)) > p->safe_range) {
        free(lit);
        return 1;
    }
    /* Four partial products shorten the multiply dependency chain. */
    for (int64_t r = 0; r < rows; r++) {
        int64_t t = p->lit_ptr[r];
        const int64_t end = p->lit_ptr[r + 1];
        double x0 = 1.0, x1 = 1.0, x2 = 1.0, x3 = 1.0;
        for (; t + 4 <= end; t += 4) {
            x0 *= lit[p->lit_idx[t]];
            x1 *= lit[p->lit_idx[t + 1]];
            x2 *= lit[p->lit_idx[t + 2]];
            x3 *= lit[p->lit_idx[t + 3]];
        }
        for (; t < end; t++) x0 *= lit[p->lit_idx[t]];
        w[r] = (x0 * x1) * (x2 * x3);
    }
    memset(up, 0, sizeof(double) * (size_t)(2 * keys));
    for (int64_t r = 0; r < rows; r++) up[p->row_key[r]] += w[r];
    /* Neumaier summation keeps the core as accurate as a pairwise sum. */
    double core = 0.0, carry = 0.0;
    for (int64_t c = 0; c < count; c++) {
        double x = 1.0;
        for (int64_t g = 0; g < groups; g++) x *= up[p->combos[g * count + c]];
        double t = core + x;
        carry += fabs(core) >= fabs(x) ? (core - t) + x : (x - t) + core;
        core = t;
        for (int64_t g = 0; g < groups; g++) mass[p->combos[g * count + c]] += x;
    }
    core += carry;
    for (int64_t k = 0; k < keys; k++) mass[k] = up[k] > 0.0 ? mass[k] / up[k] : 0.0;
    memset(out, 0, sizeof(double) * (size_t)n);
    for (int64_t r = 0; r < rows; r++) {
        const double q = w[r] * mass[p->row_key[r]];
        for (int64_t t = p->one_ptr[r]; t < p->one_ptr[r + 1]; t++) out[p->one_var[t]] += q;
    }
    double log_partition = -INFINITY, kl = NAN;
    if (core > 0.0) {
        log_partition = log(core);
        for (int64_t f = 0; f < p->nfree; f++) {
            int64_t v = p->free_vars[f];
            log_partition += log(lit[2 * v] + lit[2 * v + 1]);
        }
    }
    if (log_partition > -INFINITY) {
        for (int64_t v = 0; v < n; v++) out[v] /= core;
        for (int64_t f = 0; f < p->nfree; f++) {
            int64_t v = p->free_vars[f];
            out[v] = lit[2 * v + 1] / (lit[2 * v] + lit[2 * v + 1]);
        }
        kl = prior - log_partition;
    } else {
        for (int64_t v = 0; v < n; v++) out[v] = NAN;
    }
    out[n] = exp(log_partition);
    out[n + 1] = log_partition;
    out[n + 2] = exp(log_partition - prior);
    out[n + 3] = kl;
    free(lit);
    return 0;
}
"""

# No fast-math: the kernel relies on IEEE infinities and NaN comparisons.
FLAGS = ("-O2", "-ffp-contract=off", "-shared", "-fPIC")


class Plan(ctypes.Structure):
    _fields_ = [
        *((name, ctypes.c_int64) for name in
          ("n", "rows", "keys", "groups", "count", "nfree", "owned")),
        ("log_rows", ctypes.c_double),
        ("safe_range", ctypes.c_double),
        *((name, ctypes.c_void_p) for name in
          ("lit_ptr", "lit_idx", "one_ptr", "one_var", "row_key", "combos", "free_vars")),
    ]


_query = None  # None: not tried yet; False: unavailable


def kernel():
    """The compiled ``alg_fused_query`` function, or None if unavailable."""
    global _query
    if _query is None:
        _query = _load() or False
    return _query or None


def _cache_dir() -> Path:
    root = os.environ.get("ALG_SHARPSAT_CACHE")
    if root:
        base = Path(root)
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches" / "alg_sharpsat"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "alg_sharpsat"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = base.stat()
    # Only load code from a directory nobody else can write to.
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise OSError(f"{base} is not private to this user")
    return base


def _compiler() -> str | None:
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        return None
    # Without the command line tools, Apple's cc stub opens an install dialog.
    if sys.platform == "darwin" and subprocess.run(
        ["xcode-select", "-p"], capture_output=True, timeout=30
    ).returncode:
        return None
    return compiler


def _load():
    if os.environ.get("ALG_SHARPSAT_NATIVE", "1") == "0" or os.name != "posix":
        return None
    try:
        key = "\0".join((SOURCE, *FLAGS, sys.platform, platform.machine(),
                         str(ctypes.sizeof(ctypes.c_void_p))))
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        target = _cache_dir() / f"fused_{hashlib.sha256(key.encode()).hexdigest()[:16]}{suffix}"
        if not target.exists():
            compiler = _compiler()
            if compiler is None:
                return None
            with tempfile.TemporaryDirectory(dir=target.parent) as scratch:
                source = Path(scratch) / "fused.c"
                source.write_text(SOURCE)
                built = Path(scratch) / target.name
                subprocess.run([compiler, *FLAGS, "-o", str(built), str(source)],
                               check=True, capture_output=True, timeout=120)
                os.replace(built, target)
        if target.stat().st_uid != os.getuid():
            return None
        query = ctypes.CDLL(str(target)).alg_fused_query
    except (OSError, subprocess.SubprocessError):
        return None
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    query.restype = ctypes.c_int
    return query
