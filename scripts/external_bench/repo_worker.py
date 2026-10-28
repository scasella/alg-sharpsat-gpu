"""Time one repository counter on one DIMACS file in-process; print one JSON line.

Engines: ``sparse`` is sparse contraction (``bridge_cut=False``, the v3.4
engine), ``auto`` is ``auto_count_with_stats`` (bridge-cut search when a
two-way cut has at most six bridge clauses, else contraction) and ``dpll`` is
the reference DPLL counter.
"""
import json
import os
import sys
from statistics import median
from time import perf_counter

os.environ.update({k: "1" for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
                                    "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
import torch  # noqa: E402
from alg_sharpsat.cnf import dpll_count, parse_dimacs  # noqa: E402
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError  # noqa: E402
from alg_sharpsat.stage3_sparse_contract import auto_count_with_stats, sparse_count_with_stats  # noqa: E402

engine, path, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3])
torch.set_num_threads(1)
OPTIONS = dict(device="cpu", num_groups=2, block_size=4, max_work=2_000_000_000,
               max_memory_bytes=512 * 1024 * 1024, max_local_assignments=4_000_000,
               max_join_pairs=8_000_000)
CALLS = {
    "sparse": lambda c: sparse_count_with_stats(c, **OPTIONS),
    "auto": lambda c: auto_count_with_stats(c, **OPTIONS),
    "dpll": lambda c: (dpll_count(c), None),
}
call = CALLS[engine]
call(parse_dimacs("p cnf 3 2\n1 2 0\n-2 3 0\n"))  # warm imports and kernels
cnf = parse_dimacs(open(path).read())
samples, count, stats = [], None, None
try:
    for _ in range(repeats):
        start = perf_counter()
        value, stats = call(cnf)
        samples.append(perf_counter() - start)
        assert count is None or value == count
        count = value
except ResourceLimitError as exc:
    print(json.dumps({"status": "resource_limit", "reason": str(exc)[:200]}))
else:
    row = {"status": "ok", "count": str(count), "seconds": median(samples), "samples": samples}
    if stats is not None:
        row["selected_engine"] = stats.selected_engine
        row["bridge_cut"] = stats.bridge_cut
    print(json.dumps(row))
