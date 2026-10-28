"""Shared paths and single-thread settings for the external benchmark harness.

Usage of every script: ``python <script> TOOLS WORK [instance prefixes...]``.
TOOLS holds ``ganak/x/ganak``, ``sstd/sharpSAT`` (+ ``flow_cutter_pace17``),
``shim/timeout`` and an unpacked ``pysdd`` wheel; WORK receives DIMACS files
and JSON results. Run with the repository's Python environment.
"""
import os
import sys
from pathlib import Path

# One BLAS/OpenMP thread, matching torch.set_num_threads(1) in the workers.
SINGLE_THREAD = {name: "1" for name in (
    "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
os.environ.update(SINGLE_THREAD)

REPO = Path(__file__).resolve().parents[2]
TOOLS = Path(sys.argv[1]).resolve()
WORK = Path(sys.argv[2]).resolve()
ONLY = tuple(sys.argv[3:])
GANAK = str(TOOLS / "ganak/x/ganak")
SSTD_DIR = TOOLS / "sstd"
SSTD_ENV = dict(os.environ, PATH=f"{TOOLS / 'shim'}:{os.environ['PATH']}")
# SharpSAT-TD skips tree decomposition when -decot < 0.099; 0.02 is its fast default here.
SSTD_ARGS = ["./sharpSAT", "-decow", "100", "-tmpdir", "./tmp", "-cs", "3500"]


def selected(name):
    return not ONLY or any(name.startswith(prefix) for prefix in ONLY)
