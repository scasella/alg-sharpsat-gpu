"""Unweighted #SAT: repository counters versus GANAK and SharpSAT-TD.

Each instance runs the repository's contraction, auto-selected and DPLL
counters (in-process, warm, three repeats) and the external solvers as
subprocesses (wall time, three repeats, 300 s limit). SharpSAT-TD also runs
with tree decomposition enabled on the larger instances. Results are written
to WORK/counting.json after every instance, so an interrupted run resumes.
With REFRESH_REPO=1, existing rows keep their external measurements and only
the repository's contraction and auto columns are measured again.
"""
import json
import os
import re
import subprocess
import sys
import time
from statistics import median

from common import GANAK, REPO, SSTD_ARGS, SSTD_DIR, SSTD_ENV, WORK, selected

REPEATS, TIMEOUT = 3, 300
OUT = WORK / "counting.json"
WORKER = REPO / "scripts/external_bench/repo_worker.py"


def repo(engine, path, timeout=TIMEOUT):
    try:
        p = subprocess.run([sys.executable, str(WORKER), engine, str(path), str(REPEATS)],
                           capture_output=True, text=True, timeout=timeout, cwd=REPO)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if p.returncode:
        return {"status": "error", "reason": p.stderr.strip().splitlines()[-1][:200]}
    return json.loads(p.stdout.strip().splitlines()[-1])


def external(cmd, cwd=None, env=None, internal=None, repeats=REPEATS, timeout=TIMEOUT):
    walls, inner, count, out = [], [], None, ""
    for _ in range(repeats):
        start = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        walls.append(time.perf_counter() - start)
        out = p.stdout
        m = re.search(r"^c s exact arb int (\d+)", out, re.M)
        if not m:
            return {"status": "error", "reason": (p.stdout + p.stderr)[-300:]}
        assert count is None or m.group(1) == count
        count = m.group(1)
        inner.append(internal(out) if internal else None)
    row = {"status": "ok", "count": count, "seconds": median(walls), "samples": walls,
           "internal_seconds": median(inner) if None not in inner else None}
    width = re.search(r"^c o width (\d+)", out, re.M)
    if "sharpSAT" in cmd[0]:
        row["td_ran"] = "tw finish ok" in out
        row["td_width"] = int(width.group(1)) if width else None
    return row


def ganak_internal(out):
    m = re.search(r"Total time \[Arjun\+GANAK\]: ([\d.]+)", out)
    return float(m.group(1)) if m else None


def sstd_internal(out):
    # "Solved in" runs from program start, so it includes parsing and setup.
    m = re.search(r"Solved in ([\d.e-]+) seconds", out)
    return float(m.group(1)) if m else None


def sstd(path, decot, repeats=REPEATS, timeout=TIMEOUT):
    return external([*SSTD_ARGS, "-decot", decot, str(path)], SSTD_DIR, SSTD_ENV, sstd_internal,
                    repeats, timeout)


def large(name):
    family, n = re.match(r"(\w+?)_n(\d+)_", name).groups()
    return int(n) >= {"coupled": 40, "bipartite": 30, "random": 30}[family]


files = sorted((WORK / "cnf").glob("*.cnf"),
               key=lambda f: (f.name.split("_")[0], int(f.name.split("_n")[1].split("_")[0]), f.name))
results = json.loads(OUT.read_text()) if OUT.exists() else {}
if os.environ.get("REFRESH_REPO") == "1":
    for f in files:
        row = results.get(f.name)
        if row is None or not selected(f.name):
            continue
        row["sparse"], row["auto"] = repo("sparse", f), repo("auto", f)
        row["repo_refreshed"] = True
        counts = {k: v["count"] for k, v in row.items() if isinstance(v, dict) and v.get("status") == "ok"}
        row["agree"] = len(set(counts.values())) <= 1
        OUT.write_text(json.dumps(results, indent=1))
        print(f.name, row["sparse"].get("seconds", row["sparse"]["status"]),
              row["auto"].get("seconds", row["auto"]["status"]), "agree" if row["agree"] else "MISMATCH", flush=True)
    sys.exit(0)
tiny = WORK / "tiny.cnf"
results["_startup_floor"] = {
    "ganak": external([GANAK, str(tiny)], internal=ganak_internal),
    "ganak_fast": external([GANAK, "--fast", str(tiny)]),
    "sharpsat_td": sstd(tiny, "0.02"),
}
for f in files:
    if not selected(f.name) or f.name in results:
        continue
    n = int(f.name.split("_n")[1].split("_")[0])
    row = {
        "sparse": repo("sparse", f),
        "auto": repo("auto", f),
        "dpll": repo("dpll", f, 120) if n <= 60 else {"status": "skipped"},
        "ganak": external([GANAK, str(f)], internal=ganak_internal),
        "ganak_fast": external([GANAK, "--fast", str(f)]),
        "sharpsat_td": sstd(f, "0.02"),
    }
    if large(f.name):
        row["sharpsat_td_decot0.1"] = sstd(f, "0.1")
        row["sharpsat_td_decot1"] = sstd(f, "1", repeats=1, timeout=600)
    counts = {k: v["count"] for k, v in row.items() if v.get("status") == "ok"}
    row["agree"] = len(set(counts.values())) <= 1
    results[f.name] = row
    OUT.write_text(json.dumps(results, indent=1))
    fmt = lambda v: f"{v['seconds'] * 1e3:.1f}ms" if v.get("status") == "ok" else v["status"]
    print(f.name, {k: fmt(v) for k, v in row.items() if k != "agree"},
          row["auto"].get("selected_engine"), "agree" if row["agree"] else "MISMATCH", flush=True)
