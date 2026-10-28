"""Markdown tables and gate checks from WORK/counting.json and WORK/weighted.json.

Usage: ``python final_tables.py WORK [PREVIOUS_WORK]``. Cells are medians over
seeds 0-4. External solvers are subprocess wall times; "- launch" subtracts the
same tool's median time on a three-variable formula. PREVIOUS_WORK, when given,
adds repository-only medians from an earlier run of the same harness.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

WORK = Path(sys.argv[1])
PREVIOUS = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def ms(x):
    if x is None:
        return "—"
    x *= 1e3
    return f"{x:.3f}" if x < 0.1 else f"{x:.2f}" if x < 10 else f"{x:.1f}" if x < 1000 else f"{x:,.0f}"


def family_size(name):
    m = re.match(r"(\w+?)_n(\d+)_s\d+", name)
    return (m.group(1), int(m.group(2))) if m else None


counting = json.loads((WORK / "counting.json").read_text())
floor = {k: v["seconds"] for k, v in counting.pop("_startup_floor").items()}
groups = defaultdict(list)
for name, row in counting.items():
    groups[family_size(name)].append(row)


def cell(rows, key):
    ok = [r[key]["seconds"] for r in rows if r.get(key, {}).get("status") == "ok"]
    failed = [r[key]["status"] for r in rows if r.get(key, {}).get("status") not in (None, "ok", "skipped")]
    if not ok:
        return "limit" if failed else "—"
    return ms(median(ok)) + (f" ({len(failed)}/{len(rows)} limit)" if failed else "")


def internal_times(row):
    """Per-tool solve-time estimates used to pick the fastest tool per seed."""
    out = {}
    for key, base in (("ganak", "ganak"), ("ganak_fast", "ganak_fast"), ("sharpsat_td", "sharpsat_td"),
                      ("sharpsat_td_decot0.1", "sharpsat_td"), ("sharpsat_td_decot1", "sharpsat_td")):
        v = row.get(key, {})
        if v.get("status") != "ok":
            continue
        if key.startswith("sharpsat") and v.get("internal_seconds") is not None:
            out[key] = v["internal_seconds"]
        elif v["seconds"] > 2 * floor[base]:
            # GANAK prints its timer to 2 decimals; wall minus launch is only
            # meaningful well above the launch floor.
            out[key] = v["seconds"] - floor[base]
    return out


print("launch floor ms:", {k: ms(v) for k, v in floor.items()})
print("\n| family | n | repo contraction | repo auto | auto engine | repo DPLL | GANAK | GANAK --fast "
      "| SharpSAT-TD | SSTD internal | best external − launch | fastest (per seed) | agree |")
print("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|")
for (family, n), rows in sorted(groups.items()):
    engines = sorted({r["auto"].get("selected_engine") for r in rows if r["auto"].get("status") == "ok"})
    internal = [r["sharpsat_td"]["internal_seconds"] for r in rows
                if r["sharpsat_td"].get("internal_seconds") is not None]
    best, fastest = [], defaultdict(int)
    for r in rows:
        launched = [r[k]["seconds"] - floor[b] for k, b in (("ganak", "ganak"), ("ganak_fast", "ganak_fast"),
                                                             ("sharpsat_td", "sharpsat_td"))
                    if r[k].get("status") == "ok"]
        best.append(max(min(launched), 0.0))
        tools = internal_times(r)
        repo = r["auto"]["seconds"] if r["auto"].get("status") == "ok" else None
        winner = min(tools, key=tools.get)
        fastest["repo" if repo is not None and repo < tools[winner] else winner.split("_decot")[0]] += 1
    win = ", ".join(f"{k} {v}" for k, v in sorted(fastest.items(), key=lambda kv: -kv[1]))
    print(f"| {family} | {n} | {cell(rows, 'sparse')} | {cell(rows, 'auto')} | {', '.join(engines) or '—'} "
          f"| {cell(rows, 'dpll')} | {cell(rows, 'ganak')} | {cell(rows, 'ganak_fast')} "
          f"| {cell(rows, 'sharpsat_td')} | {ms(median(internal)) if internal else '—'} | {ms(median(best))} "
          f"| {win} | {sum(r['agree'] for r in rows)}/{len(rows)} |")

weighted = json.loads((WORK / "weighted.json").read_text())
wgroups = defaultdict(list)
for name, row in weighted.items():
    wgroups[family_size(name) or (name.replace("domain_", ""), row["n_named"])].append(row)


def ganak_exact(r):
    if r["ganak"].get("status") == "ok" and r["ganak_fast"].get("status") == "ok":
        return min(r["ganak"]["seconds"] - floor["ganak"], r["ganak_fast"]["seconds"] - floor["ganak_fast"])
    return None


COLUMNS = [("repo_compile_s", "repo compile"), ("sdd_compile_s", "SDD compile"),
           ("repo_q1_float64_s", "repo query"), ("sdd_q1_s", "SDD query"),
           ("repo_batch256_float64_s", "repo 256 batch"), ("sdd_batch256_s", "SDD 256 loop"),
           ("repo_q1_exact_s", "repo exact query"), ("repo_cold_exact_s", "repo cold exact (compile+query)"),
           (ganak_exact, "GANAK exact (one-shot, −launch)"), ("repo_map_exact_s", "repo MAP"),
           ("rc2_map_s", "RC2 MAP"), ("repo_sample1000_s", "repo 1k samples")]
print("\nTimes in ms.\n")
print("| instance | n | plan groups | " + " | ".join(h for _, h in COLUMNS) + " |")
print("|---|---:|---|" + "---:|" * len(COLUMNS))
for key, rows in sorted(wgroups.items(), key=lambda kv: (len(kv[1]) == 1, kv[0])):
    plans = sorted({r["repo_plan"]["selected_num_groups"] for r in rows if "repo_plan" in r})
    cells = []
    for column, _ in COLUMNS:
        values = [column(r) if callable(column) else r.get(column) for r in rows]
        values = [v for v in values if v is not None]
        cells.append(ms(median(values)) if values else "—")
    print(f"| {key[0]} | {key[1]} | {', '.join(map(str, plans))} | " + " | ".join(cells) + " |")

checks = defaultdict(lambda: [0, 0])
for r in weighted.values():
    for k in ("ganak_equals_repo_exact", "rc2_map_equals_repo"):
        if k in r:
            checks[k][0] += bool(r[k])
            checks[k][1] += 1
    checks["repo_ok"][0] += r.get("repo_status") == "ok"
    checks["repo_ok"][1] += 1
ok_rows = [r for r in weighted.values() if r.get("repo_status") == "ok"]
float_error = max(max(r["float_q1_marg_max_abs_err"], r["float_batch_marg_max_abs_err"]) for r in ok_rows)
z_error = max(max(r["float_q1_Z_rel_err"], r["float_batch_Z_max_rel_err"]) for r in ok_rows)
print("\nchecks:", {k: f"{a}/{b}" for k, (a, b) in checks.items()},
      "| counts agree %d/%d" % (sum(r["agree"] for r in counting.values()), len(counting)),
      "| float64 vs exact: max marginal abs err %.1e, max Z rel err %.1e" % (float_error, z_error),
      "| max SDD Z rel err %.1e" % max(r["sdd_vs_repo_Z_rel"] for r in ok_rows))

print("\nGate checks:")
domains = {name: r for name, r in weighted.items() if name.startswith("domain_")}
for name, r in sorted(domains.items()):
    print(f"- {name}: batch {ms(r['repo_batch256_float64_s'])} ms (<1.5: "
          f"{r['repo_batch256_float64_s'] < 1.5e-3}; SDD loop {ms(r['sdd_batch256_s'])}), "
          f"MAP {ms(r['repo_map_exact_s'])} ms (<0.20: {r['repo_map_exact_s'] < 2e-4}; RC2 {ms(r['rc2_map_s'])}), "
          f"query {ms(r['repo_q1_float64_s'])} ms (<0.03: {r['repo_q1_float64_s'] < 3e-5})")
for n in (100, 150, 200, 300):
    rows = groups.get(("coupled", n), [])
    status = [r["auto"].get("status") for r in rows]
    times = [r["auto"]["seconds"] for r in rows if r["auto"].get("status") == "ok"]
    print(f"- coupled n={n}: auto finished {status.count('ok')}/{len(rows)}; "
          f"per-seed ms {[round(t * 1e3, 1) for t in times]}; median {ms(median(times)) if times else '—'}")

if PREVIOUS:
    old_c = json.loads((PREVIOUS / "counting.json").read_text())
    old_w = json.loads((PREVIOUS / "weighted.json").read_text())
    print("\nRepository medians, previous run → this run (ms):\n")
    print("| rows | previous repo sparse | this run repo auto |")
    print("|---|---:|---:|")
    old_groups = defaultdict(list)
    for name, row in old_c.items():
        if not name.startswith("_"):
            old_groups[family_size(name)].append(row)
    for key in sorted(k for k in groups if k[0] == "coupled" and k[1] >= 50):
        print(f"| coupled n={key[1]} | {cell(old_groups[key], 'sparse')} | {cell(groups[key], 'auto')} |")
    print("\n| instance | metric | previous | this run |")
    print("|---|---|---:|---:|")
    for name in sorted(domains):
        for column, label in (("repo_compile_s", "compile"), ("repo_q1_float64_s", "query"),
                              ("repo_batch256_float64_s", "256 batch"), ("repo_q1_exact_s", "exact query"),
                              ("repo_map_exact_s", "MAP")):
            before = old_w.get(name, {}).get(column)
            print(f"| {name.replace('domain_', '')} | {label} | {ms(before)} | {ms(domains[name][column])} |")
