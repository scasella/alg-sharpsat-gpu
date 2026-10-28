"""Weighted inference: CompiledGuardrail versus PySDD, GANAK, SharpSAT-TD and RC2.

Compile once and query many: the guardrail (default ``num_groups="auto"``)
and a PySDD SDD (minimisation on) answer one query and a 256-query batch.
Exact rational WMC is compared with one-shot GANAK runs; the cold column times
``compile_guardrail(cnf).evaluate(..., mode="exact_rational")`` end to end.
MAP is compared with PySAT RC2Stratified. Every float64 marginal (one query
and all 256 batch rows) is checked against exact rational marginals.
"""
import contextlib
import io
import json
import math
import re
import subprocess
import sys
import time
from fractions import Fraction
from statistics import median

from common import GANAK, REPO, SSTD_DIR, SSTD_ENV, TOOLS, WORK, selected

sys.path.insert(0, str(TOOLS / "pysdd"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from alg_sharpsat.benchmarks import bipartite_3sat, coupled_3sat  # noqa: E402
from alg_sharpsat.cnf import to_dimacs  # noqa: E402
from alg_sharpsat.domains import get_domains  # noqa: E402
from alg_sharpsat.stage3_gpu_engine import ResourceLimitError  # noqa: E402
from alg_sharpsat.stage5_wmc_guardrail import compile_guardrail  # noqa: E402
from pysdd.sdd import SddManager, WmcManager  # noqa: E402

OUT = WORK / "weighted.json"
WCNF = WORK / "wcnf"
WCNF.mkdir(parents=True, exist_ok=True)
RC2 = REPO / "scripts/external_bench/rc2_worker.py"
torch.set_num_threads(1)


def timed(fn, reps):
    out, samples = None, []
    for _ in range(reps):
        start = time.perf_counter()
        out = fn()
        samples.append(time.perf_counter() - start)
    return out, median(samples)


def instances():
    for family, generator, sizes in (("coupled", coupled_3sat, (18, 22, 26, 30, 40, 50, 60)),
                                     ("bipartite", bipartite_3sat, (18, 22, 26, 30))):
        for n in sizes:
            for seed in range(5):
                rng = np.random.default_rng(1000 * seed + n)
                k = rng.integers(50, 951, size=n)
                batch = rng.integers(50, 951, size=(256, n))
                yield f"{family}_n{n}_s{seed}", generator(n, seed=seed), n, k, batch
    for domain, pack in get_domains().items():
        names = pack.policy.variable_names
        preset_name = next(iter(pack.presets))
        preset = pack.presets[preset_name]
        k = np.array([int(Fraction(str(preset[name])) * 1000) for name in names])
        batch = np.random.default_rng(7).integers(50, 951, size=(256, len(names)))
        yield f"domain_{domain}_{preset_name}", pack.policy.to_cnf(), len(names), k, batch


def pairs_exact(k, n_all):
    return [(Fraction(1000 - int(x), 1000), Fraction(int(x), 1000)) for x in k] + \
        [(Fraction(1), Fraction(1))] * (n_all - len(k))


def pairs_float(k, n_all):
    p = np.asarray(k, dtype=np.float64) / 1000
    named = np.stack([1 - p, p], axis=-1)
    aux = np.ones(named.shape[:-2] + (n_all - named.shape[-2], 2))
    return np.concatenate([named, aux], axis=-2)


def write_wcnf(name, cnf, k):
    body = to_dimacs(cnf).splitlines()
    lines = ["c t wmc", body[0]]
    for i in range(1, cnf.num_vars + 1):
        if i <= len(k):
            p = Fraction(int(k[i - 1]), 1000)
            lines += [f"c p weight {i} {float(p)!r} 0", f"c p weight -{i} {float(1 - p)!r} 0"]
        else:
            lines += [f"c p weight {i} 1 0", f"c p weight -{i} 1 0"]
    path = WCNF / f"{name}.cnf"
    path.write_text("\n".join(lines + body[1:]) + "\n")
    return path


def external(cmd, cwd, env, pattern):
    walls, value = [], None
    for _ in range(3):
        start = time.perf_counter()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=cwd, env=env)
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        walls.append(time.perf_counter() - start)
        m = re.search(pattern, p.stdout, re.M)
        if not m:
            return {"status": "error", "reason": (p.stdout + p.stderr)[-200:]}
        value = m.group(1)
    return {"status": "ok", "value": value, "seconds": median(walls)}


def exact_marginals(result, n_named):
    return np.array([float(x) if x is not None else np.nan for x in result.posterior_marginals[:n_named]])


def run(name, cnf, n_named, k, batch):
    n_all = cnf.num_vars
    row = {"n_named": n_named, "n_vars": n_all, "clauses": len(cnf.clauses)}
    wx, wf, wb = pairs_exact(k, n_all), pairs_float(k, n_all), pairs_float(batch, n_all)
    try:
        compiled, row["repo_compile_s"] = timed(lambda: compile_guardrail(cnf), 3)
        plan = compiled.stats.plan_selection
        row["repo_plan"] = {"selected_num_groups": plan["selected_num_groups"], "mode": plan["mode"],
                            "cost": plan["cost"], "candidates": plan.get("candidates", [])}
        _, row["repo_cold_exact_s"] = timed(
            lambda: compile_guardrail(cnf).evaluate(weights=wx, mode="exact_rational"), 3)
        compiled.evaluate(weights=wf, mode="float64")
        r64, row["repo_q1_float64_s"] = timed(lambda: compiled.evaluate(weights=wf, mode="float64"), 30)
        compiled.evaluate(weights=wb, mode="float64")
        rb, row["repo_batch256_float64_s"] = timed(lambda: compiled.evaluate(weights=wb, mode="float64"), 5)
        if n_all == n_named:
            pb = np.asarray(batch, dtype=np.float64) / 1000
            compiled.evaluate(pb, mode="float64")
            _, row["repo_batch256_probs_float64_s"] = timed(lambda: compiled.evaluate(pb, mode="float64"), 5)
        rx, row["repo_q1_exact_s"] = timed(lambda: compiled.evaluate(weights=wx, mode="exact_rational"), 5)
        (_, wmap), row["repo_map_exact_s"] = timed(lambda: compiled.map_inference(weights=wx), 5)
        _, row["repo_sample1000_s"] = timed(
            lambda: compiled.sample_conditional(weights=wx, num_samples=1000, seed=7), 3)
        row["repo_Z_exact"] = str(rx.partition_function)
        row["repo_Z_float64"] = float(r64.partition_function)
        row["repo_map_weight"] = str(wmap)
        repo_marg = r64.posterior_marginals[:n_named].numpy()
        # Float64 accuracy against exact rational results, one query and all batch rows.
        row["float_q1_marg_max_abs_err"] = float(np.max(np.abs(repo_marg - exact_marginals(rx, n_named))))
        row["float_q1_Z_rel_err"] = abs(float(r64.partition_function) / float(rx.partition_function) - 1)
        errors, z_errors = [], []
        for j in range(len(batch)):
            exact = compiled.evaluate(weights=pairs_exact(batch[j], n_all), mode="exact_rational")
            errors.append(np.max(np.abs(rb.posterior_marginals[j, :n_named].numpy()
                                        - exact_marginals(exact, n_named))))
            z_errors.append(abs(float(rb.partition_function[j]) / float(exact.partition_function) - 1))
        row["float_batch_marg_max_abs_err"] = float(max(errors))
        row["float_batch_Z_max_rel_err"] = float(max(z_errors))
        row["repo_status"] = "ok"
    except ResourceLimitError as exc:
        row["repo_status"] = "resource_limit"
        row["repo_reason"] = str(exc)[:200]
        repo_marg = None
    text = to_dimacs(cnf)

    def sdd_compile():
        with contextlib.redirect_stdout(io.StringIO()):
            return SddManager.from_cnf_string(text)

    (manager, node), row["sdd_compile_s"] = timed(sdd_compile, 3)
    row["sdd_size"] = node.size()
    wmc = WmcManager(node, log_mode=True)

    def sdd_query(pairs):
        wmc.set_literal_weights_from_array(np.concatenate([np.log(pairs[::-1, 0]), np.log(pairs[:, 1])]))
        z = wmc.propagate()
        return z, [wmc.literal_pr(i) for i in range(1, n_named + 1)]

    sdd_query(wf)
    (log_z, log_marg), row["sdd_q1_s"] = timed(lambda: sdd_query(wf), 30)
    _, row["sdd_batch256_s"] = timed(lambda: [sdd_query(wb[j]) for j in range(256)], 5)
    row["sdd_Z"] = math.exp(log_z)
    if repo_marg is not None:
        row["sdd_vs_repo_Z_rel"] = abs(row["sdd_Z"] - row["repo_Z_float64"]) / row["repo_Z_float64"]
        row["sdd_vs_repo_marg_max_abs"] = float(np.max(np.abs(np.exp(log_marg) - repo_marg)))
    path = write_wcnf(name, cnf, k)
    exact_pattern = r"^c s exact arb frac (\S+)"
    row["ganak"] = external([GANAK, "--mode", "1", str(path)], None, None, exact_pattern)
    row["ganak_fast"] = external([GANAK, "--fast", "--mode", "1", str(path)], None, None, exact_pattern)
    row["sharpsat_td"] = external(["./sharpSAT", "-WE", "-decot", "0.02", "-decow", "100", "-tmpdir", "./tmp",
                                   "-cs", "3500", "-prec", "30", str(path)], SSTD_DIR, SSTD_ENV,
                                  r"^c s exact arb float (\S+)")
    if row["ganak"]["status"] == "ok" and row["repo_status"] == "ok":
        row["ganak_equals_repo_exact"] = Fraction(row["ganak"]["value"]) == Fraction(row["repo_Z_exact"])
    if row["sharpsat_td"]["status"] == "ok" and row["repo_status"] == "ok":
        row["sstd_vs_repo_rel"] = abs(float(row["sharpsat_td"]["value"]) - row["repo_Z_float64"]) / row["repo_Z_float64"]
    spec = WCNF / f"{name}.rc2.json"
    spec.write_text(json.dumps({"clauses": [[l.signed for l in c.literals] for c in cnf.clauses],
                                "k": [int(x) for x in k]}))
    samples, model = [], None
    for _ in range(3):
        try:
            p = subprocess.run([sys.executable, str(RC2), str(spec)], capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            row["rc2_status"] = "timeout_60s"
            break
        out = json.loads(p.stdout.strip().splitlines()[-1])
        model = out["model"]
        samples.append(out["seconds"])
    if samples:
        row["rc2_status"] = "ok"
        row["rc2_map_s"] = median(samples)
        weight = Fraction(1)
        for i in range(n_named):
            p_i = Fraction(int(k[i]), 1000)
            weight *= p_i if model[i] > 0 else 1 - p_i
        row["rc2_map_weight"] = str(weight)
        if row["repo_status"] == "ok":
            row["rc2_map_equals_repo"] = weight == Fraction(row["repo_map_weight"])
    return row


results = json.loads(OUT.read_text()) if OUT.exists() else {}
for name, cnf, n_named, k, batch in instances():
    if name in results or not selected(name):
        continue
    start = time.perf_counter()
    results[name] = run(name, cnf, n_named, k, batch)
    OUT.write_text(json.dumps(results, indent=1))
    r = results[name]
    print(name, r.get("repo_status"), "G", r.get("repo_plan", {}).get("selected_num_groups"),
          {x: round(r[x] * 1e3, 3) for x in r if x.endswith("_s")},
          "ganak", r["ganak"].get("seconds"), "checks", r.get("ganak_equals_repo_exact"),
          r.get("rc2_map_equals_repo"), "float err", r.get("float_batch_marg_max_abs_err"),
          f"{time.perf_counter() - start:.1f}s", flush=True)
