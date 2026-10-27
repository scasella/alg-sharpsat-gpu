"""Reproducible WMC guardrail benchmarks with an independent weighted DPLL.

Run ``python -m alg_sharpsat.guardrail_benchmarks --sizes 8 --repeats 1``
for a smoke test. All timings are wall-clock measurements on the reported host.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from functools import lru_cache
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .cnf import CNF, Clause, Literal


def weighted_dpll(cnf, probabilities=None, *, weights=None, exact=False):
    """Independent recursive DPLL, unit propagation, and n+1 WMC passes.

    No compiled guardrail internals or assignment enumeration are used.
    """
    cnf.validate()
    cast = (
        (lambda x: x if isinstance(x, Fraction) else Fraction(str(x)))
        if exact
        else float
    )
    if weights is None:
        probabilities = (
            [Fraction(1, 2)] * cnf.num_vars if probabilities is None else probabilities
        )
        pairs = tuple((1 - cast(p), cast(p)) for p in probabilities)
    else:
        pairs = tuple(tuple(cast(x) for x in pair) for pair in weights)
    if len(pairs) != cnf.num_vars or any(len(p) != 2 or min(p) < 0 for p in pairs):
        raise ValueError("expected n nonnegative weight pairs")
    canonical = lambda cs: tuple(sorted(set(tuple(sorted(set(c))) for c in cs)))
    clauses = canonical(
        tuple(l.signed for l in c.literals)
        for c in cnf.clauses
        if not any(-l.signed in {x.signed for x in c.literals} for l in c.literals)
    )

    def count(extra=()):
        @lru_cache(None)
        def solve(current, remaining):
            factor = cast(1)
            while current:
                if any(not c for c in current):
                    return cast(0)
                unit = next((c[0] for c in current if len(c) == 1), None)
                if unit is None:
                    break
                v = abs(unit) - 1
                factor *= pairs[v][unit > 0]
                remaining &= ~(1 << v)
                current = canonical(
                    tuple(x for x in c if x != -unit) for c in current if unit not in c
                )
            if not current:
                for v in range(cnf.num_vars):
                    if remaining & (1 << v):
                        factor *= sum(pairs[v])
                return factor
            freq = Counter(abs(x) for c in current for x in c)
            v = min(freq, key=lambda x: (-freq[x], x))
            value = cast(0)
            for lit in (v, -v):
                reduced = canonical(
                    tuple(x for x in c if x != -lit) for c in current if lit not in c
                )
                value += pairs[v - 1][lit > 0] * solve(
                    reduced, remaining & ~(1 << (v - 1))
                )
            return factor * value

        return solve(canonical((*clauses, *extra)), (1 << cnf.num_vars) - 1)

    z = count()
    marginals = [
        count(((v + 1,),)) / z if z else float("nan") for v in range(cnf.num_vars)
    ]
    return z, marginals


def make_family(n, family):
    """Use the same seeded planted families as the repository v3 benchmark."""
    from .benchmarks import coupled_3sat, bipartite_3sat

    return {"coupled": coupled_3sat, "bipartite": bipartite_3sat}[family](n, seed=0)


def timed(fn, repeats):
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return value, {"seconds": samples, "median_seconds": float(np.median(samples))}


dpll_wmc_and_marginals = weighted_dpll


def as_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def check(result, baseline):
    z, marginals = baseline
    assert np.isclose(
        float(result.partition_function), float(z), rtol=2e-10, atol=1e-14
    )
    assert np.allclose(
        np.asarray(as_numpy(result.posterior_marginals), dtype=float),
        np.asarray(marginals, dtype=float),
        rtol=2e-9,
        atol=1e-11,
    )


def valid(cnf, assignments):
    arr = np.asarray(assignments, dtype=bool)
    if arr.ndim == 1:
        arr = arr[None]
    return all(
        any(bool(row[l.var]) == l.pos for l in c.literals)
        for row in arr
        for c in cnf.clauses
    )


def run(args):
    from .stage5_wmc_guardrail import compile_guardrail

    host = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    report = {
        "schema": "benchmark-v3.2-guardrail",
        "host": host,
        "configuration": {**vars(args), "family_seed": 0, "planted": True},
        "cases": [],
    }
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        report["configuration"]["compile_and_evaluate_torch_threads"] = 1
        for family in args.family:
            for n in args.sizes:
                cnf = make_family(n, family)
                probs = [Fraction(2 + i % 5, 8) for i in range(n)]
                torch.set_num_threads(1)
                numeric_probs = np.asarray(probs, dtype=np.float64)
                baseline, bt = timed(
                    lambda: weighted_dpll(cnf, numeric_probs), args.repeats
                )
                exact_base, et = timed(
                    lambda: weighted_dpll(cnf, probs, exact=True), args.repeats
                )
                compiled, ct = timed(lambda: compile_guardrail(cnf), args.repeats)
                count_weights = [[1, 1] for _ in range(n)]
                reference_count, _ = weighted_dpll(
                    cnf, weights=count_weights, exact=True
                )
                count_result = compiled.evaluate(
                    weights=count_weights, mode="exact_rational"
                )
                assert count_result.partition_function == reference_count
                case = {
                    "model_count": str(reference_count),
                    "family": family,
                    "num_vars": n,
                    "num_clauses": len(cnf.clauses),
                    "baseline_float64": bt,
                    "baseline_exact_rational": et,
                    "compile": ct,
                    "compile_telemetry": dict(vars(compiled.stats)),
                    "raw_sparse_stats": dict(vars(compiled.raw_sparse_stats)),
                    "retained_signature_counts": [
                        node.key_count for node in compiled.nodes
                    ],
                    "evaluations": {},
                }
                for mode in ("float64", "exact_rational"):
                    mode_inputs = probs if mode == "exact_rational" else numeric_probs
                    result, timing = timed(
                        lambda: compiled.evaluate(mode_inputs, mode=mode), args.repeats
                    )
                    _, end_to_end = timed(
                        lambda: compile_guardrail(cnf).evaluate(mode_inputs, mode=mode),
                        args.repeats,
                    )
                    check(result, exact_base)
                    if mode == "exact_rational":
                        assert result.partition_function == exact_base[0]
                        assert list(result.posterior_marginals) == exact_base[1]
                        case["exact_arithmetic"] = dict(compiled.last_evaluation)
                    case["evaluations"][mode] = {
                        **timing,
                        "partition_function": str(result.partition_function),
                    }
                    case["evaluations"][mode]["compile_plus_evaluate"] = end_to_end
                assert np.isclose(float(baseline[0]), float(exact_base[0]), rtol=2e-10)
                assert np.allclose(
                    np.asarray(baseline[1], float), np.asarray(exact_base[1], float)
                )
                batch = np.random.default_rng(1729 + n).uniform(
                    0.1, 0.9, (args.batch_size, n)
                )
                singles = [compiled.evaluate(row, mode="float64") for row in batch]
                reference_z = np.asarray([float(r.partition_function) for r in singles])
                reference_marginals = np.asarray(
                    [as_numpy(r.posterior_marginals) for r in singles]
                )
                for row, single_result in zip(batch[:3], singles[:3]):
                    check(single_result, weighted_dpll(cnf, row))
                float32_result = compiled.evaluate(batch, mode="float32", device="cpu")
                assert np.allclose(
                    as_numpy(float32_result.partition_function),
                    reference_z,
                    rtol=2e-4,
                    atol=1e-10,
                )
                assert np.allclose(
                    as_numpy(float32_result.posterior_marginals),
                    reference_marginals,
                    rtol=2e-4,
                    atol=2e-5,
                )
                case["cpu_float32_batch_validated"] = True
                case["throughput"] = {}
                for threads in sorted({1, min(4, os.cpu_count() or 1)}):
                    torch.set_num_threads(threads)
                    _, single = timed(
                        lambda: compiled.evaluate(numeric_probs, mode="float64"),
                        args.repeats,
                    )
                    result, batch_time = timed(
                        lambda: compiled.evaluate(batch, mode="float64"), args.repeats
                    )
                    assert np.allclose(
                        np.asarray(as_numpy(result.partition_function), float),
                        reference_z,
                        rtol=2e-10,
                    )
                    assert np.allclose(
                        np.asarray(as_numpy(result.posterior_marginals), float),
                        reference_marginals,
                        rtol=2e-9,
                        atol=1e-11,
                    )
                    case["throughput"][f"cpu_{threads}_threads"] = {
                        "single": single,
                        "batch": batch_time,
                        "batch_size": args.batch_size,
                        "distinct_vectors": True,
                        "validation": "all batch rows versus singles; first three versus independent DPLL",
                        "torch_num_threads": threads,
                        "backend": "torch CPU log-scatter contraction",
                        "vectors_per_second": args.batch_size
                        / batch_time["median_seconds"],
                    }
                case["accelerators"] = {}
                for accelerator in ("mps", "cuda"):
                    available = host[f"{accelerator}_available"]
                    status = {
                        "status": "skipped"
                        if args.skip_accelerators
                        else "unavailable",
                        "reason": f"{accelerator.upper()} unavailable on host"
                        if not available
                        else "disabled by CLI",
                    }
                    if not args.skip_accelerators and available:
                        try:
                            result, mt = timed(
                                lambda: compiled.evaluate(
                                    batch, mode="float32", device=accelerator
                                ),
                                args.repeats,
                            )
                            assert np.allclose(
                                as_numpy(result.partition_function),
                                reference_z,
                                rtol=2e-4,
                                atol=1e-10,
                            )
                            assert np.allclose(
                                as_numpy(result.posterior_marginals),
                                reference_marginals,
                                rtol=2e-4,
                                atol=2e-5,
                            )
                            status = {
                                "status": "available",
                                "mode": "float32",
                                "device": accelerator,
                                **mt,
                            }
                        except (RuntimeError, ValueError, NotImplementedError) as exc:
                            status = {"status": "unsupported", "reason": str(exc)}
                    case["accelerators"][accelerator] = status
                case["accelerator"] = case["accelerators"]["mps"]
                report["cases"].append(case)
                print(
                    f"{family} n={n}: compiled and independent DPLL agree", flush=True
                )
        rare = CNF(20, [Clause((Literal(i, True),)) for i in range(20)])
        probs = [Fraction(1, 2)] * 20
        compiled, compile_time = timed(lambda: compile_guardrail(rare), 1)
        result, eval_time = timed(
            lambda: compiled.evaluate(probs, mode="exact_rational"), args.repeats
        )
        check(result, weighted_dpll(rare, probs, exact=True))
        samples, sample_time = timed(
            lambda: compiled.sample_conditional(probs, num_samples=1000, seed=1729), 1
        )
        assert valid(rare, samples)
        repaired, map_time = timed(lambda: compiled.map_inference(probs), 1)
        assignment = (
            repaired[0]
            if isinstance(repaired, tuple)
            else getattr(repaired, "assignment", repaired)
        )
        if isinstance(assignment, dict):
            assignment = assignment["assignment"]
        assert valid(rare, assignment)
        rng = np.random.default_rng(1729)
        start = time.perf_counter()
        proposals = rng.random((10000, 20)) < 0.5
        accepted = int(np.all(proposals, axis=1).sum())
        rejection_simulation_seconds = time.perf_counter() - start
        report["rare_event"] = {
            "num_vars": 20,
            "acceptance_probability": float(result.acceptance_prob),
            "expected_rejection_attempts_per_sample": 1 / float(result.acceptance_prob),
            "expected_rejection_attempts_for_1000_samples": 1000
            / float(result.acceptance_prob),
            "bounded_rejection_simulation": {
                "attempts": len(proposals),
                "accepted": accepted,
                "seed": 1729,
                "seconds": rejection_simulation_seconds,
            },
            "compile": compile_time,
            "evaluate": eval_time,
            "map_repair": map_time,
            "conditional_samples": {
                **sample_time,
                "count": 1000,
                "mode": "exact_rational",
                "all_valid": True,
            },
            "posterior_marginals": list(map(float, result.posterior_marginals)),
        }
    finally:
        torch.set_num_threads(previous_threads)
    Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
    print(
        "family     n  compile+float64(s)  single(s)  batch(s)  vectors/s [CPU 1 thread]"
    )
    for case in report["cases"]:
        cpu = case["throughput"]["cpu_1_threads"]
        print(
            f"{case['family']:10s} {case['num_vars']:2d}  {case['evaluations']['float64']['compile_plus_evaluate']['median_seconds']:.6f}  {cpu['single']['median_seconds']:.6f}  {cpu['batch']['median_seconds']:.6f}  {cpu['vectors_per_second']:.1f}"
        )
    print(
        f"rare acceptance={report['rare_event']['acceptance_probability']:.9g}; 1000 exact samples={report['rare_event']['conditional_samples']['median_seconds']:.6f}s"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[18, 22, 26, 30])
    parser.add_argument(
        "--family",
        nargs="+",
        choices=["coupled", "bipartite"],
        default=["coupled", "bipartite"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--json", default="benchmark-v3.2-guardrail.json")
    parser.add_argument("--skip-accelerators", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.batch_size < 1:
        parser.error("repeats and batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    main()
