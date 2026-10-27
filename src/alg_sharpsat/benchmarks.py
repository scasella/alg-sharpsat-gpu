"""Reproducible exact-count benchmarks with explicit resource budgets.

The default family has bounded cross-group interfaces. It is a tractable
demonstration, not evidence of an asymptotic or general-purpose speedup.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import json
from math import prod
from pathlib import Path
import platform
import random
from statistics import median
import sys
from time import perf_counter

import torch

from .cnf import CNF, Clause, Literal, dpll_count
from .stage1_sparsify import (partition_and_decouple_boundary,
                              reduce_clause_width_deterministic, split_high_degree_variables)
from .stage3_gpu_engine import ResourceLimitError, UnsupportedDeviceError, count_with_stats


def _validate_size(num_vars: int, minimum: int = 3) -> None:
    if type(num_vars) is not int or num_vars < minimum:
        raise ValueError(f"num_vars must be an integer >= {minimum}")


def _clause(variables, rng: random.Random, planted: bool) -> Clause:
    signs = [bool(rng.getrandbits(1)) for _ in variables]
    # The all-true assignment is the reproducible planted witness.
    if planted and not any(signs):
        signs[rng.randrange(len(signs))] = True
    return Clause(tuple(Literal(var, sign) for var, sign in zip(variables, signs)))


def random_3sat(num_vars: int, num_clauses: int | None = None,
                seed: int = 0, planted: bool = True) -> CNF:
    """Generate random 3-SAT with distinct variables in each clause."""
    _validate_size(num_vars)
    num_clauses = 4 * num_vars if num_clauses is None else num_clauses
    if type(num_clauses) is not int or num_clauses < 0:
        raise ValueError("num_clauses must be a nonnegative integer")
    rng = random.Random(seed)
    return CNF(num_vars, [_clause(rng.sample(range(num_vars), 3), rng, planted)
                          for _ in range(num_clauses)])


def _cross_variables(left, right, rng):
    first, second = (left, right) if rng.getrandbits(1) else (right, left)
    return [*rng.sample(first, 2), rng.choice(second)]


def coupled_3sat(num_vars: int, seed: int = 0, bridges: int = 4,
                 planted: bool = True) -> CNF:
    """Dense internal halves (4*n clauses total), joined by bounded bridges."""
    _validate_size(num_vars, 6)
    if type(bridges) is not int or bridges < 0:
        raise ValueError("bridges must be a nonnegative integer")
    rng = random.Random(seed)
    midpoint = num_vars // 2
    left, right = list(range(midpoint)), list(range(midpoint, num_vars))
    clauses = [_clause(rng.sample(group, 3), rng, planted)
               for group in (left, right) for _ in range(4 * len(group))]
    clauses += [_clause(_cross_variables(left, right, rng), rng, planted)
                for _ in range(bridges)]
    return CNF(num_vars, clauses)


def bipartite_3sat(num_vars: int, num_clauses: int | None = None,
                   seed: int = 0, planted: bool = True) -> CNF:
    """Stress family: every random clause crosses the two variable halves."""
    _validate_size(num_vars, 4)
    num_clauses = 4 * num_vars if num_clauses is None else num_clauses
    if type(num_clauses) is not int or num_clauses < 0:
        raise ValueError("num_clauses must be a nonnegative integer")
    rng = random.Random(seed)
    midpoint = num_vars // 2
    left, right = list(range(midpoint)), list(range(midpoint, num_vars))
    return CNF(num_vars, [_clause(_cross_variables(left, right, rng), rng, planted)
                          for _ in range(num_clauses)])



def _stats_dict(raw_stats) -> dict:
    return asdict(raw_stats) if is_dataclass(raw_stats) else dict(raw_stats)


def _jet_metrics(stats: dict) -> dict:
    """Require actual per-node subset work; never substitute nominal 2**B."""
    evaluations = stats["node_evaluations"]
    return {
        "hermite_node_evaluations": evaluations,
        "hermite_subset_conditioning_states_sum": sum(
            item["conditioning_states_sum"] for item in evaluations
        ),
        "hermite_subset_conditioning_states_max": max(
            (item["conditioning_states_max"] for item in evaluations), default=0
        ),
        "hermite_processed_subsets": sum(item["processed_subsets"] for item in evaluations),
        "hermite_full_set_evaluations": sum(bool(item["full_set_processed"]) for item in evaluations),
    }


def _synchronize(device: str) -> None:
    """Include accelerator work in an end-to-end wall-clock sample."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "auto":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()


def _measure_engine(call, *, repeats: int, device: str, reference: int | None) -> dict:
    samples, counts, stats_samples = [], [], []
    for _ in range(repeats):
        _synchronize(device)
        start = perf_counter()
        try:
            value, raw_stats = call()
            stats = _stats_dict(raw_stats)
            _synchronize(stats.get("device", device))
        except (ResourceLimitError, UnsupportedDeviceError) as exc:
            _synchronize(device)
            return {"status": ("resource_limit" if isinstance(exc, ResourceLimitError)
                               else "unsupported_device"),
                    "reason": str(exc), "seconds": None,
                    "failed_attempt_seconds": perf_counter() - start,
                    "samples_seconds": samples, "completed_repeats": len(samples),
                    "count": None, "verified": False, "stats": None}
        samples.append(perf_counter() - start)
        if reference is not None and value != reference:
            raise AssertionError(f"engine count {value} differs from DPLL count {reference}")
        if counts and value != counts[0]:
            raise AssertionError("engine counts differ between repeated samples")
        counts.append(value)
        stats_samples.append(stats)
    representative = min(range(len(samples)), key=lambda i: abs(samples[i] - median(samples)))
    return {"status": "ok", "reason": None, "seconds": median(samples),
            "failed_attempt_seconds": None, "samples_seconds": samples,
            "completed_repeats": len(samples), "count": counts[0],
            "verified": reference is not None, "stats": stats_samples[representative]}


def run_v3_benchmark(num_vars: int, *, family: str = "coupled", seed: int = 0,
                     bridges: int = 4, planted: bool = True, device: str = "auto",
                     num_groups: int = 2, block_size: int = 4,
                     torch_threads: int = 1, repeats: int = 1,
                     jet_nodes: int = 6, jet_mode: str = "hermite",
                     jet_orders: tuple[int, ...] | None = None,
                     max_subset_terms: int = 1_000_000,
                     max_jet_work: int | None = None,
                     max_join_pairs: int = 8_000_000,
                     semi_join_pushdown: bool = True,
                     swap_max_work: int = 2_000_000, **engine_options) -> dict:
    """Compare four exact counters; preserve independent resource failures.

    Every timing sample includes preprocessing, local table construction and
    counting. The miniature SWAP reference is a separate experiment, never a
    substitute timing for the input size's combinatorial cost estimate.
    """
    from .stage3_jet_engine import jet_count_with_stats
    from .stage3_sparse_contract import sparse_count_with_stats
    from .stage4_swap_mobius import occurrence_cost_estimate, swap_mobius_count_with_stats

    for name, value in (("repeats", repeats), ("torch_threads", torch_threads)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    generators = {"coupled": coupled_3sat, "random": random_3sat, "bipartite": bipartite_3sat}
    if family not in generators:
        raise ValueError(f"unknown family: {family}")
    generator_options = {"seed": seed, "planted": planted}
    if family == "coupled":
        generator_options["bridges"] = bridges
    cnf = generators[family](num_vars, **generator_options)
    common = dict(device=device, num_groups=num_groups, block_size=block_size, **engine_options)
    jet_work = max_jet_work if max_jet_work is not None else engine_options.get("max_work", 2_000_000_000)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads)
    try:
        reference_run = _measure_engine(lambda: (dpll_count(cnf), {}), repeats=repeats,
                                        device="cpu", reference=None)
        reference = reference_run["count"]
        reference_run["verified"] = reference is not None
        runs = {"dpll": reference_run}
        calls = {
            "lagrange": lambda: count_with_stats(cnf, **common),
            "hermite": lambda: jet_count_with_stats(
                cnf, **common, num_nodes=jet_nodes, mode=jet_mode, orders=jet_orders,
                max_subset_terms=max_subset_terms, max_jet_work=jet_work),
            "sparse_contract": lambda: sparse_count_with_stats(
                cnf, **common, max_join_pairs=max_join_pairs,
                semi_join_pushdown=semi_join_pushdown),
        }
        for name, call in calls.items():
            runs[name] = _measure_engine(call, repeats=repeats, device=device, reference=reference)
        swap = {"cost_estimate": occurrence_cost_estimate(cnf),
                "runtime_scope": "miniature input only (n<=6); no extrapolated runtime",
                "runtime": {"status": "not_run", "reason": "input exceeds miniature n<=6 limit",
                            "seconds": None, "count": None, "verified": False}}
        if num_vars <= 6:
            swap["runtime"] = _measure_engine(
                lambda: swap_mobius_count_with_stats(cnf, max_work=swap_max_work),
                repeats=repeats, device="cpu", reference=reference)
    finally:
        torch.set_num_threads(previous_threads)
    successful = {name: run for name, run in runs.items() if run["status"] == "ok"}
    sparse = runs["sparse_contract"]["stats"] or {}
    return {
        "engine": "v3", "family": family, "seed": seed, "planted": planted,
        "n": num_vars, "m": len(cnf.clauses), "count": reference, "reference_count": reference,
        "repeats": repeats, "timing_statistic": "median",
        "timing_scope": "end-to-end: preprocessing, build, contraction; accelerator synchronized",
        "torch_threads": torch_threads, "device": sparse.get("device", device),
        "partition_configuration": {"num_groups": num_groups, "block_size": block_size},
        "jet_configuration": {"num_nodes": jet_nodes, "mode": jet_mode, "orders": jet_orders,
                              "max_subset_terms": max_subset_terms, "max_jet_work": jet_work},
        "engine_options": dict(engine_options, max_join_pairs=max_join_pairs,
                               semi_join_pushdown=semi_join_pushdown),
        "engine_runs": runs,
        "engine_status": {name: run["status"] for name, run in runs.items()},
        "engine_seconds": {name: run["seconds"] for name, run in successful.items()},
        "engine_counts": {name: run["count"] for name, run in successful.items()},
        "engine_stats": {name: run["stats"] for name, run in successful.items()},
        "engine_over_dpll": {name: run["seconds"] / reference_run["seconds"]
                             for name, run in successful.items() if reference_run["seconds"]},
        "boundary_links": sparse.get("boundary_links"),
        "local_assignment_counts": sparse.get("local_assignment_counts", []),
        "joint_signature_counts": sparse.get("joint_signature_counts", []),
        "join_pair_counts": sparse.get("join_pair_counts", []),
        "peak_intermediate_rows": sparse.get("peak_intermediate_rows"),
        "semi_join_pushdown": sparse.get("semi_join_pushdown", semi_join_pushdown),
        "phase1_assignment_counts": sparse.get("phase1_assignment_counts", []),
        "feasible_master_mask_counts": sparse.get("feasible_master_mask_counts", []),
        "peak_frontier_rows": sparse.get("peak_frontier_rows", []),
        "peak_temporary_rows": sparse.get("peak_temporary_rows", []),
        "peak_joint_rows": sparse.get("peak_joint_rows"),
        "sparse_phase_seconds": {name: sparse.get(name) for name in
                                 ("preprocessing_seconds", "build_seconds", "contraction_seconds")},
        "selected_engine": sparse.get("selected_engine"),
        "swap_mobius": swap,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "system": platform.system(), "machine": platform.machine()},
    }


def run_benchmark(num_vars: int, *, family: str = "coupled", seed: int = 0,
                  bridges: int = 4, planted: bool = True, device: str = "auto",
                  num_groups: int = 2, block_size: int = 4,
                  torch_threads: int = 1, engine: str = "lagrange",
                  jet_nodes: int | None = None, jet_mode: str = "hermite",
                  jet_orders: tuple[int, ...] | None = None,
                  max_subset_terms: int = 1_000_000,
                  max_jet_work: int | None = None, **engine_options) -> dict:
    """Time selected exact engines against DPLL, reporting actual signature work.

    The historical default remains Lagrange. ``engine="both"`` compares the
    original pipeline with the Hermite subset-moment engine on the same CNF.
    Ratios are measurements, and values greater than one are retained.
    """
    if engine == "v3":
        return run_v3_benchmark(
            num_vars, family=family, seed=seed, bridges=bridges, planted=planted,
            device=device, num_groups=num_groups, block_size=block_size,
            torch_threads=torch_threads, jet_nodes=(6 if jet_nodes is None else jet_nodes), jet_mode=jet_mode,
            jet_orders=jet_orders, max_subset_terms=max_subset_terms,
            max_jet_work=max_jet_work, **engine_options)
    if engine not in ("lagrange", "hermite", "both"):
        raise ValueError("engine must be lagrange, hermite, both, or v3")
    jet_nodes = 2 if jet_nodes is None else jet_nodes
    if type(torch_threads) is not int or torch_threads <= 0:
        raise ValueError("torch_threads must be a positive integer")
    generators = {"coupled": coupled_3sat, "random": random_3sat,
                  "bipartite": bipartite_3sat}
    if family not in generators:
        raise ValueError(f"unknown family: {family}")
    generator_options = {"seed": seed, "planted": planted}
    if family == "coupled":
        generator_options["bridges"] = bridges
    cnf = generators[family](num_vars, **generator_options)
    transformed = reduce_clause_width_deterministic(cnf, max_width=engine_options.get("max_width", 3))
    if engine_options.get("max_degree") is not None:
        transformed = split_high_degree_variables(transformed, max_degree=engine_options["max_degree"])
    partitioned = partition_and_decouple_boundary(transformed, num_groups=num_groups,
                                                 block_size=block_size)
    boundary = len(partitioned.boundary_links)
    measurement_states = prod(len(block.link_indices) + 1
                              for block in partitioned.measurement_blocks)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_threads)
    try:
        start = perf_counter()
        reference = dpll_count(cnf)
        dpll_seconds = perf_counter() - start
        results = {}
        timings = {}
        engine_stats = {}
        if engine in ("lagrange", "both"):
            start = perf_counter()
            value, raw_stats = count_with_stats(cnf, device=device, num_groups=num_groups,
                                                block_size=block_size, **engine_options)
            timings["lagrange"] = perf_counter() - start
            results["lagrange"] = value
            engine_stats["lagrange"] = _stats_dict(raw_stats)
        if engine in ("hermite", "both"):
            from .stage3_jet_engine import jet_count_with_stats

            start = perf_counter()
            value, raw_stats = jet_count_with_stats(
                cnf, device=device, num_groups=num_groups, block_size=block_size,
                num_nodes=jet_nodes, mode=jet_mode, orders=jet_orders,
                max_subset_terms=max_subset_terms,
                max_jet_work=(max_jet_work if max_jet_work is not None
                              else engine_options.get("max_work", 2_000_000_000)),
                **engine_options,
            )
            timings["hermite"] = perf_counter() - start
            results["hermite"] = value
            engine_stats["hermite"] = _stats_dict(raw_stats)
    finally:
        torch.set_num_threads(previous_threads)
    for name, value in results.items():
        if value != reference:
            raise AssertionError(f"{name} count {value} differs from DPLL count {reference}")
    primary = "lagrange" if "lagrange" in results else "hermite"
    counted, stats = results[primary], engine_stats[primary]
    tensor_seconds = sum(timings.values())
    conditioning = [prod(group_states) for group_states in stats.get("conditioning_states", [])]
    residual = [sum(group_bits) for group_bits in stats.get("residual_active_bits", [])]
    # Boundary-free formulas perform no interpolation evaluations.
    if not conditioning and not boundary:
        conditioning = [1]
        residual = [0]
    row = {
        "family": family, "seed": seed, "planted": planted, "engine": engine,
        "jet_configuration": ({"num_nodes": jet_nodes, "mode": jet_mode,
                               "orders": jet_orders, "max_subset_terms": max_subset_terms,
                               "max_jet_work": (max_jet_work if max_jet_work is not None
                                                else engine_options.get("max_work", 2_000_000_000))}
                              if engine != "lagrange" else None),
        "n": num_vars, "m": len(cnf.clauses), "boundary_links": boundary,
        "partition_configuration": {"num_groups": num_groups, "block_size": block_size},
        "engine_options": dict(engine_options),
        "preprocessing_num_vars": transformed.num_vars,
        "preprocessing_group_variable_counts": [len(group) for group in partitioned.group_vars],
        "binary_link_states": 1 << boundary,
        "measurement_states": measurement_states,
        "nominal_ratio": (1 << boundary) / measurement_states,
        "conditioning_states_max": max(conditioning, default=0),
        "conditioning_states_sum": sum(conditioning),
        "residual_bits_max": max(residual, default=0),
        "residual_bits_sum": sum(residual),
        "tensor_seconds": tensor_seconds, "dpll_seconds": dpll_seconds,
        "tensor_over_dpll": tensor_seconds / dpll_seconds if dpll_seconds else None,
        "evaluations_per_second": sum(s.get("evaluations", 0) for s in engine_stats.values()) / tensor_seconds,
        "count": counted, "reference_count": reference,
        "device": stats.get("device", device), "torch_threads": torch_threads,
        "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                     "system": platform.system(), "machine": platform.machine()},
        "stats": stats,
        "engine_counts": results,
        "engine_seconds": timings,
        "engine_stats": engine_stats,
        "lagrange_conditioning_states_sum": (
            sum(prod(states) for states in engine_stats["lagrange"].get("conditioning_states", []))
            if "lagrange" in engine_stats else None
        ),
        "preprocessing_local_assignment_counts": {
            name: values.get("local_assignment_counts", []) for name, values in engine_stats.items()
        },
        "preprocessing_signature_cells": {
            name: sum(source * incoming for source, incoming in zip(
                values.get("source_signature_counts", []), values.get("incoming_signature_counts", [])
            )) for name, values in engine_stats.items()
        },
        "preprocessing_source_signature_counts": {
            name: values.get("source_signature_counts", []) for name, values in engine_stats.items()
        },
        "preprocessing_incoming_signature_counts": {
            name: values.get("incoming_signature_counts", []) for name, values in engine_stats.items()
        },
    }
    if "hermite" in engine_stats:
        row.update(_jet_metrics(engine_stats["hermite"]))
        baseline_work = row["lagrange_conditioning_states_sum"]
        row["hermite_over_lagrange_signature_work"] = (
            row["hermite_subset_conditioning_states_sum"] / baseline_work if baseline_work else None
        )
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+",
                        help="variable counts, separated by spaces or commas")
    parser.add_argument("--family", choices=("coupled", "random", "bipartite", "all"),
                        default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bridges", type=int, default=4)
    parser.add_argument("--unplanted", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-groups", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--engine", choices=("lagrange", "hermite", "both", "v3"), default="lagrange")
    parser.add_argument("--jet-nodes", type=int, help="default: 6 for v3; 2 otherwise")
    parser.add_argument("--repeats", type=int, default=1, help="end-to-end v3 timing samples")
    parser.add_argument("--max-join-pairs", type=int, default=8_000_000)
    parser.add_argument("--no-semi-join-pushdown", action="store_true",
                        help="use the v3 unfiltered sparse construction in --engine v3")
    parser.add_argument("--swap-max-work", type=int, default=2_000_000)
    parser.add_argument("--jet-mode", choices=("hermite", "single_node_taylor"), default="hermite")
    parser.add_argument("--jet-orders", nargs="+", help="jet orders, separated by spaces or commas")
    parser.add_argument("--max-subset-terms", type=int, default=1_000_000)
    parser.add_argument("--max-jet-work", type=int,
                        help="jet-specific work budget; defaults to --max-work")
    parser.add_argument("--max-work", type=int, default=2_000_000_000)
    parser.add_argument("--max-memory-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-local-assignments", type=int, default=4_000_000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--csv", type=Path)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    size_arguments = args.sizes or (["18", "22", "26", "30"] if args.engine == "v3"
                                    else ["18", "22", "26"])
    sizes = [int(size) for item in size_arguments for size in item.split(",")]
    if args.family == "all":
        families = ("coupled", "random", "bipartite")
    elif args.family is None:
        families = ("coupled", "bipartite") if args.engine == "v3" else ("coupled",)
    else:
        families = (args.family,)
    jet_nodes = args.jet_nodes if args.jet_nodes is not None else (6 if args.engine == "v3" else 2)
    orders = (tuple(int(order) for item in args.jet_orders for order in item.split(","))
              if args.jet_orders is not None else None)
    print("Default coupled family uses bounded interfaces; no general speedup is proven.")
    print("ratio=2^B/product(k+1) is nominal; cond_max includes residual bits.")
    print("eval/s includes setup, all interpolation nodes, and all CRT primes; exports include full details.")
    if args.engine == "v3":
        print("v3 compares end-to-end median timings; each engine retains its own resource-limit status.")
        print("SWAP-Mobius runtime is miniature-only (n<=6); larger sizes show combinatorial cost estimates only.")
    elif args.engine != "lagrange":
        print("Hermite work counts feasible signatures for every processed derivative subset at every node/prime.")
        print("sum(k)=B and sum(q+1)=B+1: some q>=k; single-node Taylor visits all 2^B subsets.")
        print("Residual masters and preprocessing signatures remain explicit; Hermite work ratios may exceed one.")
    if args.engine == "v3":
        print(f"{'family':<9} {'n':>3} {'DPLL_ms':>10} {'v1_ms':>12} {'v2_ms':>12} "
              f"{'sparse_ms':>12} {'count':>9}")
    elif args.engine == "lagrange":
        print(f"{'family':<9} {'n':>3} {'m':>4} {'B':>3} {'ratio':>8} {'cond_max':>10} "
              f"{'DPLL_ms':>9} {'tensor_ms':>10} {'eval/s':>9} {'count':>9} {'device':<7}")
    else:
        print(f"{'family':<9} {'n':>3} {'B':>3} {'DPLL_ms':>9} {'v1_ms':>10} {'v2_ms':>10} "
              f"{'v1_visits':>12} {'v2_visits':>12} {'count':>9} {'device':<7}")
    rows = []
    try:
        for family in families:
            for size in sizes:
                v3_options = ({"repeats": args.repeats, "max_join_pairs": args.max_join_pairs,
                               "swap_max_work": args.swap_max_work,
                               "semi_join_pushdown": not args.no_semi_join_pushdown}
                              if args.engine == "v3" else {})
                row = run_benchmark(size, family=family, seed=args.seed, bridges=args.bridges,
                                    planted=not args.unplanted, device=args.device,
                                    num_groups=args.num_groups, block_size=args.block_size,
                                    torch_threads=args.torch_threads, max_work=args.max_work,
                                    max_memory_bytes=args.max_memory_bytes,
                                    max_local_assignments=args.max_local_assignments,
                                    engine=args.engine, jet_nodes=jet_nodes, jet_mode=args.jet_mode,
                                    jet_orders=orders, max_subset_terms=args.max_subset_terms,
                                    max_jet_work=args.max_jet_work, **v3_options)
                rows.append(row)
                if args.engine == "v3":
                    cells = [f"{1000 * run['seconds']:.3f}" if run["status"] == "ok"
                             else run["status"] for run in row["engine_runs"].values()]
                    print(f"{family:<9} {row['n']:>3} {cells[0]:>10} {cells[1]:>12} "
                          f"{cells[2]:>12} {cells[3]:>12} {row['count']:>9}")
                    print(f"  joint signatures={row['joint_signature_counts']}; "
                          f"join pairs={row['join_pair_counts']}; "
                          f"peak joint rows={row.get('peak_joint_rows', row['peak_intermediate_rows'])}; "
                          f"peak frontiers={row.get('peak_frontier_rows', [])}")
                    for name, run in row["engine_runs"].items():
                        if run["status"] != "ok":
                            print(f"  {name}: {run['status']}: {run['reason']}")
                elif args.engine == "lagrange":
                    print(f"{family:<9} {row['n']:>3} {row['m']:>4} {row['boundary_links']:>3} "
                          f"{row['nominal_ratio']:>8.3g} {row['conditioning_states_max']:>10} "
                          f"{1000 * row['dpll_seconds']:>9.3f} {1000 * row['tensor_seconds']:>10.3f} "
                          f"{row['evaluations_per_second']:>9.1f} {row['count']:>9} {row['device']:<7}")
                else:
                    v1_ms = (f"{1000 * row['engine_seconds']['lagrange']:.3f}"
                             if "lagrange" in row["engine_seconds"] else "-")
                    v2_ms = 1000 * row["engine_seconds"]["hermite"]
                    v1_visits = row["lagrange_conditioning_states_sum"]
                    print(f"{family:<9} {row['n']:>3} {row['boundary_links']:>3} "
                          f"{1000 * row['dpll_seconds']:>9.3f} {v1_ms:>10} {v2_ms:>10.3f} "
                          f"{str(v1_visits) if v1_visits is not None else '-':>12} "
                          f"{row['hermite_subset_conditioning_states_sum']:>12} "
                          f"{row['count']:>9} {row['device']:<7}")
                    nodes = {node["node_index"]: node for node in row["hermite_node_evaluations"]}
                    summary = "; ".join(
                        f"s{index}(k={node['k']},q={node['q']},subsets={node['processed_subsets']},"
                        f"full-set={node['full_set_processed']})" for index, node in sorted(nodes.items())
                    )
                    print(f"  exact signature-work ratio(v2/v1)={row['hermite_over_lagrange_signature_work']}; "
                          f"schedule: {summary or 'boundary-free'}")
    except (ResourceLimitError, UnsupportedDeviceError) as exc:
        print(f"Benchmark failed explicitly: {exc}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
    if args.csv:
        columns = [key for key in rows[0] if key not in ("stats", "versions", "engine_stats")] if rows else []
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                              for key, value in row.items()} for row in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
