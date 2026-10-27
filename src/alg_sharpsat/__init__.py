"""Exact cut-and-interpolate #SAT with modular PyTorch tensors."""

from .cnf import CNF, Clause, Literal, brute_force_count, dpll_count, parse_dimacs

__all__ = [
    "CNF", "Clause", "Literal", "brute_force_count", "dpll_count",
    "parse_dimacs", "gpu_algebraic_sharp_sat", "jet_sharp_sat",
    "contract_sharp_sat", "auto_sharp_sat", "swap_mobius_count",
]


def __getattr__(name: str):
    if name in ("contract_sharp_sat", "auto_sharp_sat"):
        from . import stage3_sparse_contract

        return getattr(stage3_sparse_contract, name)
    if name == "swap_mobius_count":
        from .stage4_swap_mobius import swap_mobius_count

        return swap_mobius_count
    if name == "gpu_algebraic_sharp_sat":
        from .stage3_gpu_engine import gpu_algebraic_sharp_sat

        return gpu_algebraic_sharp_sat
    if name == "jet_sharp_sat":
        from .stage3_jet_engine import jet_sharp_sat

        return jet_sharp_sat
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
