"""Exact cut-and-interpolate #SAT with modular PyTorch tensors."""

from .cnf import CNF, Clause, Literal, brute_force_count, dpll_count, parse_dimacs

__all__ = [
    "VariableSpec", "RuleSpec", "PolicySpec", "CompiledPolicy", "PolicyDiff", "diff_policies",
    "CNF", "Clause", "Literal", "brute_force_count", "dpll_count",
    "parse_dimacs", "gpu_algebraic_sharp_sat", "jet_sharp_sat",
    "CompiledGuardrail", "GuardrailResult", "WMCResult", "compile_guardrail",
    "wmc_sharp_sat", "exact_marginals", "guardrail_project",
    "contract_sharp_sat", "auto_sharp_sat", "swap_mobius_count",
]


def __getattr__(name: str):
    if name in ("VariableSpec", "RuleSpec", "PolicySpec", "CompiledPolicy", "PolicyDiff", "diff_policies"):
        from . import policy_dsl

        return getattr(policy_dsl, name)
    if name in ("CompiledGuardrail", "GuardrailResult", "WMCResult", "compile_guardrail",
                "wmc_sharp_sat", "exact_marginals", "guardrail_project"):
        from . import stage5_wmc_guardrail

        return getattr(stage5_wmc_guardrail, name)
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
