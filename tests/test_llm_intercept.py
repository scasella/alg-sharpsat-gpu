"""Extractor boundaries; live Codex use is opt-in and never part of ordinary CI."""

import json
import os
import shutil
import subprocess

import pytest

from alg_sharpsat.llm_bridge import CodexLunaExtractor, ExtractionError, extract
from alg_sharpsat.policy_dsl import PolicySpec, VariableSpec, RuleSpec

pytestmark = pytest.mark.g18


@pytest.fixture
def policy():
    return PolicySpec(
        [
            VariableSpec("personal_data", "Personal data present", role="evidence"),
            VariableSpec("consent", "Verified consent", role="safeguard"),
        ],
        [
            RuleSpec(
                "consent_required",
                "Personal data requires consent",
                "personal_data -> consent",
            )
        ],
    )


def test_explicit_deterministic_parser_and_negation(policy):
    result = extract(policy, "personal_data=true; no consent", backend="deterministic")
    assert result["backend"] == "deterministic"
    assert result["clamped_evidence"] == {"personal_data": 1}
    assert result["evidence_sources"] == {"personal_data": "personal_data=true"}
    assert result["probs"]["consent"] == 0.1
    assert "candidate evidence" in result["rationale"]
    assert result["llm_ms"] >= 0


def test_conflicting_assignments_are_uncertain(policy):
    result = extract(policy, "consent=true; consent=false", backend="deterministic")
    assert not result["clamped_evidence"]
    assert result["probs"]["consent"] == 0.5


def valid_raw():
    return {
        "clamped_evidence": {"personal_data": 1.0},
        "probs": {"personal_data": 0.9, "consent": 0.5},
        "rationale": "Review candidate evidence",
        "provenance": {"personal_data": "personal_data=true"},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["probs"].update(consent=float("nan")),
        lambda d: d["probs"].update(consent=1),
        lambda d: d["probs"].pop("consent"),
        lambda d: d["clamped_evidence"].update(unknown=1),
        lambda d: d["clamped_evidence"].update(personal_data=0.7),
        lambda d: d["provenance"].update(personal_data="invented quote"),
        lambda d: d.update(extra="unexpected"),
    ],
)
def test_invalid_model_output_is_rejected(policy, mutation):
    raw = valid_raw()
    mutation(raw)
    with pytest.raises(ExtractionError):
        CodexLunaExtractor._validate(policy, "personal_data=true", raw)


def test_actions_cannot_be_clamped():
    from types import SimpleNamespace

    policy = SimpleNamespace(
        variable_names=("personal_data", "consent"),
        variables=[
            SimpleNamespace(name="personal_data", effective_role="action"),
            SimpleNamespace(name="consent", effective_role="evidence"),
        ],
    )
    with pytest.raises(ExtractionError, match="actions"):
        CodexLunaExtractor._validate(policy, "personal_data=true", valid_raw())


def test_codex_launch_is_isolated_and_bounded(policy, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/codex")

    def run(command, **kwargs):
        assert kwargs["timeout"] == 7
        assert kwargs["cwd"] != os.getcwd()
        assert command[-1] == "-"
        assert "--ignore-user-config" in command
        assert "mcp_servers={}" in command
        assert "shell_tool" in command and "plugins" in command and "apps" in command
        assert "personal_data=true" in kwargs["input"]
        from pathlib import Path

        Path(command[command.index("-o") + 1]).write_text(json.dumps(valid_raw()))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    result = extract(policy, "personal_data=true", timeout=7)
    assert result["model"] == "gpt-6-luna"
    assert result["clamped_evidence"] == {"personal_data": 1}


def test_timeout_does_not_silently_fallback(policy, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/codex")

    def run(*args, **kwargs):
        raise subprocess.TimeoutExpired("secret command", 1)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(ExtractionError, match="timed out"):
        extract(policy, "hello", timeout=1)


@pytest.mark.skipif(
    os.environ.get("WMC_LIVE_CODEX") != "1" or not shutil.which("codex"),
    reason="set WMC_LIVE_CODEX=1 for authenticated live Codex smoke",
)
def test_live_codex_smoke(policy):
    result = extract(
        policy, "personal_data=true. Consent is explicitly verified.", timeout=60
    )
    assert result["backend"] == "codex"
    assert result["model"] == "gpt-6-luna"
    assert set(result["probs"]) == set(policy.variable_names)
    assert result["llm_ms"] > 0
    compiled = policy.compile_guardrail()
    evaluated = compiled.evaluate(
        result["probs"], mode="exact_rational", clamped_vars=result["clamped_evidence"]
    )
    assert evaluated.partition_function > 0
    assignment, weight = compiled.map_inference(
        result["probs"], clamped_vars=result["clamped_evidence"]
    )
    assert assignment is not None and weight > 0
    values = dict(zip(policy.variable_names, assignment))
    assert not policy.check_violations(values)
    assert all(
        values[name] == value for name, value in result["clamped_evidence"].items()
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("The service is in production.", 1),
        ("The service is not in production.", 0),
        ("If it were in production, deploy.", None),
        ("Is it in production?", None),
        ("Do not run in production.", None),
    ],
)
def test_narrow_natural_language_aliases(text, expected):
    policy = PolicySpec(
        [VariableSpec("prod_environment", "Production", role="evidence")], []
    )
    result = extract(policy, text, backend="deterministic")
    assert result["clamped_evidence"] == (
        {} if expected is None else {"prod_environment": expected}
    )


def test_explicit_safeguard_status_and_action_intent():
    policy = PolicySpec(
        [
            VariableSpec("mfa_verified", "MFA", role="safeguard"),
            VariableSpec("deploy", "Deploy", role="action"),
        ],
        [],
    )
    result = extract(policy, "MFA is verified. deploy=true", backend="deterministic")
    assert result["clamped_evidence"] == {"mfa_verified": 1}


def test_hypothetical_explicit_assignments_do_not_clamp():
    policy = PolicySpec(
        [VariableSpec("prod_environment", "Production", role="evidence")], []
    )
    result = extract(
        policy,
        "If prod_environment=true, propose a migration.",
        backend="deterministic",
    )
    assert result["clamped_evidence"] == {}
    assert result["probs"]["prod_environment"] == 0.5


def test_explicit_safeguard_assignments_and_proposed_status():
    names = ("mfa_verified", "peer_review_approved", "fresh_snapshot_verified")
    policy = PolicySpec([VariableSpec(n, n, role="safeguard") for n in names], [])
    result = extract(
        policy,
        "mfa_verified=true; peer_review_approved=false; fresh_snapshot_verified=false",
        backend="deterministic",
    )
    assert result["clamped_evidence"] == dict(zip(names, (1, 0, 0)))
    proposed = extract(
        policy,
        "Propose mfa_verified=true; If peer_review_approved=false, wait.",
        backend="deterministic",
    )
    assert proposed["clamped_evidence"] == {}


@pytest.mark.parametrize(
    "text",
    [
        "MFA is verified. MFA is not verified.",
        "mfa_verified=true. MFA is not verified.",
        "MFA is verified. mfa_verified=false.",
    ],
)
def test_conflicting_aliases_and_assignments_remain_uncertain(text):
    policy = PolicySpec([VariableSpec("mfa_verified", "MFA", role="safeguard")], [])
    result = extract(policy, text, backend="deterministic")
    assert result["clamped_evidence"] == {}
    assert result["probs"]["mfa_verified"] == 0.5


def test_rational_default_probabilities_are_supported(monkeypatch):
    policy = PolicySpec(
        [VariableSpec("x", "X", default_prob="1/3", role="evidence")], []
    )
    assert extract(policy, "Nothing observed.", backend="deterministic")["probs"][
        "x"
    ] == pytest.approx(1 / 3)
    monkeypatch.setattr(shutil, "which", lambda name: "/fake/codex")

    def run(command, **kwargs):
        assert json.loads(kwargs["input"].splitlines()[-1])["variables"][0][
            "default_prob"
        ] == pytest.approx(1 / 3)
        from pathlib import Path

        Path(command[command.index("-o") + 1]).write_text(
            json.dumps(
                {
                    "clamped_evidence": {"x": None},
                    "provenance": {"x": None},
                    "probs": {"x": 1 / 3},
                    "rationale": "Uncertain",
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    assert extract(policy, "Nothing observed.")["probs"]["x"] == pytest.approx(1 / 3)
