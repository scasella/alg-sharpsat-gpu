"""HTTP and command-line coverage without invoking a live language model."""

import json
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from alg_sharpsat import webapp

pytestmark = pytest.mark.g18


@pytest.fixture
def server():
    instance = webapp.create_server(port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{instance.server_port}"
    instance.shutdown()
    instance.server_close()
    thread.join()


def request(server, path, data=None):
    req = Request(
        server + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def payload(**extra):
    result = dict(
        variables=[
            dict(name="observed", description="fact", category="evidence"),
            dict(name="act", description="action"),
        ],
        rules=[
            dict(
                rule_id="r",
                description="observed prevents act",
                expr="observed -> not act",
            )
        ],
        probs=[0.1, 0.9],
        mode="exact_rational",
    )
    result.update(extra)
    return result


def test_clamps_apply_to_all_inference_and_cached_policy(server):
    data = payload(clamped_evidence={"observed": 1}, num_samples=3)
    first_status, first = request(server, "/api/evaluate", data)
    status, result = request(server, "/api/evaluate", data)
    assert first_status == status == 200
    assert result["cache_hit"]
    assert result["prior_probs"]["observed"] == 1
    assert result["raw_threshold_action"] == {"observed": True, "act": True}
    assert result["map_action"] == {"observed": True, "act": False}
    assert result["posterior_marginals"]["observed"] == 1
    assert all(s["assignment"] == result["map_action"] for s in result["samples"])
    assert all(s["conditional_probability"] == 1 for s in result["samples"])
    assert first["exact_rational_z"] == "1/10"


def test_zero_evidence_extension(server):
    data = payload(clamped_evidence={"observed": 1})
    data["probs"] = [0.1, 1]
    status, result = request(server, "/api/evaluate", data)
    assert status == 200
    assert result["map_action"] is None and result["posterior_marginals"] is None
    assert "extension" in result["diagnostics"][0]


@pytest.mark.parametrize(
    "extra",
    [
        {"clamped_evidence": {"unknown": 1}},
        {"clamped_evidence": {"observed": 0.5}},
        {"clamped_evidence": {"act": 1}},
        {"clamped_evidence": {"observed": 1}, "clamped_vars": {"observed": 0}},
    ],
)
def test_invalid_clamps(server, extra):
    assert request(server, "/api/evaluate", payload(**extra))[0] == 400


def test_mocked_extraction_and_errors(server, monkeypatch):
    from alg_sharpsat import llm_bridge

    def extract(policy, text, **kwargs):
        assert not webapp._LOCK._is_owned()
        assert kwargs["backend"] == "codex" and kwargs["timeout"] == 30
        return dict(
            clamped_evidence={"observed": 1},
            probs=[0.1, 0.9],
            rationale="fact",
            llm_ms=2,
            model="mock",
            backend="codex",
            evidence_sources={"observed": "observed"},
        )

    monkeypatch.setattr(llm_bridge, "extract", extract)
    status, result = request(
        server, "/api/intercept", payload(scenario_text="observed")
    )
    assert status == 200
    assert result["prior_probs"]["observed"] == 1
    assert result["extracted_probs"] == [0.1, 0.9]
    assert result["evidence_sources"] == {"observed": "observed"}
    monkeypatch.setattr(
        llm_bridge, "extract", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError())
    )
    assert (
        request(server, "/api/intercept", payload(scenario_text="observed"))[0] == 504
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"scenario_text": ""},
        {"scenario_text": "hi", "backend": "bad"},
        {"scenario_text": "hi", "timeout": -1},
    ],
)
def test_intercept_invalid_requests(server, extra):
    assert request(server, "/api/intercept", payload(**extra))[0] == 400


def test_domains_metadata(server):
    status, result = request(server, "/api/domains")
    assert status == 200
    for domain in result["domains"]:
        assert len(domain["sample_prompts"]) == 3
        assert all(
            "effective_role" in v and "can_clamp" in v for v in domain["variables"]
        )


def test_deterministic_http_and_cli(server):
    domain, pack = next(iter(webapp._domains().items()))
    text = pack.sample_prompts[0]
    status, result = request(
        server,
        "/api/intercept",
        dict(domain=domain, scenario_text=text, backend="deterministic"),
    )
    assert status == 200
    assert result["backend"] == "deterministic"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "alg_sharpsat.webapp",
            "--intercept",
            text,
            "--domain",
            domain,
            "--backend",
            "deterministic",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert all(
        label in proc.stdout
        for label in (
            "Evidence:",
            "Raw action:",
            "MAP changes:",
            "All marginals:",
            "Timings:",
        )
    )


def test_diff_and_batch_preserve_clamps(server, monkeypatch):
    data = payload(clamped_evidence={"observed": 1})
    data["base_rules"] = data["rules"]
    data["modified_rules"] = []
    status, result = request(server, "/api/policy-diff", data)
    assert status == 200
    assert result["old_allowed_exact_rational_prob"] == "1/10"
    assert result["new_allowed_exact_rational_prob"] == "1/1"
    from alg_sharpsat.policy_dsl import CompiledPolicy

    original = CompiledPolicy.evaluate
    seen = []

    def recording(self, probs=None, *args, **kwargs):
        if isinstance(probs, list) and probs and hasattr(probs[0], "__len__"):
            seen.extend(row[0] for row in probs)
        return original(self, probs, *args, **kwargs)

    monkeypatch.setattr(CompiledPolicy, "evaluate", recording)
    assert request(server, "/api/batch-benchmark", data)[0] == 200
    assert seen and set(seen) == {1}


@pytest.mark.parametrize(
    "extra",
    [
        {"mode": "bad"},
        {"num_samples": -1},
        {"seed": True},
        {"clamped_evidence": {"missing": 1}},
    ],
)
def test_intercept_validates_before_extraction(server, monkeypatch, extra):
    from alg_sharpsat import llm_bridge

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid requests must not invoke extraction")

    monkeypatch.setattr(llm_bridge, "extract", forbidden)
    assert (
        request(server, "/api/intercept", payload(scenario_text="observed", **extra))[0]
        == 400
    )


def test_intercept_merges_manual_clamps_and_rejects_conflicts(server, monkeypatch):
    from alg_sharpsat import llm_bridge

    extraction = dict(
        clamped_evidence={},
        probs=[0.1, 0.9],
        rationale="none",
        llm_ms=0,
        model="mock",
        backend="codex",
        evidence_sources={},
    )
    monkeypatch.setattr(llm_bridge, "extract", lambda *a, **kw: extraction)
    data = payload(scenario_text="observed", clamped_evidence={"observed": 1})
    status, result = request(server, "/api/intercept", data)
    assert status == 200 and result["clamped_evidence"] == {"observed": 1}
    extraction["clamped_evidence"] = {"observed": 0}
    status, result = request(server, "/api/intercept", data)
    assert status == 400 and "conflict" in result["error"]


def test_context_inference_is_separate_from_action_repairs(server):
    data = dict(
        variables=[
            dict(name="unknown", description="unobserved fact", role="evidence"),
            dict(name="locked", description="known fact", role="evidence"),
            dict(name="action", description="action", role="action"),
            dict(name="safe", description="safeguard", role="safeguard"),
        ],
        rules=[
            dict(rule_id="fact", description="require fact", expr="not unknown"),
            dict(rule_id="action", description="require action", expr="action"),
            dict(rule_id="safe", description="require safeguard", expr="safe"),
        ],
        probs=[0.9, 0.1, 0.1, 0.1],
        clamped_evidence={"locked": 1},
        mode="exact_rational",
    )
    status, result = request(server, "/api/evaluate", data)
    assert status == 200
    assert {v["name"] for v in result["repaired_variables"]} == {"action", "safe"}
    assert result["inferred_context_changes"] == [
        {"name": "unknown", "from": True, "to": False}
    ]
    assert result["map_action"]["locked"] is True
    assert result["map_action"]["unknown"] is False
