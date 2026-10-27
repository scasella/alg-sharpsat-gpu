"""HTTP-level Studio checks on an ephemeral loopback server."""

import json
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
import pytest
from alg_sharpsat import webapp

pytestmark = pytest.mark.g17


@pytest.fixture
def service():
    webapp._CACHE.clear()
    server = webapp.create_server(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def request(service, path, data=None, origin=None):
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
    req = Request(
        service + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers=headers,
    )
    try:
        with urlopen(req, timeout=20) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def policy(**updates):
    data = dict(
        variables=[dict(name="a", description=""), dict(name="b", description="")],
        rules=[dict(rule_id="equal", description="Agreement", expr="a iff b")],
        probs={"a": 0.2, "b": 0.7},
    )
    data.update(updates)
    return data


@pytest.mark.parametrize("mode", ["float64", "exact_rational"])
def test_evaluate_aux_neutral_and_cache(service, mode):
    status, out = request(service, "/api/evaluate", policy(mode=mode, num_samples=4))
    assert status == 200 and out["acceptance_prob"] == pytest.approx(0.38)
    assert out["posterior_marginals"]["a"] == pytest.approx(7 / 19)
    assert out["pruning_telemetry"]["valid_model_count"] == 2
    assert out["raw_is_valid"] is False and out["map_action"] == {
        "a": False,
        "b": False,
    }
    assert all(s["assignment"]["a"] == s["assignment"]["b"] for s in out["samples"])
    if mode == "exact_rational":
        assert out["exact_rational_z"] == "19/50"
    with patch.object(
        webapp.PolicySpec,
        "compile_guardrail",
        side_effect=AssertionError("compiled cache miss"),
    ):
        status, cached = request(service, "/api/evaluate", policy(mode=mode))
    assert status == 200 and cached["cache_hit"] and cached["compile_ms"] == 0


def test_zero_mass(service):
    for mode in ("float64", "exact_rational"):
        status, out = request(
            service, "/api/evaluate", policy(probs=[0, 1], mode=mode, num_samples=5)
        )
        assert status == 200 and out["acceptance_prob"] == 0
        assert out["map_action"] is None and out["posterior_marginals"] is None
        assert out["samples"] == [] and out["diagnostics"]


def test_diff_and_benchmark(service):
    data = policy()
    data["base_rules"] = data.pop("rules")
    data["modified_rules"] = []
    status, out = request(service, "/api/policy-diff", data)
    assert (
        status == 200 and out["newly_exposed_count"] == 2 and out["revoked_count"] == 0
    )
    assert out["newly_exposed_prob"] == pytest.approx(0.62)
    status, out = request(service, "/api/batch-benchmark", policy())
    assert (
        status == 200
        and out["batch_size"] == 256
        and out["median_batch_ms"] > 0
        and out["device"] == "cpu"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"probs": [2, 0.5]},
        {"mode": "exec"},
        {"num_samples": 101},
        {
            "rules": [
                dict(
                    rule_id="evil",
                    description="",
                    expr="__import__('os').system('whoami')",
                )
            ]
        },
        {"variables": [dict(name="x", description="")] * 25},
    ],
)
def test_validation(service, updates):
    assert request(service, "/api/evaluate", policy(**updates))[0] == 400


def test_origin_and_size(service):
    assert (
        request(service, "/api/evaluate", policy(), origin="https://evil.example")[0]
        == 403
    )
    assert (
        request(service, "/api/evaluate", {"padding": "x" * webapp.MAX_BODY})[0] == 413
    )


def test_domains_and_cli(service, capsys):
    status, out = request(service, "/api/domains")
    assert status == 200 and len(out["domains"]) >= 1
    for domain in out["domains"]:
        status, result = request(
            service,
            "/api/evaluate",
            {"domain": domain["domain_id"], "rules": domain["rules"]},
        )
        assert status == 200 and result["pruning_telemetry"]["valid_model_count"] > 0
    webapp.main(["--demo-cli"])
    output = capsys.readouterr().out
    assert output.count("    Models:") == sum(
        len(domain["presets"]) for domain in out["domains"]
    )
    assert "Violations:" in output and "MAP flips:" in output and "Evaluate:" in output
    assert all(
        domain["domain_id"] in output and domain["preset_labels"]
        for domain in out["domains"]
    )


def test_cache_preserves_unconstrained_originals(service):
    rules = [dict(rule_id="force", description="", expr="a")]
    one = dict(variables=[dict(name="a", description="")], rules=rules, probs=[0.3])
    two = dict(
        variables=[dict(name="a", description=""), dict(name="b", description="")],
        rules=rules,
        probs=[0.3, 0.8],
    )
    assert (
        request(service, "/api/evaluate", one)[1]["pruning_telemetry"][
            "valid_model_count"
        ]
        == 1
    )
    result = request(service, "/api/evaluate", two)[1]
    assert (
        not result["cache_hit"]
        and result["pruning_telemetry"]["valid_model_count"] == 2
    )
    assert result["posterior_marginals"]["b"] == pytest.approx(0.8)


def test_exact_rare_mass_survives_float_underflow(service):
    names = [f"x{i}" for i in range(8)]
    data = dict(
        variables=[dict(name=n, description="") for n in names],
        rules=[
            dict(
                rule_id="all",
                description="",
                expr="(" + " and ".join(names) + ") and (x0 iff x1)",
            )
        ],
        probs=["1e-50"] * 8,
        mode="exact_rational",
        num_samples=2,
    )
    status, out = request(service, "/api/evaluate", data)
    assert status == 200 and out["acceptance_prob"] == 0
    assert out["exact_rational_z"] != "0/1" and all(out["map_action"].values())
    assert all(value == 1 for value in out["posterior_marginals"].values())
    assert all(sample["conditional_probability"] == 1 for sample in out["samples"])
    assert all(sample["probability"] == 0 for sample in out["samples"])
    assert all(
        sample["exact_rational_probability"] != "0/1" for sample in out["samples"]
    )


def test_fraction_batch_and_sample_probability_contract(service):
    status, out = request(
        service, "/api/batch-benchmark", policy(probs=["1/2", "7/10"])
    )
    assert status == 200 and out["batch_size"] == 256
    status, out = request(service, "/api/evaluate", policy(num_samples=5))
    assert status == 200 and out["inference_mode"] == "exact_rational"
    assert out["exact_rational_map_prob"] == "6/25"
    for sample in out["samples"]:
        prior = 0.14 if sample["assignment"]["a"] else 0.24
        assert sample["probability"] == pytest.approx(prior)
        assert sample["conditional_probability"] == pytest.approx(prior / 0.38)
    status, warm = request(service, "/api/evaluate", policy())
    assert status == 200 and warm["compile_ms"] == 0
    assert warm["eval_ms"] >= 0 and warm["map_ms"] >= 0 and warm["sample_ms"] >= 0
    assert warm["total_ms"] >= warm["eval_ms"] + warm["map_ms"]


def test_bad_syntax_diagnostic(service):
    status, out = request(
        service,
        "/api/evaluate",
        policy(rules=[dict(rule_id="bad", description="", expr="a implies (")]),
    )
    assert status == 400 and "expression" in out["error"]
    assert "Traceback" not in out["error"]


@pytest.mark.parametrize(
    "updates",
    [
        {"probs": ["1e-1000000000", ".5"]},
        {"variables": [dict(name="a", description="", default_prob="1e-1000000000")]},
    ],
)
def test_probability_exponent_bomb_rejected(service, updates):
    status, out = request(service, "/api/evaluate", policy(**updates))
    assert status == 400 and "exponent" in out["error"]
