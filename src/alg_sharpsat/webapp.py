"""Local, bounded HTTP interface for the Policy Guardrail Studio."""

from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import asdict
from fractions import Fraction
from decimal import Decimal, InvalidOperation
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import statistics
import threading
import time
import webbrowser

from .policy_dsl import PolicySpec, VariableSpec, RuleSpec, CompiledPolicy

MAX_BODY = 128 * 1024
_CACHE = OrderedDict()
_LOCK = threading.RLock()
OPTIONS = dict(
    device="cpu",
    max_memory_bytes=64 * 1024 * 1024,
    max_work=20_000_000,
    max_join_pairs=200_000,
)


def _domains():
    from .domains import get_domains

    return get_domains()


def _policy(data, rules_key="rules"):
    if data.get("domain"):
        pack = _domains().get(data["domain"])
        if pack is None:
            raise ValueError("unknown domain")
        variables = pack.policy.variables
        rules = pack.policy.rules if rules_key not in data else _rules(data[rules_key])
    else:
        values = data.get("variables")
        if not isinstance(values, list) or not 1 <= len(values) <= 24:
            raise ValueError("provide 1 to 24 variables")
        variables = []
        for value in values:
            if isinstance(value, dict):
                _probability(value.get("default_prob", 0.5))
                variables.append(VariableSpec(**value))
            else:
                variables.append(VariableSpec(value, ""))
        rules = _rules(data.get(rules_key, []))
    if len(variables) > 24:
        raise ValueError("too many variables")
    return PolicySpec(
        variables,
        rules,
        max_variables=24,
        max_tokens=512,
        max_depth=32,
        max_clauses=1024,
        max_aux_variables=128,
    )


def _rules(values):
    if not isinstance(values, list) or len(values) > 64:
        raise ValueError("provide at most 64 rules")
    result = []
    for value in values:
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("expr"), str)
            or len(value["expr"]) > 2048
        ):
            raise ValueError("invalid rule")
        if "enabled" in value and type(value["enabled"]) is not bool:
            raise ValueError("enabled must be boolean")
        result.append(RuleSpec(**value))
    return result


def _compiled(policy):
    cnf = policy.to_cnf()
    key = (
        len(policy.variables),
        cnf.num_vars,
        tuple(
            sorted(
                set(
                    tuple(sorted(set(l.signed for l in c.literals)))
                    for c in cnf.clauses
                )
            )
        ),
    )
    if key in _CACHE:
        raw, historical, count = _CACHE.pop(key)
        _CACHE[key] = (raw, historical, count)
        wrapper = CompiledPolicy.__new__(CompiledPolicy)
        wrapper.policy, wrapper.cnf, wrapper.compiled, wrapper.variable_names = (
            policy,
            cnf,
            raw,
            policy.variable_names,
        )
        return wrapper, True, 0.0, historical, count
    start = time.perf_counter()
    wrapper = policy.compile_guardrail(**OPTIONS)
    count = int(
        wrapper.compiled.evaluate(
            weights=[(1, 1)] * cnf.num_vars, mode="exact_rational"
        ).partition_function
    )
    elapsed = (time.perf_counter() - start) * 1000
    _CACHE[key] = (wrapper.compiled, elapsed, count)
    while len(_CACHE) > 8:
        _CACHE.popitem(last=False)
    return wrapper, False, elapsed, elapsed, count


def _probs(policy, data):
    values = data.get("probs", [v.default_prob for v in policy.variables])
    if isinstance(values, dict):
        if set(values) != set(policy.variable_names):
            raise ValueError("probability names must match variables")
        values = [values[n] for n in policy.variable_names]
    if not isinstance(values, list) or len(values) != len(policy.variables):
        raise ValueError("wrong probability vector length")
    for value in values:
        _probability(value)
    return values


def _probability(value: object) -> Fraction:
    """Validate decimal scale before constructing potentially large integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("invalid probability")
    text = str(value)
    if len(text) > 64:
        raise ValueError("probability text exceeds limit")
    try:
        if "/" in text:
            pieces = text.split("/")
            if len(pieces) != 2 or any(len(part.strip("+-")) > 64 for part in pieces):
                raise ValueError("invalid rational probability")
            result = Fraction(text)
        else:
            decimal = Decimal(text)
            if not decimal.is_finite() or abs(decimal.as_tuple().exponent) > 128:
                raise ValueError("probability exponent outside bounds")
            result = Fraction(decimal)
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ValueError("invalid probability") from exc
    if not 0 <= result <= 1 or result.denominator.bit_length() > 256:
        raise ValueError("probability outside bounds")
    return result


def _rational(value):
    value = Fraction(value)
    return f"{value.numerator}/{value.denominator}"


def _float(value):
    v = float(value)
    return v if math.isfinite(v) else None


def evaluate(data: dict) -> dict:
    total_start = time.perf_counter()
    policy = _policy(data)
    probs = _probs(policy, data)
    names = policy.variable_names
    mode = data.get("mode", "float64")
    if mode not in ("float64", "exact_rational"):
        raise ValueError("invalid mode")
    n = data.get("num_samples", 0)
    seed = data.get("seed", 0)
    if (
        type(n) is not int
        or not 0 <= n <= 100
        or type(seed) is not int
        or abs(seed) > 2**63
    ):
        raise ValueError("invalid sampling options")
    with _LOCK:
        engine, hit, compile_ms, historical, count = _compiled(policy)
        start = time.perf_counter()
        result = engine.evaluate(probs, mode=mode)
        eval_ms = (time.perf_counter() - start) * 1000
        z = result.partition_function
        positive = (
            (z > 0)
            if mode == "exact_rational"
            else math.isfinite(float(result.log_partition))
        )
        raw = dict(zip(names, [Fraction(str(p)) >= Fraction(1, 2) for p in probs]))
        violations = policy.check_violations(raw)
        map_start = time.perf_counter()
        assignment, weight = (
            engine.map_inference(probs, mode="exact_rational")
            if positive
            else (None, None)
        )
        map_ms = (time.perf_counter() - map_start) * 1000
        action = (
            dict(zip(names, map(bool, assignment))) if assignment is not None else None
        )
        samples = []
        sample_start = time.perf_counter()
        if positive and n:
            exact_z = (
                z
                if mode == "exact_rational"
                else engine.evaluate(probs, mode="exact_rational").partition_function
            )
            for sample in engine.sample_conditional(
                probs, num_samples=n, seed=seed, mode="exact_rational"
            ):
                probability = Fraction(1)
                for p, bit in zip(probs, sample):
                    p = Fraction(str(p))
                    probability *= p if bit else 1 - p
                samples.append(
                    {
                        "assignment": dict(zip(names, map(bool, sample))),
                        "probability": _float(probability),
                        "exact_rational_probability": _rational(probability),
                        "conditional_probability": _float(probability / exact_z),
                        "exact_rational_conditional_probability": _rational(
                            probability / exact_z
                        ),
                    }
                )
        sample_ms = (time.perf_counter() - sample_start) * 1000
        stats = engine.compiled.stats
        before = engine.compiled.raw_sparse_stats
        marginals = (
            dict(zip(names, [_float(x) for x in result.posterior_marginals]))
            if positive
            else None
        )
        out = dict(
            compile_ms=compile_ms,
            compile_historical_ms=historical,
            eval_ms=eval_ms,
            map_ms=map_ms,
            sample_ms=sample_ms,
            total_ms=(time.perf_counter() - total_start) * 1000,
            cache_hit=hit,
            acceptance_prob=_float(z),
            log_acceptance_prob=_float(result.log_partition),
            kl_divergence_nats=_float(result.kl_divergence_nats),
            kl_divergence_bits=_float(result.kl_divergence_nats / math.log(2)),
            exact_rational_z=_rational(z) if mode == "exact_rational" else None,
            prior_probs=dict(zip(names, map(float, map(Fraction, map(str, probs))))),
            posterior_marginals=marginals,
            exact_rational_marginals=dict(
                zip(names, map(_rational, result.posterior_marginals))
            )
            if positive and mode == "exact_rational"
            else None,
            raw_threshold_action=raw,
            raw_is_valid=not violations,
            violated_rules=[asdict(r) for r in violations],
            map_action=action,
            map_prob=_float(weight) if weight is not None else None,
            exact_rational_map_prob=_rational(weight) if weight is not None else None,
            posterior_mode=mode,
            inference_mode="exact_rational",
            repaired_variables=[
                dict(name=n, **{"from": raw[n], "to": action[n]})
                for n in names
                if action is not None and raw[n] != action[n]
            ],
            samples=samples,
            diagnostics=[]
            if positive
            else [
                "Policy has zero probability mass; conditional inference is undefined."
            ],
            pruning_telemetry=dict(
                total_nominal_states=2 ** len(names),
                valid_model_count=count,
                phase1_rows=list(getattr(before, "phase1_assignment_counts", [])),
                semi_join_rows=list(getattr(before, "local_assignment_counts", [])),
                backward_pruned_rows=list(
                    getattr(stats, "retained_assignment_counts", [])
                ),
                row_counts_description="Per partition: phase-one support, semi-join support, and retained support after backward pruning.",
            ),
        )
        return out


def policy_diff(data: dict) -> dict:
    old = _policy(data, "base_rules")
    new = _policy(data, "modified_rules")
    probs = _probs(old, data)
    combined = PolicySpec(
        old.variables,
        [
            RuleSpec("base_" + str(i), r.description, r.expr, r.enabled)
            for i, r in enumerate(old.rules)
        ]
        + [
            RuleSpec("new_" + str(i), r.description, r.expr, r.enabled)
            for i, r in enumerate(new.rules)
        ],
        max_clauses=2048,
        max_aux_variables=256,
    )
    with _LOCK:
        engines = [_compiled(p) for p in (old, new, combined)]
        counts = [e[4] for e in engines]
        masses = [
            e[0].evaluate(probs, mode="exact_rational").partition_function
            for e in engines
        ]
    a, b, c = counts
    x, y, z = masses
    out = dict(
        old_allowed_count=a,
        new_allowed_count=b,
        intersection_count=c,
        newly_exposed_count=b - c,
        revoked_count=a - c,
    )
    for name, v in zip(
        ("old_allowed", "new_allowed", "intersection", "newly_exposed", "revoked"),
        (x, y, z, y - z, x - z),
    ):
        out[name + "_prob"] = _float(v)
        out[name + "_exact_rational_prob"] = _rational(v)
    return out


def batch_benchmark(data: dict) -> dict:
    policy = _policy(data)
    probs = _probs(policy, data)
    batch = [
        [0.98 * float(Fraction(str(p))) + 0.02 * ((i + 0.5) / 256) for p in probs]
        for i in range(256)
    ]
    with _LOCK:
        engine, hit, ms, historical, count = _compiled(policy)
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            engine.evaluate(batch, mode="float64")
            timings.append((time.perf_counter() - start) * 1000)
    median = statistics.median(timings)
    return dict(
        batch_size=256,
        repeats=3,
        median_batch_ms=median,
        evaluations_per_second=256000 / median,
        device="cpu",
        torch_threads=__import__("torch").get_num_threads(),
        cache_hit=hit,
    )


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8
    slots = threading.BoundedSemaphore(4)

    def process_request(self, request, address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *args):
        pass

    def _send(self, status, value, html=False):
        body = value if html else json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header(
            "Content-Type", "text/html; charset=utf-8" if html else "application/json"
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _origin_ok(self):
        host = self.headers.get("Host", "")
        allowed = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
            f"{self.server.server_address[0]}:{self.server.server_port}",
        }
        return host in allowed and (
            not self.headers.get("Origin") or self.headers["Origin"] == f"http://{host}"
        )

    def do_GET(self):
        if not self._origin_ok():
            return self._send(403, {"error": "Origin or host not allowed"})
        if self.path == "/":
            return self._send(
                200, Path(__file__).with_name("ui_studio.html").read_bytes(), True
            )
        if self.path == "/api/domains":
            return self._send(
                200,
                {
                    "domains": [
                        dict(
                            domain_id=p.domain_id,
                            title=p.title,
                            description=p.description,
                            variables=[asdict(v) for v in p.policy.variables],
                            rules=[asdict(r) for r in p.policy.rules],
                            presets=p.presets,
                            preset_labels=p.preset_labels,
                        )
                        for p in _domains().values()
                    ]
                },
            )
        self._send(404, {"error": "Not found"})

    def do_POST(self):
        if not self._origin_ok():
            return self._send(403, {"error": "Origin or host not allowed"})
        routes = {
            "/api/evaluate": evaluate,
            "/api/policy-diff": policy_diff,
            "/api/batch-benchmark": batch_benchmark,
        }
        if self.path not in routes:
            return self._send(404, {"error": "Not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self._send(413, {"error": "JSON body exceeds limit or is empty"})
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Content-Type must be application/json")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("JSON object required")
            self._send(200, routes[self.path](data))
        except ValueError as exc:
            message = " ".join(str(exc).split())[:240]
            self._send(
                400,
                {"error": message or "Invalid policy, probability, or request options"},
            )
        except (TypeError, KeyError, RecursionError, OverflowError):
            self._send(
                400, {"error": "Invalid policy, probability, or request options"}
            )
        except Exception as exc:
            from .stage3_gpu_engine import ResourceLimitError

            self._send(
                422 if isinstance(exc, ResourceLimitError) else 500,
                {
                    "error": "Resource limit exceeded"
                    if isinstance(exc, ResourceLimitError)
                    else "Evaluation failed"
                },
            )


def create_server(host: str = "127.0.0.1", port: int = 8765) -> StudioServer:
    import os, torch

    torch.set_num_threads(
        max(1, min(8, int(os.environ.get("GUARDRAIL_TORCH_THREADS", "1"))))
    )
    return StudioServer((host, port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--demo-cli", action="store_true")
    args = parser.parse_args(argv)
    if args.demo_cli:
        import torch

        torch.set_num_threads(1)
        print("Policy Guardrail Studio — CPU demonstration")
        for domain, pack in _domains().items():
            print(f"\n{pack.title} [{domain}]")
            for preset, probs in pack.presets.items():
                result = evaluate(
                    {
                        "domain": domain,
                        "probs": probs,
                        "mode": "exact_rational",
                        "num_samples": 0,
                    }
                )
                telemetry = result["pruning_telemetry"]
                violations = (
                    ", ".join(r["rule_id"] for r in result["violated_rules"]) or "none"
                )
                flips = (
                    ", ".join(
                        f"{r['name']} {int(r['from'])}→{int(r['to'])}"
                        for r in result["repaired_variables"]
                    )
                    or "none"
                )
                print(f"  {pack.preset_labels.get(preset, preset)} [{preset}]")
                print(
                    f"    Models: {telemetry['valid_model_count']}/{telemetry['total_nominal_states']} | Z: {result['acceptance_prob']:.8g} ({result['exact_rational_z']}) | KL: {result['kl_divergence_nats']} nats"
                )
                print(f"    Violations: {violations} | MAP flips: {flips}")
                print(
                    f"    Compile: {result['compile_ms']:.3f} ms | Evaluate: {result['eval_ms']:.3f} ms | MAP: {result['map_ms']:.3f} ms | Total: {result['total_ms']:.3f} ms | Cache hit: {result['cache_hit']}"
                )
        return
    server = create_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_port}"
    print(f"Guardrail Studio: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
