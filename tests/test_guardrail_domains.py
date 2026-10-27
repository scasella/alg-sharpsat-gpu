"""G16: three complete editable domain packs and their nine scenario projections."""

from fractions import Fraction
from math import prod

import pytest

from alg_sharpsat.domains import get_domain, get_domains

pytestmark = pytest.mark.g16


@pytest.fixture(scope="module")
def domains():
    return get_domains()


@pytest.fixture(scope="module")
def engines(domains):
    return {name: pack.policy.compile_guardrail() for name, pack in domains.items()}


SCENARIOS = [
    ("cloud_ops_agent", "routine_canary"),
    ("cloud_ops_agent", "risky_db_migration"),
    ("cloud_ops_agent", "hallucinated_rogue_agent"),
    ("clinical_prescribing", "standard_afib_infection"),
    ("clinical_prescribing", "renal_anticoagulant_conflict"),
    ("clinical_prescribing", "critical_polypharmacy_hazard"),
    ("fintech_treasury", "compliant_rebalance"),
    ("fintech_treasury", "unverified_whale_bridge"),
    ("fintech_treasury", "sanctions_circuit_breaker_conflict"),
]


def test_pack_sizes_and_complete_presets(domains):
    assert {key: len(pack.policy.variables) for key, pack in domains.items()} == {
        "cloud_ops_agent": 20,
        "clinical_prescribing": 22,
        "fintech_treasury": 20,
    }
    for pack in domains.values():
        assert 12 <= len(pack.policy.rules) <= 20
        assert len(pack.presets) == 3
        assert len(set(pack.policy.variable_names)) == len(pack.policy.variables)
        assert set(pack.preset_labels) == set(pack.presets)
        for preset in pack.presets.values():
            assert set(preset) == set(pack.policy.variable_names)
            assert all(0 < p < 1 for p in preset.values())
    assert "not a validated" in domains["clinical_prescribing"].description
    assert "not validated" in domains["fintech_treasury"].description


@pytest.mark.parametrize("domain_id,preset_id", SCENARIOS)
def test_all_nine_exact_map_assignments_are_safe_and_positive(
    domains, engines, domain_id, preset_id
):
    pack = domains[domain_id]
    probabilities = pack.presets[preset_id]
    assignment, weight = engines[domain_id].map_inference(probabilities)
    assert assignment is not None
    assert isinstance(weight, Fraction)
    assert weight > 0
    assert len(assignment) == len(pack.policy.variable_names)
    named = dict(zip(pack.policy.variable_names, map(bool, assignment)))
    assert pack.policy.check_violations(named) == []
    # Auxiliary variables must be neutral: weight is exactly the original
    # independent Bernoulli product, with no hidden 1/2 factors.
    expected = prod(
        Fraction(str(probabilities[name]))
        if named[name]
        else 1 - Fraction(str(probabilities[name]))
        for name in pack.policy.variable_names
    )
    assert weight == expected
    mass = (
        engines[domain_id]
        .evaluate(probabilities, mode="exact_rational")
        .partition_function
    )
    assert 0 < weight <= mass <= 1


@pytest.mark.parametrize("domain_id,preset_id", SCENARIOS)
def test_raw_threshold_scenarios_distinguish_routine_from_conflict(
    domains, domain_id, preset_id
):
    pack = domains[domain_id]
    raw = {name: p >= 0.5 for name, p in pack.presets[preset_id].items()}
    violations = pack.policy.check_violations(raw)
    if preset_id in {
        "routine_canary",
        "standard_afib_infection",
        "compliant_rebalance",
    }:
        assert violations == []
    else:
        assert violations, (
            "Conflict and adversarial presets must expose a raw policy violation"
        )


def test_cloud_acceptance_bands(engines, domains):
    pack = domains["cloud_ops_agent"]
    masses = {
        key: float(engines[pack.domain_id].evaluate(p).partition_function)
        for key, p in pack.presets.items()
    }
    assert masses["routine_canary"] > 0.4
    assert 0.007 < masses["risky_db_migration"] < 0.014
    assert 0 < masses["hallucinated_rogue_agent"] < 1e-6


def test_domain_lookup_is_fresh_and_unknown_id_is_explicit():
    first = get_domain("cloud_ops_agent")
    first.presets["routine_canary"]["prod_environment"] = 0
    assert (
        get_domain("cloud_ops_agent").presets["routine_canary"]["prod_environment"] > 0
    )
    with pytest.raises(KeyError):
        get_domain("missing_domain")


@pytest.mark.parametrize(
    "operation", ["run_schema_migration", "drop_or_truncate_table"]
)
@pytest.mark.parametrize("path", ["planned", "breakglass"])
def test_production_database_mutations_accept_both_authorization_paths(
    domains, engines, operation, path
):
    pack = domains["cloud_ops_agent"]
    values = {name: False for name in pack.policy.variable_names}
    values.update(
        prod_environment=True,
        mfa_verified=True,
        fresh_snapshot_verified=True,
        rollback_script_tested=True,
        integration_tests_green=True,
        audit_log_streaming=True,
    )
    values[operation] = True
    if path == "planned":
        values.update(change_window_open=True, peer_review_approved=True)
        assert not values["emergency_breakglass_ticket"]
    else:
        values.update(incident_sev1_active=True, emergency_breakglass_ticket=True)
        assert not values["change_window_open"]
        assert not values["peer_review_approved"]
    assert pack.policy.check_violations(values) == []
    # Deterministic priors hold every contextual fact fixed: a valid tuple has
    # unit probability, so inference cannot rescue an incorrectly strict rule.
    assert (
        engines[pack.domain_id]
        .evaluate(
            {name: int(bit) for name, bit in values.items()}, mode="exact_rational"
        )
        .partition_function
        == 1
    )
    values["mfa_verified"] = False
    assert pack.policy.check_violations(values)


def test_variable_categories_reflect_domain_roles(domains):
    categories = {
        key: {v.name: v.category for v in pack.policy.variables}
        for key, pack in domains.items()
    }
    assert categories["cloud_ops_agent"]["drop_or_truncate_table"] == "database"
    assert categories["cloud_ops_agent"]["mfa_verified"] == "authorization"
    assert categories["cloud_ops_agent"]["prod_environment"] == "context"
    assert (
        categories["clinical_prescribing"]["severe_renal_impairment"] == "patient_state"
    )
    assert categories["clinical_prescribing"]["rx_warfarin"] == "prescriptions"
    assert categories["clinical_prescribing"]["order_inr_monitoring"] == "safeguards"
    assert categories["fintech_treasury"]["wire_over_250k"] == "transaction"
    assert categories["fintech_treasury"]["ofac_sanctions_cleared"] == "compliance"
    assert categories["fintech_treasury"]["hardware_enclave_cosign"] == "controls"


@pytest.mark.parametrize("prescription", ["rx_warfarin", "rx_ace_inhibitor"])
def test_combined_pregnancy_state_excludes_fixture_prescriptions(
    domains, engines, prescription
):
    pack = domains["clinical_prescribing"]
    values = {name: False for name in pack.policy.variable_names}
    values.update(
        pregnant_or_lactating=True,
        order_inr_monitoring=True,
        specialist_cardiology_consult=True,
    )
    values[prescription] = True
    assert any(
        r.rule_id == "pregnancy_exclusions"
        for r in pack.policy.check_violations(values)
    )
    assert (
        engines[pack.domain_id]
        .evaluate(
            {name: int(bit) for name, bit in values.items()}, mode="exact_rational"
        )
        .partition_function
        == 0
    )


def test_hyperkalemia_dual_therapy_requires_renal_panel(domains, engines):
    pack = domains["clinical_prescribing"]
    values = {name: False for name in pack.policy.variable_names}
    values.update(
        hyperkalemia_risk=True,
        rx_spironolactone=True,
        rx_ace_inhibitor=True,
        order_renal_panel_stat=True,
    )
    assert pack.policy.check_violations(values) == []
    assert (
        engines[pack.domain_id]
        .evaluate(
            {name: int(bit) for name, bit in values.items()}, mode="exact_rational"
        )
        .partition_function
        == 1
    )
    values["order_renal_panel_stat"] = False
    violations = {r.rule_id for r in pack.policy.check_violations(values)}
    assert {"potassium_dual_therapy", "hyperkalemia_spironolactone"} <= violations
    assert (
        engines[pack.domain_id]
        .evaluate(
            {name: int(bit) for name, bit in values.items()}, mode="exact_rational"
        )
        .partition_function
        == 0
    )
