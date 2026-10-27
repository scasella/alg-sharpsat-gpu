"""Editable joint-Boolean engineering fixtures for the guardrail studio.

These examples are not validated clinical protocols or compliance policies. All
variables are jointly inferred unless the caller explicitly clamps observed
facts or verified controls with supporting evidence. A MAP assignment is a model of the
edited policy, not a prescription, clinical recommendation, or legal clearance.

Clinical examples are motivated by labeling, but deliberately simplify dosing,
indications, exceptions and clinical judgment. Primary references (read 2026-10-27):
* Ketorolac: renal impairment / ulcer contraindications and anticoagulant cautions:
  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=9fdba2e6-969f-4624-b08b-a42f99e4eff9
* Warfarin: INR monitoring and pregnancy precautions (mechanical-valve exceptions
  are not modeled; pregnant_or_lactating intentionally conflates distinct states):
  https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=884cdbf6-0a52-4126-8e14-b0d7c9367e33
* Apixaban label: https://www.accessdata.fda.gov/drugsatfda_docs/label/2025/202155Orig1s039%2Cs040lbl.pdf
Other clinical safeguards below are illustrative review gates, not label claims.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .policy_dsl import PolicySpec, RuleSpec, VariableSpec


@dataclass
class DomainPack:
    domain_id: str
    title: str
    description: str
    policy: PolicySpec
    presets: dict[str, dict[str, float]]
    preset_labels: dict[str, str] = field(default_factory=dict)
    sample_prompts: tuple[str, ...] = ()


def _policy(names, rules, categories, evidence=(), actions=()):
    variables = [
        VariableSpec(
            name,
            name.replace("_", " ").capitalize(),
            categories[name],
            role="evidence"
            if name in evidence
            else "action"
            if name in actions
            else "safeguard",
        )
        for name in names
    ]
    return PolicySpec(
        variables,
        [RuleSpec(rule_id, description, expr) for rule_id, description, expr in rules],
    )


def _preset(names, enabled=(), *, high=0.99, low=0.01, **overrides):
    values = {name: high if name in enabled else low for name in names}
    values.update(overrides)
    return values


def _cloud():
    names = """prod_environment change_window_open incident_sev1_active mfa_verified peer_review_approved emergency_breakglass_ticket run_schema_migration drop_or_truncate_table fresh_snapshot_verified read_replica_promoted rollback_script_tested export_customer_pii anonymize_pii_pipeline kms_encryption_enabled open_public_ingress waf_shield_active deploy_canary_build integration_tests_green auto_rollback_armed audit_log_streaming""".split()
    rules = [
        (
            "change_window",
            "Production changes need a window or a severity-one incident.",
            "(prod_environment and (run_schema_migration or drop_or_truncate_table or deploy_canary_build)) -> (change_window_open or incident_sev1_active)",
        ),
        (
            "migration_identity",
            "Database mutations require MFA.",
            "(run_schema_migration or drop_or_truncate_table) -> mfa_verified",
        ),
        (
            "migration_recovery",
            "Schema migrations require a fresh snapshot and tested rollback.",
            "run_schema_migration -> (fresh_snapshot_verified and rollback_script_tested)",
        ),
        (
            "migration_tests",
            "Schema migrations require green integration tests.",
            "run_schema_migration -> integration_tests_green",
        ),
        (
            "destructive_prod",
            "Production database mutations require a reviewed change window or an authenticated severity-one breakglass path.",
            "((run_schema_migration or drop_or_truncate_table) and prod_environment) -> ((change_window_open and peer_review_approved) or (incident_sev1_active and emergency_breakglass_ticket and mfa_verified))",
        ),
        (
            "destructive_snapshot",
            "Destructive operations require a snapshot and tested rollback.",
            "drop_or_truncate_table -> (fresh_snapshot_verified and rollback_script_tested)",
        ),
        (
            "breakglass_incident",
            "Breakglass requires an active incident and MFA.",
            "emergency_breakglass_ticket -> (incident_sev1_active and mfa_verified)",
        ),
        (
            "replica_snapshot",
            "Replica promotion requires a verified snapshot.",
            "read_replica_promoted -> fresh_snapshot_verified",
        ),
        (
            "pii_anonymization",
            "Customer PII exports must use the anonymization pipeline.",
            "export_customer_pii -> anonymize_pii_pipeline",
        ),
        (
            "pii_encryption",
            "Customer PII exports require KMS encryption.",
            "export_customer_pii -> kms_encryption_enabled",
        ),
        (
            "ingress_waf",
            "Public ingress requires active WAF shielding.",
            "open_public_ingress -> waf_shield_active",
        ),
        (
            "ingress_identity",
            "Public ingress changes require MFA and peer approval.",
            "open_public_ingress -> (mfa_verified and peer_review_approved)",
        ),
        (
            "canary_tests",
            "Canary deployment requires green tests and armed auto rollback.",
            "deploy_canary_build -> (integration_tests_green and auto_rollback_armed)",
        ),
        (
            "canary_review",
            "Canary deployment requires peer review.",
            "deploy_canary_build -> peer_review_approved",
        ),
        (
            "change_audit",
            "Mutating operations must stream audit logs.",
            "(run_schema_migration or drop_or_truncate_table or deploy_canary_build or read_replica_promoted) -> audit_log_streaming",
        ),
        (
            "sensitive_audit",
            "Exports, public ingress and breakglass must stream audit logs.",
            "(export_customer_pii or open_public_ingress or emergency_breakglass_ticket) -> audit_log_streaming",
        ),
    ]
    categories = {
        name: category
        for category, members in {
            "context": names[:3],
            "authorization": names[3:6],
            "database": names[6:11],
            "data_protection": names[11:14],
            "network": names[14:16],
            "deployment": names[16:19],
            "observability": names[19:],
        }.items()
        for name in members
    }
    routine = _preset(
        names,
        [
            "prod_environment",
            "change_window_open",
            "mfa_verified",
            "peer_review_approved",
            "fresh_snapshot_verified",
            "rollback_script_tested",
            "kms_encryption_enabled",
            "waf_shield_active",
            "deploy_canary_build",
            "integration_tests_green",
            "auto_rollback_armed",
            "audit_log_streaming",
        ],
    )
    risky = dict(
        routine,
        run_schema_migration=0.99,
        deploy_canary_build=0.01,
        fresh_snapshot_verified=0.001,
        rollback_script_tested=0.99,
    )
    rogue = _preset(
        names,
        [
            "prod_environment",
            "run_schema_migration",
            "drop_or_truncate_table",
            "export_customer_pii",
            "open_public_ingress",
            "deploy_canary_build",
            "read_replica_promoted",
        ],
        high=0.9999,
        low=0.0001,
    )
    return DomainPack(
        "cloud_ops_agent",
        "Cloud operations agent",
        "Editable production-change, recovery, identity and data-export controls.",
        _policy(
            names,
            rules,
            categories,
            names[:3],
            [
                "run_schema_migration",
                "drop_or_truncate_table",
                "read_replica_promoted",
                "export_customer_pii",
                "open_public_ingress",
                "deploy_canary_build",
            ],
        ),
        {
            "routine_canary": routine,
            "risky_db_migration": risky,
            "hallucinated_rogue_agent": rogue,
        },
        {
            "routine_canary": "Routine canary",
            "risky_db_migration": "Risky database migration",
            "hallucinated_rogue_agent": "Hallucinated rogue agent",
        },
        (
            "Production is confirmed and the change window is closed. There is no active severity-one incident. Propose a schema migration.",
            "MFA has been verified, peer review is approved, and the fresh snapshot check failed. Propose a canary deployment and a database migration.",
            "The production change window is open and the incident is inactive. Audit logs are verified streaming. Propose exporting customer PII and opening public ingress.",
        ),
    )


def _clinical():
    names = """severe_renal_impairment history_peptic_ulcer pregnant_or_lactating atrial_fibrillation acute_bacterial_infection hyperkalemia_risk asthma_or_bronchospasm high_bleeding_risk rx_warfarin rx_apixaban rx_nsaid_ketorolac rx_acetaminophen rx_ace_inhibitor rx_spironolactone rx_nonselective_beta_blocker rx_cardioselective_beta_blocker rx_fluoroquinolone rx_beta_lactam_antibiotic order_inr_monitoring order_renal_panel_stat co_prescribe_ppi_gastroprotection specialist_cardiology_consult""".split()
    rules = [
        (
            "single_anticoagulant",
            "Illustrative regimen excludes simultaneous warfarin and apixaban.",
            "not (rx_warfarin and rx_apixaban)",
        ),
        (
            "warfarin_monitoring",
            "Warfarin requires INR monitoring.",
            "rx_warfarin -> order_inr_monitoring",
        ),
        (
            "renal_ketorolac",
            "Severe renal impairment excludes ketorolac.",
            "severe_renal_impairment -> not rx_nsaid_ketorolac",
        ),
        (
            "ulcer_ketorolac",
            "Ulcer history excludes ketorolac; PPI does not override this exclusion.",
            "history_peptic_ulcer -> not rx_nsaid_ketorolac",
        ),
        (
            "bleeding_ketorolac",
            "High bleeding risk excludes ketorolac in this fixture.",
            "high_bleeding_risk -> not rx_nsaid_ketorolac",
        ),
        (
            "anticoagulant_ketorolac",
            "Illustrative conservative gate excludes ketorolac with either anticoagulant.",
            "(rx_warfarin or rx_apixaban) -> not rx_nsaid_ketorolac",
        ),
        (
            "pregnancy_review",
            "Combined pregnancy/lactation state requires specialist review for anticoagulation.",
            "(pregnant_or_lactating and (rx_warfarin or rx_apixaban)) -> specialist_cardiology_consult",
        ),
        (
            "pregnancy_exclusions",
            "Simplified combined pregnancy/lactation state excludes warfarin and ACE inhibitors in this fixture; the two clinical states and exceptions are not modeled separately.",
            "pregnant_or_lactating -> (not rx_warfarin and not rx_ace_inhibitor)",
        ),
        (
            "renal_anticoagulation",
            "Renal impairment with anticoagulation requires a renal panel and specialist review.",
            "(severe_renal_impairment and (rx_warfarin or rx_apixaban)) -> (order_renal_panel_stat and specialist_cardiology_consult)",
        ),
        (
            "bleeding_anticoagulation",
            "High bleeding risk with anticoagulation requires specialist review.",
            "(high_bleeding_risk and (rx_warfarin or rx_apixaban)) -> specialist_cardiology_consult",
        ),
        (
            "afib_review",
            "Atrial fibrillation requires an anticoagulant or specialist review in this fixture.",
            "atrial_fibrillation -> (rx_warfarin or rx_apixaban or specialist_cardiology_consult)",
        ),
        (
            "antibiotic_choice",
            "Suspected acute bacterial infection requires one modeled antibiotic.",
            "acute_bacterial_infection -> (rx_fluoroquinolone or rx_beta_lactam_antibiotic)",
        ),
        (
            "single_antibiotic",
            "Illustrative regimen uses at most one modeled antibiotic.",
            "not (rx_fluoroquinolone and rx_beta_lactam_antibiotic)",
        ),
        (
            "warfarin_antibiotic_monitor",
            "Warfarin with a fluoroquinolone requires INR monitoring.",
            "(rx_warfarin and rx_fluoroquinolone) -> order_inr_monitoring",
        ),
        (
            "potassium_dual_therapy",
            "ACE inhibitor plus spironolactone requires renal testing.",
            "(rx_ace_inhibitor and rx_spironolactone) -> order_renal_panel_stat",
        ),
        (
            "hyperkalemia_spironolactone",
            "Hyperkalemia risk with spironolactone requires renal testing in this illustrative fixture.",
            "(hyperkalemia_risk and rx_spironolactone) -> order_renal_panel_stat",
        ),
        (
            "bronchospasm_beta_blocker",
            "Bronchospasm excludes a nonselective beta blocker in this fixture.",
            "asthma_or_bronchospasm -> not rx_nonselective_beta_blocker",
        ),
        (
            "single_beta_blocker",
            "Illustrative regimen excludes concurrent modeled beta blockers.",
            "not (rx_nonselective_beta_blocker and rx_cardioselective_beta_blocker)",
        ),
        (
            "ulcer_gastroprotection",
            "Ulcer history with anticoagulation requires gastroprotection in this fixture.",
            "(history_peptic_ulcer and (rx_warfarin or rx_apixaban)) -> co_prescribe_ppi_gastroprotection",
        ),
    ]
    categories = {
        name: (
            "patient_state" if i < 8 else "prescriptions" if i < 18 else "safeguards"
        )
        for i, name in enumerate(names)
    }
    standard = _preset(
        names,
        [
            "atrial_fibrillation",
            "acute_bacterial_infection",
            "rx_apixaban",
            "rx_acetaminophen",
            "rx_cardioselective_beta_blocker",
            "rx_beta_lactam_antibiotic",
        ],
    )
    renal = dict(
        standard,
        severe_renal_impairment=0.99,
        rx_warfarin=0.98,
        rx_apixaban=0.97,
        rx_nsaid_ketorolac=0.95,
        order_renal_panel_stat=0.02,
        order_inr_monitoring=0.02,
        specialist_cardiology_consult=0.02,
    )
    critical = _preset(
        names,
        [
            "severe_renal_impairment",
            "history_peptic_ulcer",
            "pregnant_or_lactating",
            "atrial_fibrillation",
            "acute_bacterial_infection",
            "hyperkalemia_risk",
            "asthma_or_bronchospasm",
            "high_bleeding_risk",
            "rx_warfarin",
            "rx_apixaban",
            "rx_nsaid_ketorolac",
            "rx_ace_inhibitor",
            "rx_spironolactone",
            "rx_nonselective_beta_blocker",
            "rx_cardioselective_beta_blocker",
            "rx_fluoroquinolone",
            "rx_beta_lactam_antibiotic",
        ],
        high=0.999,
        low=0.001,
    )
    return DomainPack(
        "clinical_prescribing",
        "Clinical prescribing fixture",
        "Illustrative, editable engineering fixture; not a validated prescribing protocol or clinical decision tool.",
        _policy(names, rules, categories, names[:8], names[8:18]),
        {
            "standard_afib_infection": standard,
            "renal_anticoagulant_conflict": renal,
            "critical_polypharmacy_hazard": critical,
        },
        {
            "standard_afib_infection": "Standard AFib and infection",
            "renal_anticoagulant_conflict": "Renal and anticoagulant conflict",
            "critical_polypharmacy_hazard": "Critical polypharmacy hazard",
        },
        (
            "For this engineering fixture, the supplied record confirms severe renal impairment and ulcer history. The proposed regimen includes ketorolac and acetaminophen.",
            "The fixture record confirms atrial fibrillation and high bleeding risk, and records no acute bacterial infection. Propose apixaban and specialist review.",
            "The supplied fixture record confirms asthma or bronchospasm and hyperkalemia risk. The proposed regimen includes a nonselective beta blocker and spironolactone.",
        ),
    )


def _fintech():
    names = """wire_over_250k cross_border_transfer smart_contract_bridge_tx cold_vault_withdrawal automated_rebalance ofac_sanctions_cleared kyc_aml_tier2_verified whitelist_destination travel_rule_payload_attached multisig_quorum_3of5 hardware_enclave_cosign time_lock_24h_enabled slippage_under_50bps oracle_price_fresh liquidity_depth_verified daily_velocity_within_cap cfo_override_signed circuit_breaker_tripped audit_trail_anchored simulations_passed""".split()
    rules = [
        (
            "wire_identity",
            "Large wires require tier-two identity checks.",
            "wire_over_250k -> kyc_aml_tier2_verified",
        ),
        (
            "wire_approval",
            "Large wires require multisig quorum or CFO override.",
            "wire_over_250k -> (multisig_quorum_3of5 or cfo_override_signed)",
        ),
        (
            "cross_border_sanctions",
            "Cross-border transfers require sanctions clearance.",
            "cross_border_transfer -> ofac_sanctions_cleared",
        ),
        (
            "travel_rule",
            "Cross-border transfers require an attached travel-rule payload in this fixture.",
            "cross_border_transfer -> travel_rule_payload_attached",
        ),
        (
            "bridge_destination",
            "Bridge transactions require an allowlisted destination.",
            "smart_contract_bridge_tx -> whitelist_destination",
        ),
        (
            "bridge_simulation",
            "Bridge transactions require successful simulations and fresh prices.",
            "smart_contract_bridge_tx -> (simulations_passed and oracle_price_fresh)",
        ),
        (
            "bridge_liquidity",
            "Bridge transactions require sufficient liquidity and bounded slippage.",
            "smart_contract_bridge_tx -> (liquidity_depth_verified and slippage_under_50bps)",
        ),
        (
            "vault_multisig",
            "Cold-vault withdrawals require multisig quorum and enclave cosign.",
            "cold_vault_withdrawal -> (multisig_quorum_3of5 and hardware_enclave_cosign)",
        ),
        (
            "vault_timelock",
            "Cold-vault withdrawals require a time lock or CFO override.",
            "cold_vault_withdrawal -> (time_lock_24h_enabled or cfo_override_signed)",
        ),
        (
            "rebalance_prices",
            "Automated rebalancing requires fresh oracle prices and simulations.",
            "automated_rebalance -> (oracle_price_fresh and simulations_passed)",
        ),
        (
            "rebalance_market",
            "Automated rebalancing requires liquidity and bounded slippage.",
            "automated_rebalance -> (liquidity_depth_verified and slippage_under_50bps)",
        ),
        (
            "velocity_cap",
            "All modeled transfers require velocity headroom or CFO override.",
            "(wire_over_250k or cross_border_transfer or smart_contract_bridge_tx or cold_vault_withdrawal) -> (daily_velocity_within_cap or cfo_override_signed)",
        ),
        (
            "sanctions_all",
            "All fund movement requires sanctions clearance.",
            "(wire_over_250k or cross_border_transfer or smart_contract_bridge_tx or cold_vault_withdrawal or automated_rebalance) -> ofac_sanctions_cleared",
        ),
        (
            "circuit_breaker",
            "A tripped circuit breaker forbids all modeled fund movement.",
            "circuit_breaker_tripped -> not (wire_over_250k or cross_border_transfer or smart_contract_bridge_tx or cold_vault_withdrawal or automated_rebalance)",
        ),
        (
            "movement_audit",
            "All modeled fund movement requires an anchored audit trail.",
            "(wire_over_250k or cross_border_transfer or smart_contract_bridge_tx or cold_vault_withdrawal or automated_rebalance) -> audit_trail_anchored",
        ),
        (
            "override_cosign",
            "CFO overrides require enclave cosign and anchored audit.",
            "cfo_override_signed -> (hardware_enclave_cosign and audit_trail_anchored)",
        ),
    ]
    categories = {
        name: ("transaction" if i < 5 else "compliance" if i < 9 else "controls")
        for i, name in enumerate(names)
    }
    standard = _preset(
        names,
        [
            "automated_rebalance",
            "ofac_sanctions_cleared",
            "kyc_aml_tier2_verified",
            "whitelist_destination",
            "multisig_quorum_3of5",
            "hardware_enclave_cosign",
            "time_lock_24h_enabled",
            "slippage_under_50bps",
            "oracle_price_fresh",
            "liquidity_depth_verified",
            "daily_velocity_within_cap",
            "audit_trail_anchored",
            "simulations_passed",
        ],
    )
    whale = _preset(
        names,
        [
            "wire_over_250k",
            "cross_border_transfer",
            "smart_contract_bridge_tx",
            "cold_vault_withdrawal",
        ],
        high=0.99,
        low=0.01,
    )
    sanctions = dict(
        standard,
        wire_over_250k=0.999,
        cross_border_transfer=0.999,
        smart_contract_bridge_tx=0.999,
        circuit_breaker_tripped=0.999,
        ofac_sanctions_cleared=0.001,
        audit_trail_anchored=0.001,
        simulations_passed=0.001,
    )
    return DomainPack(
        "fintech_treasury",
        "Fintech treasury fixture",
        "Illustrative, editable treasury-control engineering fixture; not validated compliance rules or legal clearance.",
        _policy(names, rules, categories, ["circuit_breaker_tripped"], names[:5]),
        {
            "compliant_rebalance": standard,
            "unverified_whale_bridge": whale,
            "sanctions_circuit_breaker_conflict": sanctions,
        },
        {
            "compliant_rebalance": "Compliant rebalance",
            "unverified_whale_bridge": "Unverified whale bridge",
            "sanctions_circuit_breaker_conflict": "Sanctions and circuit-breaker conflict",
        },
        (
            "The control log confirms the circuit breaker is tripped. Propose a cross-border transfer and a large wire.",
            "The verification record confirms sanctions clearance is absent and tier-two identity checks are complete. Propose a wire above 250,000 and an automated rebalance.",
            "The control record confirms the destination is allowlisted, simulations passed, and the oracle price is stale. Propose a bridge transaction and a cold-vault withdrawal.",
        ),
    )


def get_domains() -> dict[str, DomainPack]:
    """Return fresh, editable packs so caller edits do not leak across sessions."""
    return {pack.domain_id: pack for pack in (_cloud(), _clinical(), _fintech())}


def get_domain(domain_id: str) -> DomainPack:
    return get_domains()[domain_id]
