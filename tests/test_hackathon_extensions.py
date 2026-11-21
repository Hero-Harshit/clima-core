"""
Unit and Integration Tests for Hackathon-Winning Cognitive Assistant Capabilities:
1. Autonomous Parametric Climate Insurance & Settlement Voucher Engine (Climate-FinTech)
2. Compound Domino Cascade Simulator (Systems Physics)
3. Interactive TreeSHAP Feature Attribution on Demand (Explainable AI)
"""

import pytest
from fastapi.testclient import TestClient

from climacore.service.app import app
from climacore.ingestion.engine import LiveForecastEngine
from climacore.service.agent_tools import AgentToolExecutor
from climacore.analytics.parametric_insurance import evaluate_parametric_crop_claim
from climacore.analytics.cascade_simulator import simulate_compound_cascade


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def tool_executor() -> AgentToolExecutor:
    return AgentToolExecutor(live_engine=LiveForecastEngine())


# =============================================================================
# 1. PARAMETRIC CLIMATE INSURANCE TESTS
# =============================================================================
def test_evaluate_parametric_crop_claim_wheat_heat_trigger():
    """Verify that 35C terminal heat during rabi grain-filling triggers 50% payout tier."""
    voucher = evaluate_parametric_crop_claim(
        city="Jaipur",
        crop="wheat",
        hectares=2.5,
        temperature_max_c=35.0,
        temperature_min_c=20.0,
        precipitation_mm=0.0,
        conformal_lower_bound=33.5,
    )

    assert voucher.payout_status == "APPROVED_INSTANT_DISBURSEMENT"
    assert voucher.crop == "Rabi Wheat"
    assert voucher.season == "Rabi"
    assert voucher.payout_percentage == 50.0
    # 2.5 ha * 65,000 INR * 0.50 = 81,250 INR
    assert voucher.net_claim_payout_inr == 81250.0
    assert len(voucher.breached_triggers) >= 1
    assert "PMFBY-WHE-JAI" in voucher.voucher_id
    assert len(voucher.verification_hash) == 16
    assert "90% Conformal Coverage guarantee" in voucher.statistical_guarantee


def test_evaluate_parametric_crop_claim_cotton_deluge_trigger():
    """Verify that 130mm deluge rainfall triggers 70% payout tier for Bt Cotton."""
    voucher = evaluate_parametric_crop_claim(
        city="Nagpur",
        crop="cotton",
        hectares=1.0,
        temperature_max_c=32.0,
        precipitation_mm=130.0,
    )

    assert voucher.payout_status == "APPROVED_INSTANT_DISBURSEMENT"
    assert voucher.crop == "Bt Cotton"
    assert voucher.payout_percentage == 70.0
    # 1.0 ha * 75,000 INR * 0.70 = 52,500 INR
    assert voucher.net_claim_payout_inr == 52500.0
    assert any("rootzone_waterlogging" in t["trigger_type"] for t in voucher.breached_triggers)


def test_evaluate_parametric_crop_claim_no_breach():
    """Verify that safe conditions yield NO_BREACH_DETECTED and zero payout."""
    voucher = evaluate_parametric_crop_claim(
        city="Amritsar",
        crop="wheat",
        hectares=1.0,
        temperature_max_c=26.0,
        precipitation_mm=5.0,
    )

    assert voucher.payout_status == "NO_BREACH_DETECTED"
    assert voucher.payout_percentage == 0.0
    assert voucher.net_claim_payout_inr == 0.0
    assert len(voucher.breached_triggers) == 0


# =============================================================================
# 2. COMPOUND DOMINO CASCADE SIMULATOR TESTS
# =============================================================================
def test_simulate_compound_domino_cascade_heatwave():
    """Verify multi-tier failure propagation under a 45C severe heatwave with 29C night."""
    cascade = simulate_compound_cascade(
        city="Delhi",
        temp_max_c=45.0,
        temp_min_c=29.0,
        rainfall_mm=0.0,
        wet_bulb_c=29.5,
    )

    assert cascade.severity_tier in ["CRITICAL", "ELEVATED"]
    assert cascade.total_systemic_vulnerability_score >= 60.0
    assert len(cascade.nodes) >= 4

    sectors = [n.sector for n in cascade.nodes]
    assert "Meteorology" in sectors
    assert any("Grid" in s for s in sectors)
    assert any("Cold Chain" in s for s in sectors)
    assert any("Public Health" in s for s in sectors)

    # Verify Mermaid diagram syntax
    assert "graph TD" in cascade.mermaid_diagram
    assert "-->" in cascade.mermaid_diagram

    # Verify emergency civil defense interventions
    assert len(cascade.emergency_interventions) >= 2


def test_simulate_compound_domino_cascade_cloudburst():
    """Verify urban deluge cascade under 110mm rainfall."""
    cascade = simulate_compound_cascade(
        city="Mumbai",
        temp_max_c=31.0,
        temp_min_c=25.0,
        rainfall_mm=110.0,
    )

    assert len(cascade.nodes) >= 3
    sectors = [n.sector for n in cascade.nodes]
    assert any("Hydrology" in s for s in sectors)
    assert any("Drainage" in s for s in sectors)
    assert any("Transit" in s for s in sectors)
    assert any("subway underpasses" in act for act in cascade.emergency_interventions)


# =============================================================================
# 3. INTERACTIVE TREESHAP ATTRIBUTION TESTS
# =============================================================================
def test_explain_risk_attribution_shap_flood(tool_executor: AgentToolExecutor):
    """Verify native C++ TreeSHAP explanation for flood risk model."""
    res = tool_executor.explain_risk_attribution_shap(city="Mumbai", target_risk="flood")

    assert res["city"] == "Mumbai"
    assert res["model_id"] == "flood_risk_v1"
    assert res["target_unit"] == "log-odds"
    assert "top_amplifying_features" in res
    assert len(res["top_amplifying_features"]) > 0
    assert "top_mitigating_features" in res
    assert "disclaimer" in res
    assert "e-" in res["additivity_error"] or float(res["additivity_error"]) < 1e-5


# =============================================================================
# 4. END-TO-END /api/assistant/chat INTEGRATION TESTS
# =============================================================================
def test_api_assistant_chat_parametric_insurance_query(api_client: TestClient):
    """Verify conversational query requesting PMFBY insurance payout returns verifiable voucher."""
    payload = {
        "message": "Can I claim PMFBY insurance compensation for wheat heat stress if temp reaches 35C in Jaipur?",
        "city": "Jaipur",
        "persona": "farmer",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "Parametric Insurance Settlement Voucher" in data["response"]
    assert "PMFBY-" in data["response"]
    assert "SHA-256" in data["response"] or "Hash" in data["response"]
    assert "APPROVED_INSTANT_DISBURSEMENT" in data["response"] or "Settlement" in data["response"]
    assert len(data["tool_calls"]) > 0


def test_api_assistant_chat_domino_cascade_query(api_client: TestClient):
    """Verify conversational query requesting compound domino cascade returns mermaid diagram."""
    payload = {
        "message": "Simulate a compound domino cascade if temperatures reach 45C in Delhi",
        "city": "Delhi",
        "persona": "city_official",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "Domino Cascade" in data["response"]
    assert "mermaid" in data["response"]
    assert "Sequential Failure Propagation" in data["response"]
    assert len(data["tool_calls"]) > 0


def test_api_assistant_chat_shap_explainability_query(api_client: TestClient):
    """Verify conversational query requesting SHAP explainability returns additive feature attributions."""
    payload = {
        "message": "Why is the flood risk predicted? Explain what factors drive the risk score in Mumbai",
        "city": "Mumbai",
        "persona": "commuter",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "TreeSHAP Mathematical Risk Attribution" in data["response"]
    assert "baseline_value_phi0" in str(data["tool_calls"]) or "Baseline Expectation" in data["response"]
    assert "Top Amplifying Risk Drivers" in data["response"]
    assert "Additivity Error" in data["response"]
    assert len(data["tool_calls"]) > 0
