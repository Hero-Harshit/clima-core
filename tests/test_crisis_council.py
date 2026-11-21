"""
Unit and Integration Tests for Autonomous Multi-Agent Crisis Council (Synthetic ICS War Room).
Verifies multi-sector authority deliberation across Public Health, Power Grid, and
Cold-Chain Logistics with Incident Command (ICS-201) compromise directives.
"""

import pytest
from fastapi.testclient import TestClient

from climacore.service.app import app
from climacore.analytics.crisis_council import simulate_crisis_council_deliberation
from climacore.ingestion.engine import LiveForecastEngine
from climacore.service.agent_tools import AgentToolExecutor


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def tool_executor() -> AgentToolExecutor:
    return AgentToolExecutor(live_engine=LiveForecastEngine())


def test_simulate_crisis_council_heatwave():
    """Verify multi-agent war-room deliberation under a 45C heatwave with 29.5C wet-bulb."""
    council = simulate_crisis_council_deliberation(
        city="Delhi",
        temp_max_c=45.0,
        temp_min_c=29.0,
        rainfall_mm=0.0,
        wet_bulb_c=29.5,
        peak_grid_load_mw=9500.0,
    )

    assert council.city == "Delhi"
    assert council.incident_type == "SEVERE_HEATWAVE_GRID_COLLAPSE"
    assert council.severity_tier in ["CRITICAL", "HIGH"]
    assert "ICS-201-DEL-" in council.incident_id
    assert len(council.verification_hash) == 16
    assert council.life_preservation_index >= 90.0
    assert council.economic_continuity_index >= 75.0

    # Verify sector authority proposals
    member_ids = [m.member_id for m in council.member_proposals]
    assert "public_health_officer" in member_ids
    assert "grid_operations_chief" in member_ids
    assert "coldchain_logistics_director" in member_ids

    # Verify friction matrix
    assert "Health_vs_Grid" in council.friction_matrix
    assert "Health_vs_Logistics" in council.friction_matrix
    assert "Grid_vs_Logistics" in council.friction_matrix

    # Verify executive directives
    directive_ids = [d.directive_id for d in council.executive_directives]
    assert "DIR-HEAT-01" in directive_ids
    assert "DIR-GRID-02" in directive_ids
    assert "DIR-LOG-03" in directive_ids

    # Verify Mermaid sequence diagram syntax
    assert "sequenceDiagram" in council.mermaid_sequence_diagram
    assert "actor H as NDMA Public Health Officer" in council.mermaid_sequence_diagram
    assert "actor G as SLDC Grid Operations Chief" in council.mermaid_sequence_diagram


def test_simulate_crisis_council_cloudburst():
    """Verify multi-agent deliberation under 125mm urban cloudburst."""
    council = simulate_crisis_council_deliberation(
        city="Mumbai",
        temp_max_c=31.0,
        temp_min_c=25.0,
        rainfall_mm=125.0,
    )

    assert council.city == "Mumbai"
    assert council.incident_type == "URBAN_CLOUDBURST_DELUGE"
    assert council.severity_tier == "CRITICAL"
    assert "ICS-201-MUM-" in council.incident_id

    # Verify flood directives
    directive_ids = [d.directive_id for d in council.executive_directives]
    assert "DIR-FLOOD-01" in directive_ids
    assert "DIR-FLOOD-02" in directive_ids
    assert "DIR-FLOOD-03" in directive_ids

    # Check that dewatering pump protection and underpass barricades are addressed
    all_mandates = " ".join([d.mandate for d in council.executive_directives])
    assert "pumping" in all_mandates.lower()
    assert "underpass" in all_mandates.lower()


def test_simulate_crisis_council_tool_executor(tool_executor: AgentToolExecutor):
    """Verify AgentToolExecutor correctly executes convene_crisis_council."""
    res = tool_executor.convene_crisis_council(
        city="Jaipur",
        temp_max_c=46.0,
        rainfall_mm=0.0,
    )

    assert res["city"] == "Jaipur"
    assert res["incident_type"] == "SEVERE_HEATWAVE_GRID_COLLAPSE"
    assert len(res["member_proposals"]) == 3
    assert len(res["executive_directives"]) >= 3
    assert "sequenceDiagram" in res["mermaid_sequence_diagram"]


def test_api_crisis_council_endpoint(api_client: TestClient):
    """Verify dedicated POST /api/crisis-council/deliberate endpoint."""
    payload = {
        "city": "Delhi",
        "temp_max_c": 44.5,
        "temp_min_c": 28.5,
        "rainfall_mm": 0.0,
        "wet_bulb_c": 29.0,
    }
    resp = api_client.post("/api/crisis-council/deliberate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["city"] == "Delhi"
    assert data["incident_type"] == "SEVERE_HEATWAVE_GRID_COLLAPSE"
    assert "ICS-201-" in data["incident_id"]
    assert "sequenceDiagram" in data["mermaid_sequence_diagram"]
    assert len(data["executive_directives"]) >= 3
    assert "latency_ms" in data


def test_api_assistant_chat_crisis_council_query(api_client: TestClient):
    """Verify conversational query requesting Crisis Council war room debate returns full synthesis."""
    payload = {
        "message": "Convene the crisis council war room to resolve conflict between health and grid under a 46C heatwave in Delhi",
        "city": "Delhi",
        "persona": "city_official",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "Autonomous Multi-Agent Crisis Council" in data["response"]
    assert "ICS-201" in data["response"]
    assert "sequenceDiagram" in data["response"]
    assert "Executive Incident Commander Directives" in data["response"]
    assert len(data["tool_calls"]) > 0
