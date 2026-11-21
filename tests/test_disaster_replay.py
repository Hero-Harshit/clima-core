"""
Unit and Integration Tests for Historical Disaster Replay Time Machine & Catastrophe Backtester.
Verifies predictive model lead times, parametric insurance triggers, and civil defense
mitigation against landmark Indian climate catastrophes.
"""

import pytest
from fastapi.testclient import TestClient

from climacore.service.app import app
from climacore.analytics.disaster_replay import list_historical_disasters, run_disaster_replay
from climacore.ingestion.engine import LiveForecastEngine
from climacore.service.agent_tools import AgentToolExecutor


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def tool_executor() -> AgentToolExecutor:
    return AgentToolExecutor(live_engine=LiveForecastEngine())


def test_list_historical_disasters():
    """Verify that catalog returns all landmark historical catastrophe benchmarks."""
    events = list_historical_disasters()
    assert len(events) >= 3

    event_ids = [e["disaster_id"] for e in events]
    assert "MUMBAI_2005_CLOUDBURST" in event_ids
    assert "CHENNAI_2015_DELUGE" in event_ids
    assert "NORTH_INDIA_2022_HEATWAVE" in event_ids

    # Verify metadata fields
    for ev in events:
        assert "name" in ev
        assert "city" in ev
        assert "category" in ev
        assert ev["total_advance_warning_hours"] > 0
        assert ev["simulated_parametric_payout_inr_cr"] > 0
        assert ev["lives_protected_estimate"] > 0


def test_run_disaster_replay_mumbai():
    """Verify full backtest replay for 2005 Mumbai 944mm cloudburst."""
    replay = run_disaster_replay("MUMBAI_2005_CLOUDBURST")

    assert replay.disaster_id == "MUMBAI_2005_CLOUDBURST"
    assert replay.city == "Mumbai"
    assert replay.category == "URBAN_DELUGE"
    assert "944.2 mm" in replay.historical_peak_metric
    assert replay.total_advance_warning_hours == 36.0
    assert replay.simulated_parametric_payout_inr_cr == 145.0
    assert replay.lives_protected_estimate >= 400

    # Verify chronological phases
    assert len(replay.timeline_phases) == 4
    leads = [p.lead_time_hours for p in replay.timeline_phases]
    assert leads == [-48, -24, -6, 0]

    # Check phase statuses
    statuses = [p.civil_defense_status for p in replay.timeline_phases]
    assert "WATCH" in statuses
    assert "WARNING" in statuses
    assert "EVACUATION_MANDATE" in statuses
    assert "DISASTER_PEAK" in statuses

    # Verify Mermaid timeline chart
    assert "graph LR" in replay.mermaid_timeline_chart
    assert "-->" in replay.mermaid_timeline_chart


def test_run_disaster_replay_delhi_heatwave():
    """Verify backtest replay for 2022 North India 49.2C record heatwave."""
    replay = run_disaster_replay("NORTH_INDIA_2022_HEATWAVE")

    assert replay.disaster_id == "NORTH_INDIA_2022_HEATWAVE"
    assert replay.category == "SEVERE_HEATWAVE"
    assert "49.2°C" in replay.historical_peak_metric
    assert replay.total_advance_warning_hours == 72.0
    assert replay.simulated_parametric_payout_inr_cr >= 200.0
    assert replay.lives_protected_estimate >= 1000

    # Verify wheat terminal heat PMFBY trigger
    p24 = next(p for p in replay.timeline_phases if p.lead_time_hours == -24)
    all_actions = " ".join(p24.autonomous_actions_taken)
    assert "PMFBY" in all_actions or "Parametric" in all_actions
    assert "DBT" in all_actions or "payout" in all_actions.lower()


def test_run_disaster_replay_tool_executor(tool_executor: AgentToolExecutor):
    """Verify AgentToolExecutor executes replay_historical_disaster."""
    res = tool_executor.replay_historical_disaster(disaster_id="CHENNAI_2015_DELUGE")

    assert res["disaster_id"] == "CHENNAI_2015_DELUGE"
    assert res["city"] == "Chennai"
    assert len(res["timeline_phases"]) >= 3
    assert res["total_advance_warning_hours"] >= 40.0
    assert "graph LR" in res["mermaid_timeline_chart"]


def test_api_disaster_replay_events_endpoint(api_client: TestClient):
    """Verify GET /api/disaster-replay/events."""
    resp = api_client.get("/api/disaster-replay/events")
    assert resp.status_code == 200
    data = resp.json()

    assert "total_events" in data
    assert data["total_events"] >= 3
    assert len(data["events"]) >= 3


def test_api_disaster_replay_run_endpoint(api_client: TestClient):
    """Verify POST /api/disaster-replay/run."""
    payload = {"disaster_id": "MUMBAI_2005_CLOUDBURST"}
    resp = api_client.post("/api/disaster-replay/run", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["disaster_id"] == "MUMBAI_2005_CLOUDBURST"
    assert data["total_advance_warning_hours"] == 36.0
    assert "timeline_phases" in data
    assert "latency_ms" in data


def test_api_assistant_chat_disaster_replay_query(api_client: TestClient):
    """Verify conversational query requesting disaster replay returns complete backtest dossier."""
    payload = {
        "message": "Run the disaster replay time machine for the 2005 Mumbai floods",
        "city": "Mumbai",
        "persona": "city_official",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "Disaster Replay Time Machine" in data["response"]
    assert "2005" in data["response"]
    assert "Advance Warning Delivered" in data["response"]
    assert "Parametric DBT Claims" in data["response"]
    assert "mermaid" in data["response"]
    assert len(data["tool_calls"]) > 0
