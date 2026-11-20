"""
Unit and integration tests for Clima-Core Cognitive AI Assistant Service.
Verifies grounded prompt construction, zero-hallucination metric extraction,
multi-turn conversational support, and /api/assistant/chat endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from climacore.service.app import app
from climacore.service.assistant import (
    AssistantEngine,
    AssistantChatRequest,
    build_grounded_system_prompt,
    build_metrics_summary,
    generate_suggested_followups,
    generate_deterministic_synthesis,
)
from climacore.ingestion.engine import LiveForecastEngine


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_prediction():
    engine = LiveForecastEngine()
    return engine.predict(query="Mumbai", persona="student")


def test_build_metrics_summary_contains_essential_physics(sample_prediction):
    metrics = build_metrics_summary(sample_prediction, "student")
    assert metrics["city"] == "Mumbai"
    assert "temperature_max_c" in metrics
    assert "precipitation_mm" in metrics
    assert "conformal_intervals_90pct" in metrics
    assert "temperature_range" in metrics["conformal_intervals_90pct"]
    assert "rainfall_range" in metrics["conformal_intervals_90pct"]
    assert "pm25_range" in metrics["conformal_intervals_90pct"]


def test_build_grounded_system_prompt(sample_prediction):
    prompt = build_grounded_system_prompt(sample_prediction, "student")
    assert "Clima-Core Cognitive Environmental Risk" in prompt
    assert "Mumbai" in prompt
    assert "STUDENT" in prompt
    assert "STRICT PHYSICAL GROUNDING" in prompt
    assert "90% Conformal Temperature Range" in prompt


def test_generate_suggested_followups(sample_prediction):
    pills_student = generate_suggested_followups(sample_prediction, "student")
    assert len(pills_student) == 3
    assert any("school" in p.lower() or "umbrella" in p.lower() or "sports" in p.lower() for p in pills_student)

    pills_farmer = generate_suggested_followups(sample_prediction, "farmer")
    assert len(pills_farmer) == 3
    assert any("pesticide" in p.lower() or "irrigate" in p.lower() or "crop" in p.lower() for p in pills_farmer)


def test_deterministic_synthesis_answers_umbrella_query(sample_prediction):
    ans = generate_deterministic_synthesis("Should I carry an umbrella today?", sample_prediction, "student")
    assert isinstance(ans, str)
    assert len(ans) > 50
    assert "Rainfall Forecast" in ans or "umbrella" in ans.lower()


def test_api_assistant_chat_umbrella_query(api_client: TestClient):
    payload = {
        "message": "Should I carry an umbrella today?",
        "city": "Mumbai",
        "persona": "student",
        "history": [],
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert len(data["response"]) > 20
    assert data["city"] == "Mumbai"
    assert data["persona"] == "student"
    assert "temperature_max_c" in data["metrics_used"]
    assert len(data["suggested_followups"]) == 3
    assert data["latency_ms"] >= 0.0


def test_api_assistant_chat_farmer_pesticide_query(api_client: TestClient):
    payload = {
        "message": "Can I spray pesticide on my cotton crop tomorrow morning?",
        "city": "Nagpur",
        "persona": "farmer",
        "history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! I am your Clima-Core Agricultural Assistant."},
        ],
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["city"] == "Nagpur"
    assert data["persona"] == "farmer"
    assert any("spray" in p.lower() or "pesticide" in p.lower() for p in data["suggested_followups"])


def test_agent_what_if_simulation_cloudburst(api_client: TestClient):
    """Test autonomous counterfactual simulation of a 120mm rainfall cloudburst."""
    payload = {
        "message": "What if rainfall reaches 120mm in Mumbai tomorrow?",
        "city": "Mumbai",
        "persona": "commuter",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Counterfactual Simulation" in data["response"]
    assert "SCS Surface Runoff" in data["response"]
    assert "Hydrological Hazard Probability" in data["response"]
    assert len(data["tool_calls"]) > 0


def test_agent_what_if_simulation_heatwave(api_client: TestClient):
    """Test autonomous counterfactual simulation of a 46C extreme heatwave."""
    payload = {
        "message": "What if temperatures surge to 46C in Delhi?",
        "city": "Delhi",
        "persona": "city_official",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Counterfactual Simulation" in data["response"]
    assert "Wet-Bulb Temperature" in data["response"]
    assert "Power Grid" in data["response"]
    assert len(data["tool_calls"]) > 0


def test_agent_multi_city_comparison(api_client: TestClient):
    """Test multi-city route and comparative risk evaluation tool."""
    payload = {
        "message": "Compare Delhi and Jaipur for travel tomorrow",
        "city": "Delhi",
        "persona": "commuter",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "Comparative Environmental Risk" in data["response"]
    assert "Delhi" in data["response"]
    assert "Jaipur" in data["response"]
    assert len(data["tool_calls"]) > 0


def test_agent_ui_action_dispatch(api_client: TestClient):
    """Test frontend dashboard UI state control action dispatch."""
    payload = {
        "message": "Switch to Chennai as a farmer and show flood risk",
        "city": "Mumbai",
        "persona": "student",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] is not None
    assert data["action"]["type"] == "SET_DASHBOARD_STATE"
    assert data["action"]["target_city"] == "Chennai"
    assert data["action"]["target_persona"] == "farmer"
    assert "flood" in data["action"]["active_card"]

