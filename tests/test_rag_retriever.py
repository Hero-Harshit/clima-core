"""
Tests for Institutional Indian Environmental & Regulatory Knowledge Base and Serverless RAG Retriever.
Verifies zero-database dense semantic retrieval across NDMA, ICAR-CRIDA, BIS IS 10500:2012,
CPCB, CEA, and IMD/CWC standards.
"""

import pytest
from fastapi.testclient import TestClient

from climacore.service.app import app
from climacore.analytics.rag_retriever import (
    load_knowledge_base,
    retrieve_relevant_standards,
    format_retrieved_standards_for_prompt,
)


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


def test_load_knowledge_base_manifest_and_embeddings():
    """Verify curated standards JSON and .npy matrix load with correct shapes and formats."""
    standards, embeddings, vec_bundle = load_knowledge_base()

    assert len(standards) >= 20
    assert embeddings is not None
    assert embeddings.shape[0] == len(standards)
    assert embeddings.ndim == 2
    assert vec_bundle is not None
    assert "vectorizer" in vec_bundle
    assert "svd" in vec_bundle

    # Verify essential fields in every standard
    for std in standards:
        assert "id" in std
        assert "domain" in std
        assert "authority" in std
        assert "title" in std
        assert "rule_text" in std
        assert "trigger_condition" in std
        assert "mandatory_directive" in std
        assert "citation" in std


def test_retrieve_icar_pesticide_washoff_standard():
    """Verify queries on pesticide foliar spraying retrieve ICAR-CRIDA 4-6h dry period rule."""
    query = "Is it safe to spray chemical pesticide on my crops before it rains tomorrow?"
    results = retrieve_relevant_standards(query, top_k=2)

    assert len(results) > 0
    top = results[0]
    assert "pesticide" in top["title"].lower() or "spray" in top["title"].lower()
    assert "ICAR" in top["authority"]
    assert "4 to 6" in top["rule_text"]
    assert top["relevance_score"] > 0.3


def test_retrieve_bis_fluoride_water_standard():
    """Verify drinking water fluoride queries retrieve BIS IS 10500:2012 permissible limit (1.5 mg/L)."""
    query = "What is the permissible limit for fluoride in drinking water in India?"
    results = retrieve_relevant_standards(query, top_k=1)

    assert len(results) == 1
    top = results[0]
    assert "Fluoride" in top["title"]
    assert "BIS" in top["authority"] or "Bureau of Indian Standards" in top["authority"]
    assert "1.5 mg/L" in top["rule_text"]
    assert "Nalgonda" in top["mandatory_directive"] or "RO" in top["mandatory_directive"]


def test_retrieve_bis_nitrate_blue_baby_standard():
    """Verify nitrate potability queries retrieve 45 mg/L limit and methemoglobinemia warning."""
    query = "Is water with 60 mg/L nitrate safe for infants to drink?"
    results = retrieve_relevant_standards(query, top_k=1)

    assert len(results) == 1
    top = results[0]
    assert "Nitrate" in top["title"]
    assert "45.0 mg/L" in top["rule_text"] or "45 mg/L" in top["rule_text"]
    assert "blue-baby" in top["rule_text"].lower() or "methemoglobinemia" in top["rule_text"].lower()


def test_retrieve_ndma_heatwave_school_shift():
    """Verify queries on extreme heat school hours retrieve NDMA morning shift guidelines."""
    query = "Should school hours be rescheduled when temperatures exceed 42C during a heatwave?"
    results = retrieve_relevant_standards(query, top_k=2)

    assert len(results) > 0
    titles = [r["title"] for r in results]
    assert any("School" in t or "Heatwave" in t for t in titles)
    school_std = next((r for r in results if "School" in r["title"]), results[0])
    assert "42.0°C" in school_std["rule_text"] or "Heatwave" in school_std["title"]


def test_retrieve_cpcb_grap_standard():
    """Verify air pollution queries retrieve CAQM/CPCB Graded Response Action Plan."""
    query = "What actions are enforced under GRAP Stage 3 when AQI exceeds 400?"
    results = retrieve_relevant_standards(query, top_k=1)

    assert len(results) == 1
    top = results[0]
    assert "GRAP" in top["title"] or "Graded Response Action Plan" in top["title"]
    assert "Stage 3" in top["rule_text"]
    assert "CPCB" in top["authority"] or "CAQM" in top["authority"]


def test_retrieve_cea_transformer_tropical_night_standard():
    """Verify grid power queries on nighttime heat retrieve CEA transformer dissipation rules."""
    query = "Will power transformers overheat if night temperatures stay above 28C?"
    results = retrieve_relevant_standards(query, top_k=1)

    assert len(results) == 1
    top = results[0]
    assert "Transformer" in top["title"]
    assert "28.0°C" in top["rule_text"] or "Tropical Night" in top["title"]
    assert "Central Electricity Authority" in top["authority"]


def test_format_retrieved_standards_for_prompt():
    """Verify formatted prompt markdown contains titles, authorities, citations, and trigger thresholds."""
    results = retrieve_relevant_standards("Can I spray pesticide tomorrow?", top_k=1)
    prompt_md = format_retrieved_standards_for_prompt(results)

    assert "RELEVANT INDIAN REGULATORY STANDARDS" in prompt_md
    assert "Operational Rule" in prompt_md
    assert "Trigger Threshold" in prompt_md
    assert "Mandatory Directive" in prompt_md
    assert "Citation" in prompt_md


def test_api_assistant_chat_includes_retrieved_standards(api_client: TestClient):
    """Test that /api/assistant/chat returns populated retrieved_standards list in response."""
    payload = {
        "message": "Can I spray pesticide on my wheat crop tomorrow in Jaipur?",
        "city": "Jaipur",
        "persona": "farmer",
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "retrieved_standards" in data
    assert isinstance(data["retrieved_standards"], list)
    assert len(data["retrieved_standards"]) > 0

    top_std = data["retrieved_standards"][0]
    assert "title" in top_std
    assert "authority" in top_std
    assert "rule_text" in top_std
    assert "relevance_score" in top_std
    assert "ICAR" in top_std["authority"] or "pesticide" in top_std["title"].lower() or "wheat" in top_std["title"].lower()


def test_multi_shard_domain_loading():
    """Verify that all 5 individual domain shards load with correct schemas and embeddings."""
    from climacore.analytics.rag_retriever import load_shard, KNOWN_DOMAINS
    for domain in KNOWN_DOMAINS:
        stds, embs, vec = load_shard(domain)
        assert len(stds) >= 6, f"Shard {domain} should have at least 6 standards"
        assert embs is not None, f"Shard {domain} embeddings should exist"
        assert embs.shape[0] == len(stds)
        assert vec is not None, f"Shard {domain} vectorizer pipeline should exist"


def test_regional_context_detection_and_boost():
    """Verify regional detector identifies Western Arid, Indo-Gangetic, and Deccan contexts."""
    from climacore.analytics.rag_retriever import detect_regional_context, retrieve_relevant_standards

    assert detect_regional_context("wheat sowing in Punjab", "Amritsar") == "Indo-Gangetic Plain"
    assert detect_regional_context("groundwater borewell fluoride in Thar", "Jaipur") == "Western Arid Zone"
    assert detect_regional_context("black cotton soil waterlogging", "Nagpur") == "Deccan Plateau"
    assert detect_regional_context("storm surge and salinity ingress", "Mumbai") == "Coastal & Deltaic"

    # Verify regional query retrieves region-aligned standard
    res_jaipur = retrieve_relevant_standards("Mustard pod-setting temperature in Rajasthan", city="Jaipur", top_k=1)
    assert len(res_jaipur) > 0
    assert "Mustard" in res_jaipur[0]["title"] or "Arid" in res_jaipur[0].get("region", "")


def test_retrieve_railway_track_buckling_rdso():
    """Verify railway safety queries retrieve RDSO CWR track buckling and hot weather patrol rules."""
    query = "When do Indian Railways enforce hot weather patrolling to prevent track buckling?"
    results = retrieve_relevant_standards(query, top_k=1)
    assert len(results) == 1
    top = results[0]
    assert "Rail" in top["title"] or "Track Buckling" in top["title"]
    assert "RDSO" in top["authority"] or "Railways" in top["authority"]
    assert "Td + 25" in top["rule_text"] or "patrolling" in top["rule_text"].lower()


def test_retrieve_uranium_groundwater_cgwb():
    """Verify groundwater uranium queries retrieve AERB/CGWB 30 ppb radiological limits."""
    query = "Is groundwater with 45 ppb uranium safe to drink in Punjab?"
    results = retrieve_relevant_standards(query, city="Amritsar", top_k=1)
    assert len(results) == 1
    top = results[0]
    assert "Uranium" in top["title"]
    assert "30.0 ppb" in top["rule_text"] or "30 ppb" in top["rule_text"]
    assert "CGWB" in top["authority"] or "AERB" in top["authority"]

