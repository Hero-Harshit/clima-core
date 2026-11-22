"""
Automated Pytest Suite for Clima Core Multilingual API Endpoints.
Verifies Requirement R2 (Dynamic API & Assistant Localization) and R3 (Automated Verification):
- /api/persona/briefing accepts language parameter across all 7 languages.
- /api/assistant/chat accepts language parameter across all 7 languages.
- Default language behavior is English ('en') when parameter is omitted.
- Cache key isolation ensures language requests do not collide.
- Rigorous script validation ensures responses match the expected Indic Unicode block.
"""

import re
import pytest
from fastapi.testclient import TestClient
from climacore.service.app import app
from climacore.personas.gemini_briefing import _BRIEFING_CACHE, generate_deterministic_briefing, generate_gemini_persona_briefing

# Unicode script ranges for Indic language validation
UNICODE_PATTERNS = {
    "hi": re.compile(r"[\u0900-\u097F]"),  # Devanagari (Hindi)
    "bn": re.compile(r"[\u0980-\u09FF]"),  # Bengali
    "ta": re.compile(r"[\u0B80-\u0BFF]"),  # Tamil
    "te": re.compile(r"[\u0C00-\u0C7F]"),  # Telugu
    "mr": re.compile(r"[\u0900-\u097F]"),  # Devanagari (Marathi)
    "gu": re.compile(r"[\u0A80-\u0AFF]"),  # Gujarati
}

ALL_LANGUAGES = ["en", "hi", "bn", "ta", "te", "mr", "gu"]
INDIC_LANGUAGES = ["hi", "bn", "ta", "te", "mr", "gu"]


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    return TestClient(app)


def test_persona_briefing_default_english(api_client: TestClient):
    """Omitting language parameter must default to English."""
    payload = {
        "city": "Mumbai",
        "persona": "gig_worker",
        "day_offset": 0,
    }
    resp = api_client.post("/api/persona/briefing", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "operational_guidance" in data
    assert "advisories" in data
    assert len(data["advisories"]) >= 3
    assert data.get("language", "en") == "en"
    # English response should contain ASCII characters
    assert any(c.isascii() and c.isalpha() for c in data["operational_guidance"])


@pytest.mark.parametrize("lang", ALL_LANGUAGES)
def test_persona_briefing_all_languages(api_client: TestClient, lang: str):
    """Verify /api/persona/briefing returns valid localized output for each supported language."""
    payload = {
        "city": "Mumbai",
        "persona": "gig_worker",
        "day_offset": 0,
        "language": lang,
    }
    resp = api_client.post("/api/persona/briefing", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "operational_guidance" in data
    assert len(data["operational_guidance"]) > 10
    assert "advisories" in data
    assert len(data["advisories"]) >= 3
    assert "workability_score" in data
    assert data["city"] == "Mumbai"
    assert data.get("language") == lang

    # For Indic languages, verify the script matches Unicode blocks
    if lang in UNICODE_PATTERNS:
        pattern = UNICODE_PATTERNS[lang]
        guidance = data["operational_guidance"]
        assert pattern.search(guidance) is not None, f"Language {lang} guidance must contain native script: {guidance}"
        first_advisory = data["advisories"][0]
        assert pattern.search(first_advisory) is not None, f"Language {lang} advisory must contain native script: {first_advisory}"


def test_persona_briefing_cache_isolation_by_language(api_client: TestClient):
    """Verify that requests for different languages do not collide in the briefing cache."""
    # Request English
    resp_en = api_client.post(
        "/api/persona/briefing",
        json={"city": "Pune", "persona": "farmer", "day_offset": 0, "language": "en"}
    )
    assert resp_en.status_code == 200
    data_en = resp_en.json()

    # Request Hindi for the exact same city and persona
    resp_hi = api_client.post(
        "/api/persona/briefing",
        json={"city": "Pune", "persona": "farmer", "day_offset": 0, "language": "hi"}
    )
    assert resp_hi.status_code == 200
    data_hi = resp_hi.json()

    # The English and Hindi outputs must be distinct and language-specific
    assert data_en["operational_guidance"] != data_hi["operational_guidance"]
    assert UNICODE_PATTERNS["hi"].search(data_hi["operational_guidance"]) is not None
    assert data_en.get("language") == "en"
    assert data_hi.get("language") == "hi"


def test_assistant_chat_default_english(api_client: TestClient):
    """Omitting language parameter in assistant chat must default to English."""
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
    assert data.get("language", "en") == "en"
    assert len(data["suggested_followups"]) == 3


@pytest.mark.parametrize("lang", ALL_LANGUAGES)
def test_assistant_chat_all_languages(api_client: TestClient, lang: str):
    """Verify /api/assistant/chat returns localized response and followups for each supported language."""
    payload = {
        "message": "Should I carry an umbrella today?",
        "city": "Delhi",
        "persona": "student",
        "history": [],
        "language": lang,
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert "response" in data
    assert len(data["response"]) > 20
    assert data["city"] == "Delhi"
    assert data["persona"] == "student"
    assert data.get("language") == lang
    assert len(data["suggested_followups"]) == 3

    # For Indic languages, check response and followups script
    if lang in UNICODE_PATTERNS:
        pattern = UNICODE_PATTERNS[lang]
        response_text = data["response"]
        assert pattern.search(response_text) is not None, f"Response for {lang} must contain native script: {response_text[:100]}"
        first_followup = data["suggested_followups"][0]
        assert pattern.search(first_followup) is not None, f"Followup for {lang} must contain native script: {first_followup}"


@pytest.mark.parametrize("lang", INDIC_LANGUAGES)
def test_assistant_chat_farmer_pesticide_query_indic(api_client: TestClient, lang: str):
    """Verify assistant handles domain-specific agricultural inquiries in Indic scripts."""
    payload = {
        "message": "Can I spray pesticide on my crops tomorrow?",
        "city": "Nagpur",
        "persona": "farmer",
        "history": [],
        "language": lang,
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("language") == lang
    pattern = UNICODE_PATTERNS[lang]
    assert pattern.search(data["response"]) is not None, f"Pesticide response for {lang} must contain native script"


@pytest.mark.parametrize("unsupported_lang", ["fr", "de", "zh", "unknown", "es"])
def test_assistant_chat_unsupported_language_fallback_english(api_client: TestClient, unsupported_lang: str):
    """Unsupported language codes must fall back cleanly to English, not Hindi or other Indic scripts."""
    payload = {
        "message": "Do I need an umbrella today?",
        "city": "Mumbai",
        "persona": "commuter",
        "language": unsupported_lang,
    }
    resp = api_client.post("/api/assistant/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    response_text = data["response"]
    # Ensure no Indic Unicode characters (Devanagari, Bengali, Tamil, Telugu, Gujarati, etc. U+0900 to U+0D7F)
    assert not any('\u0900' <= ch <= '\u0d7f' for ch in response_text), (
        f"Unsupported language '{unsupported_lang}' returned Indic script: {response_text[:100]}"
    )
    # Ensure response contains valid English text
    assert "umbrella" in response_text.lower() or "rain" in response_text.lower() or "mumbai" in response_text.lower()
    # Check followups are also in English (no Indic)
    for followup in data.get("suggested_followups", []):
        assert not any('\u0900' <= ch <= '\u0d7f' for ch in followup), (
            f"Followup for '{unsupported_lang}' contained Indic script: {followup}"
        )


def test_persona_briefing_whitespace_normalization_and_cache_integrity(api_client: TestClient):
    """Whitespace-padded language strings (e.g. '  hi  ') must return native script and not poison cache."""
    _BRIEFING_CACHE.clear()

    # 1. Request with leading/trailing whitespace around Hindi
    resp_ws = api_client.post(
        "/api/persona/briefing",
        json={"city": "Jaipur", "persona": "farmer", "day_offset": 0, "language": "  hi  "}
    )
    assert resp_ws.status_code == 200
    data_ws = resp_ws.json()
    assert UNICODE_PATTERNS["hi"].search(data_ws["operational_guidance"]) is not None, (
        f"Whitespace request '  hi  ' failed to return Hindi script: {data_ws['operational_guidance']}"
    )

    # 2. Subsequent request with canonical 'hi' should hit cache or yield identical valid Hindi content
    resp_hi = api_client.post(
        "/api/persona/briefing",
        json={"city": "Jaipur", "persona": "farmer", "day_offset": 0, "language": "hi"}
    )
    assert resp_hi.status_code == 200
    data_hi = resp_hi.json()
    assert UNICODE_PATTERNS["hi"].search(data_hi["operational_guidance"]) is not None, (
        "Subsequent 'hi' request was contaminated by whitespace cache entry!"
    )
    assert data_hi["operational_guidance"] == data_ws["operational_guidance"]


def test_persona_briefing_language_none_handling(api_client: TestClient):
    """Explicit language=None in request body must not crash with 500 or drop live metrics."""
    resp = api_client.post(
        "/api/persona/briefing",
        json={"city": "Delhi", "persona": "farmer", "day_offset": 0, "language": None}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "operational_guidance" in data
    assert len(data["advisories"]) >= 3
    # Default is English
    assert data.get("language") == "en"
    assert any(c.isascii() and c.isalpha() for c in data["operational_guidance"])


@pytest.mark.anyio
async def test_generate_gemini_persona_briefing_direct_none_language():
    """Direct invocation of generate_gemini_persona_briefing with language=None must not raise AttributeError."""
    briefing = await generate_gemini_persona_briefing(
        city="Bengaluru",
        persona_id="student",
        metrics={"temp_max": 28.0, "precipitation_mm": 5.0},
        language=None,
    )
    assert briefing is not None
    assert "operational_guidance" in briefing
    assert briefing["language"] == "en"


def test_assistant_chat_whitespace_normalization(api_client: TestClient):
    """Whitespace-padded language in assistant chat should be normalized and return native script."""
    resp = api_client.post(
        "/api/assistant/chat",
        json={
            "message": "Do I need an umbrella today?",
            "city": "Mumbai",
            "persona": "student",
            "language": "  hi  ",
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert UNICODE_PATTERNS["hi"].search(data["response"]) is not None
    assert data.get("language") == "hi"
