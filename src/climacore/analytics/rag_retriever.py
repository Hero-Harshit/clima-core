"""
Institutional Indian Environmental & Regulatory Knowledge Base RAG Retriever.
Federated Multi-Shard Architecture (Option C: Multi-Shard + Regional Focus).
Executes zero-database, serverless semantic vector retrieval across 5 domain shards:
1. agro_meteorology (ICAR-CRIDA, IIWBR, NRRI, CICR, DRMR, CPRI)
2. hydro_geology (BIS IS 10500:2012, CGWB, AERB)
3. public_health_disaster (NDMA, IIPH, MoE, NCDC)
4. air_quality_emissions (CPCB, CAQM, MoEFCC)
5. infrastructure_energy (CEA, POSOCO/Grid-India, RDSO Indian Railways, NHAI, CPHEEO/CWC)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Master caches
_STANDARDS_CACHE: Optional[List[Dict[str, Any]]] = None
_EMBEDDINGS_CACHE: Optional[np.ndarray] = None
_VECTORIZER_CACHE: Optional[Dict[str, Any]] = None

# Shard-level caches
_SHARD_CACHE: Dict[str, Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, Any]]]] = {}

KNOWN_DOMAINS = [
    "agro_meteorology",
    "hydro_geology",
    "public_health_disaster",
    "air_quality_emissions",
    "infrastructure_energy",
]

REGIONAL_MAPPINGS: Dict[str, List[str]] = {
    "Western Arid Zone": [
        "rajasthan", "jaipur", "jodhpur", "bikaner", "jaisalmer", "barmer", "udaipur",
        "thar", "arid", "desert", "mustard", "bajra", "guar"
    ],
    "Indo-Gangetic Plain": [
        "delhi", "punjab", "haryana", "uttar pradesh", "bihar", "kolkata", "patna",
        "lucknow", "chandigarh", "ncr", "amritsar", "ludhiana", "kanpur", "ganga",
        "stubble", "smog", "wheat", "alluvium", "potato"
    ],
    "Deccan Plateau": [
        "maharashtra", "nagpur", "pune", "hyderabad", "telangana", "karnataka",
        "bengaluru", "andhra", "black soil", "vertisol", "cotton", "pulses", "deccan"
    ],
    "Coastal & Deltaic": [
        "mumbai", "chennai", "kolkata", "kochi", "visakhapatnam", "goa", "surat",
        "puri", "bhubaneswar", "coastal", "cyclone", "salinity", "seawater", "storm surge"
    ],
    "North-East & Hill Zone": [
        "himachal", "uttarakhand", "shimla", "dehradun", "guwahati", "assam",
        "shillong", "srinagar", "cloudburst", "landslide", "hilly"
    ],
}


def get_knowledge_base_dir() -> Path:
    """Resolve directory holding standards JSON and .npy vector files."""
    return Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge"


def load_knowledge_base() -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load master curated Indian environmental standards, .npy embeddings, and vectorizer pipeline."""
    global _STANDARDS_CACHE, _EMBEDDINGS_CACHE, _VECTORIZER_CACHE

    if _STANDARDS_CACHE is not None:
        return _STANDARDS_CACHE, _EMBEDDINGS_CACHE, _VECTORIZER_CACHE

    kb_dir = get_knowledge_base_dir()
    json_path = kb_dir / "indian_environmental_standards.json"
    npy_path = kb_dir / "knowledge_embeddings.npy"
    vec_path = kb_dir / "knowledge_vectorizer.joblib"

    if not json_path.exists():
        logger.warning(f"Knowledge base JSON not found at {json_path}")
        _STANDARDS_CACHE = []
        return [], None, None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            _STANDARDS_CACHE = json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load standards JSON: {exc}")
        _STANDARDS_CACHE = []

    if npy_path.exists():
        try:
            _EMBEDDINGS_CACHE = np.load(npy_path)
        except Exception as exc:
            logger.error(f"Failed to load embeddings .npy: {exc}")
            _EMBEDDINGS_CACHE = None

    if vec_path.exists():
        try:
            import joblib
            _VECTORIZER_CACHE = joblib.load(vec_path)
        except Exception as exc:
            logger.error(f"Failed to load vectorizer pipeline: {exc}")
            _VECTORIZER_CACHE = None

    return _STANDARDS_CACHE, _EMBEDDINGS_CACHE, _VECTORIZER_CACHE


def load_shard(domain: str) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load a specific domain shard's standards, embeddings, and vectorizer pipeline."""
    global _SHARD_CACHE

    if domain in _SHARD_CACHE:
        return _SHARD_CACHE[domain]

    kb_dir = get_knowledge_base_dir()
    shards_dir = kb_dir / "shards"
    json_path = shards_dir / f"{domain}.json"
    npy_path = shards_dir / f"{domain}_embeddings.npy"
    vec_path = shards_dir / f"{domain}_vectorizer.joblib"

    if not json_path.exists():
        # Fall back to master filtering if shard file not found
        master_stds, _, _ = load_knowledge_base()
        filtered = [s for s in master_stds if s.get("domain") == domain]
        _SHARD_CACHE[domain] = (filtered, None, None)
        return filtered, None, None

    standards = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            standards = json.load(f)
    except Exception as exc:
        logger.error(f"Failed to load shard JSON {domain}: {exc}")

    embeddings = None
    if npy_path.exists():
        try:
            embeddings = np.load(npy_path)
        except Exception as exc:
            logger.error(f"Failed to load shard embeddings {domain}: {exc}")

    vec_bundle = None
    if vec_path.exists():
        try:
            import joblib
            vec_bundle = joblib.load(vec_path)
        except Exception as exc:
            logger.error(f"Failed to load shard vectorizer {domain}: {exc}")

    _SHARD_CACHE[domain] = (standards, embeddings, vec_bundle)
    return standards, embeddings, vec_bundle


def detect_regional_context(query: str, city: Optional[str] = None) -> Optional[str]:
    """Infer the Indian geographic/agro-climatic region from query tokens or target city."""
    search_text = (f"{city or ''} {query}").lower()
    for region, keywords in REGIONAL_MAPPINGS.items():
        if any(kw in search_text for kw in keywords):
            return region
    return None


def detect_target_domains(query: str, persona: Optional[str] = None) -> List[str]:
    """Determine the most relevant domain shards based on persona affinity and query intent."""
    q_lower = query.lower()
    p_lower = (persona or "").lower()
    targets = []

    # Persona Priors
    if any(k in p_lower for k in ["farmer", "agriculture", "kisan"]):
        targets.append("agro_meteorology")
    if any(k in p_lower for k in ["water", "groundwater", "hydro"]):
        targets.append("hydro_geology")
    if any(k in p_lower for k in ["student", "vulnerable", "health", "hospital"]):
        targets.append("public_health_disaster")
    if any(k in p_lower for k in ["commuter", "industry", "factory"]):
        targets.append("air_quality_emissions")
    if any(k in p_lower for k in ["utilities", "power", "grid", "transport", "railway"]):
        targets.append("infrastructure_energy")

    # Query Keywords
    if re.search(r"\b(pesticide|spray|crop|crops|wheat|rice|paddy|cotton|mustard|potato|irrigate|farming|sowing|soil|blight|anthesis)\b", q_lower):
        if "agro_meteorology" not in targets:
            targets.insert(0, "agro_meteorology")

    if re.search(r"\b(fluoride|nitrate|arsenic|lead|uranium|hardness|tds|water|borewell|potable|drinking|coliform|aquifer)\b", q_lower):
        if "hydro_geology" not in targets:
            targets.insert(0, "hydro_geology")

    if re.search(r"\b(heatwave|wet-bulb|wet bulb|school|children|cold wave|cyclone|dengue|malaria|heatstroke|alert)\b", q_lower):
        if "public_health_disaster" not in targets:
            targets.insert(0, "public_health_disaster")

    if re.search(r"\b(aqi|pm2.5|pm10|air quality|grap|stubble|smog|inversion|stagnation|kiln)\b", q_lower):
        if "air_quality_emissions" not in targets:
            targets.insert(0, "air_quality_emissions")

    if re.search(r"\b(transformer|cooling tower|rail|track|buckling|railway|bitumen|asphalt|drainage|substation|transmission|power plant)\b", q_lower):
        if "infrastructure_energy" not in targets:
            targets.insert(0, "infrastructure_energy")

    return targets if targets else KNOWN_DOMAINS


def score_standards_list(
    standards: List[Dict[str, Any]],
    embeddings: Optional[np.ndarray],
    vec_bundle: Optional[Dict[str, Any]],
    query: str,
    target_region: Optional[str] = None,
) -> np.ndarray:
    """Compute dense similarities and lexical/regional boost for a list of standards."""
    scores = np.zeros(len(standards), dtype=np.float32)
    q_lower = query.lower()

    # 1. Dense Semantic Similarity
    if embeddings is not None and vec_bundle is not None:
        try:
            vectorizer = vec_bundle["vectorizer"]
            svd = vec_bundle["svd"]
            q_tfidf = vectorizer.transform([query])
            q_dense = svd.transform(q_tfidf)
            norm = np.linalg.norm(q_dense)
            if norm > 1e-9:
                q_norm = q_dense / norm
                scores += np.dot(embeddings, q_norm.T).flatten()
        except Exception as exc:
            logger.warning(f"Dense scoring error: {exc}")

    # 2. Token Matching & Keyword Boost
    for idx, std in enumerate(standards):
        std_id = std["id"].lower()
        std_text = (std["title"] + " " + std["rule_text"] + " " + std.get("trigger_condition", "")).lower()

        # Direct token overlaps
        q_tokens = [w for w in re.findall(r"\w+", q_lower) if len(w) > 3]
        overlap_count = sum(1 for t in q_tokens if t in std_text)
        if overlap_count > 0:
            scores[idx] += min(0.40, overlap_count * 0.12)

        # Regional Alignment Boost
        std_region = std.get("region", "National")
        if target_region and std_region != "National":
            if std_region.lower() == target_region.lower():
                scores[idx] += 0.25

    return scores


def retrieve_relevant_standards(
    query: str,
    persona: Optional[str] = None,
    city: Optional[str] = None,
    top_k: int = 2,
    domain_filter: Optional[str] = None,
    similarity_threshold: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Federated Multi-Shard Institutional RAG Retriever.
    Queries targeted domain shards and unifies candidates with regional prioritization.
    """
    target_region = detect_regional_context(query, city)
    candidate_domains = [domain_filter] if domain_filter else detect_target_domains(query, persona)

    all_scored_candidates: List[Tuple[float, Dict[str, Any]]] = []

    # 1. Search across identified domain shards
    for dom in candidate_domains:
        if dom not in KNOWN_DOMAINS:
            continue
        shard_stds, shard_emb, shard_vec = load_shard(dom)
        if not shard_stds:
            continue
        shard_scores = score_standards_list(shard_stds, shard_emb, shard_vec, query, target_region)
        for s_idx, score in enumerate(shard_scores):
            std_copy = dict(shard_stds[s_idx])
            std_copy["relevance_score"] = round(float(score), 4)
            std_copy["retrieved_from_shard"] = dom
            if target_region:
                std_copy["detected_region"] = target_region
            all_scored_candidates.append((float(score), std_copy))

    # 2. If no shard matches or cross-domain search is needed, also score master index
    if not all_scored_candidates:
        master_stds, master_emb, master_vec = load_knowledge_base()
        if master_stds:
            m_scores = score_standards_list(master_stds, master_emb, master_vec, query, target_region)
            for m_idx, score in enumerate(m_scores):
                std_copy = dict(master_stds[m_idx])
                std_copy["relevance_score"] = round(float(score), 4)
                std_copy["retrieved_from_shard"] = "master"
                all_scored_candidates.append((float(score), std_copy))

    # 3. Deduplicate and Rank by score descending
    all_scored_candidates.sort(key=lambda x: x[0], reverse=True)
    seen_ids = set()
    results = []

    for score, item in all_scored_candidates:
        if item["id"] in seen_ids:
            continue
        if score < similarity_threshold and len(results) >= 1:
            break
        seen_ids.add(item["id"])
        results.append(item)
        if len(results) >= top_k:
            break

    return results


def format_retrieved_standards_for_prompt(standards: List[Dict[str, Any]]) -> str:
    """Format retrieved standards into clean markdown for injection into agent context."""
    if not standards:
        return ""

    blocks = ["### RELEVANT INDIAN REGULATORY STANDARDS & SCIENTIFIC BENCHMARKS (RAG INJECTED):"]
    for std in standards:
        region_str = f" [Region: {std.get('region', 'National')}]" if std.get("region") != "National" else ""
        block = (
            f"- **{std['title']}** ({std['authority']}){region_str}\n"
            f"  *Citation*: {std['citation']}\n"
            f"  *Operational Rule*: {std['rule_text']}\n"
            f"  *Trigger Threshold*: `{std['trigger_condition']}`\n"
            f"  *Mandatory Directive*: {std['mandatory_directive']}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)
