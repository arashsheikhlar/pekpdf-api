"""
Provenance mapping and confidence scoring for Summarize outputs.
"""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


def compute_lexical_overlap(claim: str, source_pages: List[str]) -> Tuple[float, List[int]]:
    """
    Compute lexical overlap between a claim and source pages.
    Returns: (overlap_score, relevant_page_numbers)
    """
    claim_words = set(re.findall(r'\w+', claim.lower()))
    if not claim_words:
        return 0.0, []
    
    page_scores = []
    for idx, page_text in enumerate(source_pages):
        page_words = set(re.findall(r'\w+', page_text.lower()))
        if page_words:
            overlap = len(claim_words & page_words) / len(claim_words)
            page_scores.append((idx + 1, overlap))
    
    page_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_pages = page_scores[:3]
    if top_pages:
        avg_score = sum(s for _, s in top_pages) / len(top_pages)
        page_nums = [p for p, _ in top_pages if _ > 0.1]
        return avg_score, page_nums
    
    return 0.0, []


def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text."""
    numbers = re.findall(r'\$?[\d,]+\.?\d*%?', text)
    return [n.strip('$,') for n in numbers]


def compute_numeric_alignment(claim: str, source_text: str) -> float:
    """Check if numbers in claim appear in source."""
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return 1.0
    
    source_numbers = set(extract_numbers(source_text))
    if not source_numbers:
        return 0.0
    
    matched = sum(1 for n in claim_numbers if n in source_numbers)
    return matched / len(claim_numbers)


def compute_confidence(claim: str, source_pages: List[str], full_text: str) -> Dict[str, Any]:
    """
    Compute confidence metrics for a claim.
    """
    lexical_score, pages = compute_lexical_overlap(claim, source_pages)
    numeric_score = compute_numeric_alignment(claim, full_text)
    
    confidence = 0.6 * lexical_score + 0.4 * numeric_score
    
    return {
        "confidence": round(confidence * 100, 1),
        "sources": pages,
        "numeric_alignment": round(numeric_score * 100, 1),
        "lexical_overlap": round(lexical_score * 100, 1)
    }


def map_summary_to_provenance(result: Dict[str, Any], source_pages: List[str], full_text: str) -> Dict[str, Any]:
    """
    Map summary elements to provenance.
    Returns: {summary: {...}, key_topics: [...], main_points: [...], recommendations: [...]}
    """
    provenance = {}
    
    # Summary
    if result.get("summary"):
        provenance["summary"] = compute_confidence(result["summary"], source_pages, full_text)
    
    # Key topics (aggregate)
    if result.get("key_topics"):
        topics_text = " ".join(result["key_topics"])
        provenance["key_topics"] = compute_confidence(topics_text, source_pages, full_text)
    
    # Main points (per item)
    if result.get("main_points"):
        main_points_prov = []
        for point in result["main_points"]:
            if isinstance(point, str) and len(point) > 10:
                prov = compute_confidence(point, source_pages, full_text)
                main_points_prov.append(prov)
        provenance["main_points"] = main_points_prov
    
    # Recommendations (per item)
    if result.get("recommendations"):
        recs_prov = []
        for rec in result["recommendations"]:
            if isinstance(rec, str) and len(rec) > 10:
                prov = compute_confidence(rec, source_pages, full_text)
                recs_prov.append(prov)
        provenance["recommendations"] = recs_prov
    
    return provenance


def compute_provenance_density(provenance: Dict) -> float:
    """
    Compute overall provenance density.
    """
    confidences = []
    
    if provenance.get("summary"):
        confidences.append(provenance["summary"].get("confidence", 0))
    
    if provenance.get("key_topics"):
        confidences.append(provenance["key_topics"].get("confidence", 0))
    
    if provenance.get("main_points"):
        for mp in provenance["main_points"]:
            if isinstance(mp, dict):
                confidences.append(mp.get("confidence", 0))
    
    if provenance.get("recommendations"):
        for rec in provenance["recommendations"]:
            if isinstance(rec, dict):
                confidences.append(rec.get("confidence", 0))
    
    if confidences:
        return round(sum(confidences) / len(confidences), 1)
    return 0.0

