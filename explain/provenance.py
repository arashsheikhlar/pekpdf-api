"""
Provenance mapping and confidence scoring for Explain outputs.
"""
import re
from typing import List, Dict, Any, Tuple
from collections import Counter


def compute_lexical_overlap(claim: str, source_pages: List[str]) -> Tuple[float, List[int]]:
    """
    Compute lexical overlap between a claim and source pages.
    Returns: (overlap_score, relevant_page_numbers)
    """
    # Normalize claim
    claim_words = set(re.findall(r'\w+', claim.lower()))
    if not claim_words:
        return 0.0, []
    
    page_scores = []
    for idx, page_text in enumerate(source_pages):
        page_words = set(re.findall(r'\w+', page_text.lower()))
        if page_words:
            overlap = len(claim_words & page_words) / len(claim_words)
            page_scores.append((idx + 1, overlap))  # 1-indexed pages
    
    # Sort by overlap score
    page_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return average of top 3 and their page numbers
    top_pages = page_scores[:3]
    if top_pages:
        avg_score = sum(s for _, s in top_pages) / len(top_pages)
        page_nums = [p for p, _ in top_pages if _ > 0.1]  # Only pages with >10% overlap
        return avg_score, page_nums
    
    return 0.0, []


def extract_numbers(text: str) -> List[str]:
    """Extract numbers from text for numeric alignment checks."""
    # Match numbers including decimals, percentages, currency
    numbers = re.findall(r'\$?[\d,]+\.?\d*%?', text)
    return [n.strip('$,') for n in numbers]


def compute_numeric_alignment(claim: str, source_text: str) -> float:
    """
    Check if numbers in claim appear in source.
    Returns: alignment ratio (0.0 to 1.0)
    """
    claim_numbers = extract_numbers(claim)
    if not claim_numbers:
        return 1.0  # No numbers to check
    
    source_numbers = set(extract_numbers(source_text))
    if not source_numbers:
        return 0.0
    
    matched = sum(1 for n in claim_numbers if n in source_numbers)
    return matched / len(claim_numbers)


def compute_confidence(claim: str, source_pages: List[str], full_text: str) -> Dict[str, Any]:
    """
    Compute confidence metrics for a claim.
    Returns: {confidence: float, sources: [pages], numeric_alignment: float}
    """
    lexical_score, pages = compute_lexical_overlap(claim, source_pages)
    numeric_score = compute_numeric_alignment(claim, full_text)
    
    # Combined confidence: weighted average
    confidence = 0.6 * lexical_score + 0.4 * numeric_score
    
    return {
        "confidence": round(confidence * 100, 1),  # Convert to percentage
        "sources": pages,
        "numeric_alignment": round(numeric_score * 100, 1),
        "lexical_overlap": round(lexical_score * 100, 1)
    }


def map_explanations_to_provenance(explanations: Dict[str, str], source_pages: List[str], full_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Map each explanation section to provenance.
    Returns: {section_name: {confidence, sources, ...}}
    """
    provenance = {}
    
    for section, text in explanations.items():
        if text and len(text) > 10:
            prov = compute_confidence(text, source_pages, full_text)
            provenance[section] = prov
    
    return provenance


def compute_provenance_density(provenance: Dict[str, Dict]) -> float:
    """
    Compute overall provenance density (how well-grounded the output is).
    Returns: average confidence across all sections
    """
    if not provenance:
        return 0.0
    
    confidences = [p.get("confidence", 0) for p in provenance.values()]
    if confidences:
        return round(sum(confidences) / len(confidences), 1)
    return 0.0

