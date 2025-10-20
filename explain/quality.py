"""
Quality scoring and business rule validation for Explain outputs.
"""
from typing import Dict, Any, List
from .prompt_pack import get_domain_checklist


def check_outline_coverage(explanations: Dict[str, str], expected_sections: List[str]) -> Dict[str, Any]:
    """
    Check if all expected sections are present and non-empty.
    Returns: {coverage_ratio: float, missing: [sections]}
    """
    present = [s for s in expected_sections if explanations.get(s) and len(explanations[s]) > 10]
    coverage = len(present) / len(expected_sections) if expected_sections else 1.0
    missing = [s for s in expected_sections if s not in present]
    
    return {
        "coverage_ratio": round(coverage * 100, 1),
        "present_sections": present,
        "missing_sections": missing
    }


def check_domain_requirements(result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Check domain-specific business rules.
    Returns: {passed: bool, checks: {check_name: bool}, warnings: [str]}
    """
    checklist = get_domain_checklist(domain)
    checks = {}
    warnings = []
    
    explanations_text = " ".join(str(v) for v in result.get("explanations", {}).values()).lower()
    summary_text = result.get("summary", "").lower()
    combined = explanations_text + " " + summary_text
    
    for required in checklist:
        # Simple keyword presence check (can be made more sophisticated)
        present = required.lower() in combined or any(required.lower() in k.lower() for k in result.get("explanations", {}).keys())
        checks[required] = present
        if not present:
            warnings.append(f"Missing required element: {required}")
    
    passed = all(checks.values()) if checks else True
    
    return {
        "passed": passed,
        "checks": checks,
        "warnings": warnings
    }


def evaluate_quality(result: Dict[str, Any], domain: str, provenance: Dict[str, Dict], expected_sections: List[str]) -> Dict[str, Any]:
    """
    Comprehensive quality evaluation.
    Returns: {overall_score: float, coverage: {}, domain_checks: {}, provenance_density: float}
    """
    # Coverage
    coverage = check_outline_coverage(result.get("explanations", {}), expected_sections)
    
    # Domain requirements
    domain_checks = check_domain_requirements(result, domain)
    
    # Provenance density
    from .provenance import compute_provenance_density
    prov_density = compute_provenance_density(provenance)
    
    # Overall score: weighted average
    coverage_score = coverage["coverage_ratio"]
    domain_score = 100 if domain_checks["passed"] else 50
    
    overall = (0.4 * coverage_score + 0.3 * domain_score + 0.3 * prov_density)
    
    return {
        "overall_score": round(overall, 1),
        "coverage": coverage,
        "domain_checks": domain_checks,
        "provenance_density": prov_density,
        "gate_passed": overall >= 70  # Threshold for "acceptable" quality
    }

