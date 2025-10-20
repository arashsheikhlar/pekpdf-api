"""
Quality scoring and business rule validation for Summarize outputs.
"""
from typing import Dict, Any, List
from .prompt_pack import get_domain_checklist


def check_completeness(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if all expected fields are present and populated.
    """
    required_fields = ["summary", "key_topics", "main_points", "recommendations"]
    present = []
    missing = []
    
    for field in required_fields:
        value = result.get(field)
        if value:
            if isinstance(value, str) and len(value) > 10:
                present.append(field)
            elif isinstance(value, list) and len(value) > 0:
                present.append(field)
            else:
                missing.append(field)
        else:
            missing.append(field)
    
    coverage = len(present) / len(required_fields) if required_fields else 1.0
    
    return {
        "coverage_ratio": round(coverage * 100, 1),
        "present_fields": present,
        "missing_fields": missing
    }


def check_domain_requirements(result: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Check domain-specific business rules.
    """
    checklist = get_domain_checklist(domain)
    checks = {}
    warnings = []
    
    # Combine all text for checking
    combined = ""
    if result.get("summary"):
        combined += result["summary"] + " "
    if result.get("key_topics"):
        combined += " ".join(str(t) for t in result["key_topics"]) + " "
    if result.get("main_points"):
        combined += " ".join(str(p) for p in result["main_points"]) + " "
    
    combined = combined.lower()
    
    for required in checklist:
        present = required.lower() in combined
        checks[required] = present
        if not present:
            warnings.append(f"Missing required element: {required}")
    
    passed = all(checks.values()) if checks else True
    
    return {
        "passed": passed,
        "checks": checks,
        "warnings": warnings
    }


def evaluate_quality(result: Dict[str, Any], domain: str, provenance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive quality evaluation.
    """
    # Completeness
    completeness = check_completeness(result)
    
    # Domain requirements
    domain_checks = check_domain_requirements(result, domain)
    
    # Provenance density
    from .provenance import compute_provenance_density
    prov_density = compute_provenance_density(provenance)
    
    # Overall score
    completeness_score = completeness["coverage_ratio"]
    domain_score = 100 if domain_checks["passed"] else 50
    
    overall = (0.4 * completeness_score + 0.3 * domain_score + 0.3 * prov_density)
    
    return {
        "overall_score": round(overall, 1),
        "completeness": completeness,
        "domain_checks": domain_checks,
        "provenance_density": prov_density,
        "gate_passed": overall >= 70
    }

