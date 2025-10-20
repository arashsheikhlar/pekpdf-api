"""
Security utilities: PII detection, redaction, policy enforcement.
"""
import re
from typing import List, Dict, Tuple


# PII patterns
SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PHONE_PATTERN = re.compile(r'\b\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b')


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Detect PII in text.
    Returns: {"ssn": [...], "credit_card": [...], "email": [...], "phone": [...]}
    """
    return {
        "ssn": SSN_PATTERN.findall(text),
        "credit_card": CREDIT_CARD_PATTERN.findall(text),
        "email": EMAIL_PATTERN.findall(text),
        "phone": PHONE_PATTERN.findall(text)
    }


def redact_pii(text: str, types: List[str] = None) -> str:
    """
    Redact PII from text.
    types: list of PII types to redact (default: all)
    """
    if types is None:
        types = ["ssn", "credit_card", "email", "phone"]
    
    result = text
    
    if "ssn" in types:
        result = SSN_PATTERN.sub("[SSN-REDACTED]", result)
    if "credit_card" in types:
        result = CREDIT_CARD_PATTERN.sub("[CC-REDACTED]", result)
    if "email" in types:
        result = EMAIL_PATTERN.sub("[EMAIL-REDACTED]", result)
    if "phone" in types:
        result = PHONE_PATTERN.sub("[PHONE-REDACTED]", result)
    
    return result


def check_policy_flags() -> Dict[str, bool]:
    """
    Check tenant/org policy flags.
    Returns: {"allow_external_llm": bool, "require_redaction": bool, etc.}
    """
    # Placeholder for future tenant-specific policy checks
    return {
        "allow_external_llm": True,
        "require_redaction": False,
        "data_residency": "any"
    }

