"""
Response normalization for Explain outputs.
"""
import json
import re
from typing import Dict, Any


def parse_llm_response(text: str) -> Dict[str, Any]:
    """
    Parse LLM response, attempting JSON extraction.
    """
    # Try direct JSON parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except:
            pass
    
    # Try to find JSON object in text
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except:
            pass
    
    # Fallback: return raw text wrapped
    return {
        "summary": text[:500],
        "key_concepts": [],
        "explanations": {"content": text},
        "context": "",
        "definitions": {}
    }


def normalize_explain_result(llm_response: str) -> Dict[str, Any]:
    """
    Normalize LLM response to expected Explain schema.
    Returns: {summary, key_concepts, explanations, context, definitions}
    """
    parsed = parse_llm_response(llm_response)
    
    # Ensure all expected fields
    result = {
        "summary": parsed.get("summary", ""),
        "key_concepts": parsed.get("key_concepts", []),
        "explanations": parsed.get("explanations", {}),
        "context": parsed.get("context", ""),
        "definitions": parsed.get("definitions", {})
    }
    
    # Validate types
    if not isinstance(result["key_concepts"], list):
        result["key_concepts"] = []
    
    if not isinstance(result["explanations"], dict):
        result["explanations"] = {"content": str(result["explanations"])}
    
    if not isinstance(result["definitions"], dict):
        result["definitions"] = {}
    
    return result

