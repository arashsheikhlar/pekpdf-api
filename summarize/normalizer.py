"""
Response normalization for Summarize outputs.
"""
import json
import re
from typing import Dict, Any


def parse_llm_response(text: str) -> Dict[str, Any]:
    """Parse LLM response, attempting JSON extraction."""
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
        "key_topics": [],
        "main_points": [text[:1000]],
        "recommendations": []
    }


def normalize_summarize_result(llm_response: str) -> Dict[str, Any]:
    """
    Normalize LLM response to expected Summarize schema.
    Returns: {summary, key_topics, main_points, recommendations}
    """
    parsed = parse_llm_response(llm_response)
    
    result = {
        "summary": parsed.get("summary", ""),
        "key_topics": parsed.get("key_topics", []),
        "main_points": parsed.get("main_points", []),
        "recommendations": parsed.get("recommendations", [])
    }
    
    # Validate types
    if not isinstance(result["key_topics"], list):
        result["key_topics"] = []
    
    if not isinstance(result["main_points"], list):
        result["main_points"] = []
    
    if not isinstance(result["recommendations"], list):
        result["recommendations"] = []
    
    return result

