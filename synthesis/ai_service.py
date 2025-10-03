from typing import Callable, List, Tuple
import re


class AIServiceRouter:
    """Simple AI service abstraction with provider fallbacks and response normalization.

    callers: list of (name, callable) where callable takes (prompt: str) -> str
    """

    def __init__(self, callers: List[Tuple[str, Callable[[str], str]]], min_len: int = 40):
        self.callers = callers or []
        self.min_len = max(0, int(min_len))

    def _normalize(self, text: str) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        s = text.strip()
        # Strip triple backticks blocks if the whole response is fenced
        if s.startswith("```") and s.endswith("```"):
            s = re.sub(r"^```(?:[a-zA-Z]+)?\n?", "", s).rstrip("`")
        # Remove spurious JSON-like wrappers if they occur (rare for synthesis)
        return s.strip()

    def _looks_valid(self, text: str) -> bool:
        if not text:
            return False
        if text.lower().startswith("error:"):
            return False
        if len(text) < self.min_len:
            # Still accept very short but non-error responses? No, require min length
            return False
        return True

    def generate(self, prompt: str) -> str:
        last = ""
        for name, caller in self.callers:
            try:
                raw = caller(prompt)
                norm = self._normalize(raw)
                if self._looks_valid(norm):
                    return norm
                last = norm or last
            except Exception as e:
                last = f"Error: {name} failed: {e}"
                continue
        # If none valid, return the last best attempt (even if short) to help debugging
        return last or "Error: No AI providers returned a valid response." 