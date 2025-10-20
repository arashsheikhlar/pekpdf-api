"""
Metrics and observability utilities.
"""
import time
from typing import Dict, Any
from datetime import datetime


class JobMetrics:
    """Track metrics for a job."""
    
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.stages: Dict[str, float] = {}
        self.llm_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = []
    
    def mark_stage(self, stage_name: str):
        """Mark completion of a stage."""
        self.stages[stage_name] = time.time()
    
    def finish(self):
        """Mark job as finished."""
        self.end_time = time.time()
    
    def add_llm_call(self, tokens: int, cost: float = 0.0):
        """Record an LLM call."""
        self.llm_calls += 1
        self.total_tokens += tokens
        self.total_cost += cost
    
    def add_error(self, error: str):
        """Record an error."""
        self.errors.append(error)
    
    def duration(self) -> float:
        """Get total duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dict."""
        return {
            "duration_seconds": self.duration(),
            "llm_calls": self.llm_calls,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "stages": {k: v - self.start_time for k, v in self.stages.items()},
            "errors": self.errors,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None
        }

