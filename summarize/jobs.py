"""
Job storage and lifecycle management for Summarize tasks.
"""
import threading
from typing import Dict, Any
from datetime import datetime


# In-memory job storage (migrate to DB later)
SUMMARIZE_JOBS: Dict[str, Dict[str, Any]] = {}
SUMMARIZE_JOBS_LOCK = threading.Lock()


def create_job(job_id: str, params: Dict[str, Any]) -> None:
    """Create a new summarize job."""
    with SUMMARIZE_JOBS_LOCK:
        SUMMARIZE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.utcnow().isoformat(),
            "finished_at": None,
            "params": params,
            "result": None,
            "error": None,
            "metrics": {}
        }


def update_job(job_id: str, updates: Dict[str, Any]) -> None:
    """Update job status/progress."""
    with SUMMARIZE_JOBS_LOCK:
        if job_id in SUMMARIZE_JOBS:
            SUMMARIZE_JOBS[job_id].update(updates)


def get_job(job_id: str) -> Dict[str, Any]:
    """Get job by ID."""
    with SUMMARIZE_JOBS_LOCK:
        return SUMMARIZE_JOBS.get(job_id)


def set_job_result(job_id: str, result: Dict[str, Any], metrics: Dict[str, Any] = None) -> None:
    """Set job result and mark as completed."""
    with SUMMARIZE_JOBS_LOCK:
        if job_id in SUMMARIZE_JOBS:
            SUMMARIZE_JOBS[job_id]["result"] = result
            SUMMARIZE_JOBS[job_id]["status"] = "completed"
            SUMMARIZE_JOBS[job_id]["progress"] = 100
            SUMMARIZE_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()
            if metrics:
                SUMMARIZE_JOBS[job_id]["metrics"] = metrics


def set_job_error(job_id: str, error: str) -> None:
    """Set job error and mark as failed."""
    with SUMMARIZE_JOBS_LOCK:
        if job_id in SUMMARIZE_JOBS:
            SUMMARIZE_JOBS[job_id]["error"] = error
            SUMMARIZE_JOBS[job_id]["status"] = "failed"
            SUMMARIZE_JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat()

