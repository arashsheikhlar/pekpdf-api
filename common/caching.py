"""
In-memory caching for AI results (can be extended to Redis later).
"""
from typing import Optional, Dict, Any
import threading


class SimpleCache:
    """Thread-safe in-memory cache."""
    
    def __init__(self, max_size: int = 100):
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            return self._cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set item in cache with LRU eviction."""
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Simple FIFO eviction (can be improved to LRU)
                first_key = next(iter(self._cache))
                del self._cache[first_key]
            self._cache[key] = value
    
    def clear(self):
        """Clear all cache."""
        with self._lock:
            self._cache.clear()


# Global cache instances
explain_cache = SimpleCache(max_size=50)
summarize_cache = SimpleCache(max_size=50)

