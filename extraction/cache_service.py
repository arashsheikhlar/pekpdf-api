# SPDX-License-Identifier: AGPL-3.0-only

"""
Cache service for extraction results.

This module handles caching of extraction results both in memory and on disk
to improve performance and reduce redundant processing.
"""

import hashlib
import json
import os
import time
from typing import Dict, Optional

from .models import ExtractionResult, CacheEntry, ExtractionOptions


class CacheService:
    """Service for caching extraction results."""
    
    def __init__(self, cache_dir: Optional[str] = None, ttl_seconds: int = 86400, pipeline_version: str = "v3-ai-only"):
        """
        Initialize the cache service.
        
        Args:
            cache_dir: Directory for disk cache (defaults to uploads/extract_cache)
            ttl_seconds: Time to live for cache entries in seconds (default: 1 day)
            pipeline_version: Version identifier for cache invalidation
        """
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.ttl_seconds = ttl_seconds
        self.pipeline_version = pipeline_version
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Ensure cache directory exists
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except Exception:
            # Fallback to current directory if cache dir creation fails
            self.cache_dir = os.getcwd()
    
    def _get_default_cache_dir(self) -> str:
        """Get the default cache directory."""
        try:
            # Try to get upload folder from Flask app config
            from flask import current_app
            base = current_app.config.get("UPLOAD_FOLDER") or os.path.join(os.getcwd(), "uploads")
            return os.path.join(base, "extract_cache")
        except Exception:
            return os.path.join(os.getcwd(), "extract_cache")
    
    def compute_cache_key(self, file_bytes: bytes, options: ExtractionOptions) -> str:
        """
        Compute a cache key for the given file and options.
        
        Args:
            file_bytes: Raw file content
            options: Extraction options
            
        Returns:
            MD5 hash string as cache key
        """
        # Create options identity string
        options_dict = {
            "v": self.pipeline_version,
            "domain_override": (options.domain_override or '').strip().lower(),
            "selected_fields": sorted(options.selected_fields or []),
            "custom_instructions": options.custom_instructions.strip(),
            "enrich": options.enrich,
            "use_ocr": options.use_ocr,
            "extract_tables": options.extract_tables,
            "extract_formulas": options.extract_formulas,
        }
        
        options_identity = json.dumps(options_dict, sort_keys=True)
        return hashlib.md5(file_bytes + b"|" + options_identity.encode("utf-8")).hexdigest()
    
    def get_cached_result(self, cache_key: str) -> Optional[ExtractionResult]:
        """
        Get cached extraction result.
        
        Args:
            cache_key: Cache key to look up
            
        Returns:
            Cached extraction result or None if not found/expired
        """
        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]
            if not entry.is_expired():
                return entry.result
            else:
                # Remove expired entry
                del self.memory_cache[cache_key]
        
        # Check disk cache
        disk_path = self._get_disk_cache_path(cache_key)
        try:
            if os.path.exists(disk_path):
                with open(disk_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check if cache entry is expired
                timestamp = data.get('timestamp', 0)
                if time.time() - timestamp <= self.ttl_seconds:
                    result = ExtractionResult.from_dict(data['result'])
                    # Store in memory cache for faster access
                    self.memory_cache[cache_key] = CacheEntry(
                        result=result,
                        timestamp=timestamp,
                        ttl=self.ttl_seconds
                    )
                    return result
                else:
                    # Remove expired disk cache
                    os.remove(disk_path)
        except Exception:
            pass
        
        return None
    
    def cache_result(self, cache_key: str, result: ExtractionResult) -> None:
        """
        Cache an extraction result.
        
        Args:
            cache_key: Cache key to store under
            result: Extraction result to cache
        """
        timestamp = time.time()
        
        # Store in memory cache
        self.memory_cache[cache_key] = CacheEntry(
            result=result,
            timestamp=timestamp,
            ttl=self.ttl_seconds
        )
        
        # Store in disk cache
        try:
            disk_path = self._get_disk_cache_path(cache_key)
            cache_data = {
                'result': result.to_dict(),
                'timestamp': timestamp,
                'ttl': self.ttl_seconds
            }
            
            with open(disk_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception:
            # Disk cache failure is not critical
            pass
    
    def _get_disk_cache_path(self, cache_key: str) -> str:
        """Get the disk cache path for a cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Clear cache entries.
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if cache_key:
            # Clear specific entry
            self.memory_cache.pop(cache_key, None)
            disk_path = self._get_disk_cache_path(cache_key)
            try:
                if os.path.exists(disk_path):
                    os.remove(disk_path)
            except Exception:
                pass
        else:
            # Clear all cache
            self.memory_cache.clear()
            try:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, filename))
            except Exception:
                pass
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        cleaned = 0
        
        # Clean memory cache
        expired_keys = []
        for key, entry in self.memory_cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned += 1
        
        # Clean disk cache
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        timestamp = data.get('timestamp', 0)
                        if time.time() - timestamp > self.ttl_seconds:
                            os.remove(file_path)
                            cleaned += 1
                    except Exception:
                        # Remove corrupted cache files
                        os.remove(file_path)
                        cleaned += 1
        except Exception:
            pass
        
        return cleaned
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        memory_count = len(self.memory_cache)
        memory_size = sum(len(str(entry.result.to_dict())) for entry in self.memory_cache.values())
        
        disk_count = 0
        disk_size = 0
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    disk_count += 1
                    file_path = os.path.join(self.cache_dir, filename)
                    disk_size += os.path.getsize(file_path)
        except Exception:
            pass
        
        return {
            'memory_entries': memory_count,
            'memory_size_bytes': memory_size,
            'disk_entries': disk_count,
            'disk_size_bytes': disk_size,
            'total_entries': memory_count + disk_count,
            'ttl_seconds': self.ttl_seconds,
            'pipeline_version': self.pipeline_version
        }
