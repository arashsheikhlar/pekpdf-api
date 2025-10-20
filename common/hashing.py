"""
File hashing utilities for cache keys.
"""
import hashlib


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for caching."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        raise Exception(f"Failed to hash file: {str(e)}")


def compute_cache_key(file_hash: str, params: dict) -> str:
    """
    Compute cache key from file hash and parameters.
    Params should be a dict of request parameters.
    """
    # Sort params for consistency
    param_str = "_".join(f"{k}={v}" for k, v in sorted(params.items()) if v)
    combined = f"{file_hash}_{param_str}"
    return hashlib.sha256(combined.encode()).hexdigest()

