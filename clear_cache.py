#!/usr/bin/env python3
"""
Script to clear corrupted cache files.
Run this if you encounter cache loading errors.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_cache_files():
    """Clear all cache files that might be corrupted."""
    cache_files = [
        "system_feedback_cache.pkl",
        "feedback_cache.pkl",
        "cache.pkl"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                logger.info(f"✅ Cleared cache file: {cache_file}")
            except Exception as e:
                logger.error(f"❌ Could not clear {cache_file}: {e}")
        else:
            logger.info(f"ℹ️  Cache file not found: {cache_file}")

if __name__ == "__main__":
    print("🧹 Clearing corrupted cache files...")
    clear_cache_files()
    print("✅ Cache clearing complete!") 