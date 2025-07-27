#!/usr/bin/env python3
"""
Script to clear corrupted cache files or migrate old cache format.
Run this if you encounter cache loading errors.
"""

import os
import logging
import pickle
from typing import Dict, Any

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

def migrate_cache_files():
    """Migrate old cache format to new format."""
    cache_files = [
        "system_feedback_cache.pkl",
        "feedback_cache.pkl",
        "cache.pkl"
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    old_cache_data = pickle.load(f)
                
                migrated_count = 0
                new_cache_data = {}
                
                for key, value in old_cache_data.items():
                    if isinstance(value, dict) and 'results' in value:
                        # This is old batch format, extract first result
                        if value['results'] and len(value['results']) > 0:
                            first_result = value['results'][0]
                            if isinstance(first_result, dict):
                                new_cache_data[key] = first_result
                                migrated_count += 1
                    else:
                        # Keep existing format
                        new_cache_data[key] = value
                
                if migrated_count > 0:
                    # Save migrated cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(new_cache_data, f)
                    logger.info(f"✅ Migrated {migrated_count} entries in {cache_file}")
                else:
                    logger.info(f"ℹ️  No migration needed for {cache_file}")
                    
            except Exception as e:
                logger.error(f"❌ Could not migrate {cache_file}: {e}")
        else:
            logger.info(f"ℹ️  Cache file not found: {cache_file}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        print("🔄 Migrating old cache format...")
        migrate_cache_files()
        print("✅ Cache migration complete!")
    else:
        print("🧹 Clearing corrupted cache files...")
        clear_cache_files()
        print("✅ Cache clearing complete!")
        print("\n💡 To migrate instead of clear, run: python clear_cache.py migrate") 