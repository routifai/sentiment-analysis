#!/usr/bin/env python3
"""
Script to load and display results from existing cache file.
This allows you to use your cached results without running the full analysis again.
"""

import pickle
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cache_results(cache_file: str = "system_feedback_cache.pkl"):
    """Load results from cache file and convert to DataFrame."""
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        
        logger.info(f"Loaded cache with {len(cached_data)} entries")
        
        # Convert cache to list of results
        results = []
        for key, value in cached_data.items():
            if isinstance(value, dict):
                # Handle old batch format
                if 'results' in value and value['results']:
                    first_result = value['results'][0]
                    if isinstance(first_result, dict):
                        results.append({
                            'cache_key': key,
                            'has_system_feedback': first_result.get('has_system_feedback', False),
                            'confidence_score': first_result.get('confidence_score', 0.0),
                            'reasoning': first_result.get('reasoning', ''),
                            'feedback_type': first_result.get('feedback_type', 'general')
                        })
                else:
                    # Handle individual format
                    results.append({
                        'cache_key': key,
                        'has_system_feedback': value.get('has_system_feedback', False),
                        'confidence_score': value.get('confidence_score', 0.0),
                        'reasoning': value.get('reasoning', ''),
                        'feedback_type': value.get('feedback_type', 'general')
                    })
        
        logger.info(f"Converted {len(results)} cache entries to results")
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Could not load cache: {e}")
        return pd.DataFrame()

def analyze_cache_results(df: pd.DataFrame):
    """Analyze the cached results."""
    if df.empty:
        logger.error("No results to analyze")
        return
    
    logger.info("=" * 60)
    logger.info("CACHE RESULTS ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Total cached entries: {len(df)}")
    
    # Count system feedback
    system_feedback_count = len(df[df['has_system_feedback'] == True])
    logger.info(f"System feedback messages: {system_feedback_count} ({system_feedback_count/len(df)*100:.1f}%)")
    
    # Count by feedback type
    feedback_types = df['feedback_type'].value_counts()
    logger.info("\nFeedback types:")
    for feedback_type, count in feedback_types.items():
        logger.info(f"  {feedback_type}: {count} ({count/len(df)*100:.1f}%)")
    
    # Confidence score distribution
    high_confidence = len(df[df['confidence_score'] >= 0.8])
    medium_confidence = len(df[(df['confidence_score'] >= 0.5) & (df['confidence_score'] < 0.8)])
    low_confidence = len(df[df['confidence_score'] < 0.5])
    
    logger.info(f"\nConfidence distribution:")
    logger.info(f"  High (≥0.8): {high_confidence} ({high_confidence/len(df)*100:.1f}%)")
    logger.info(f"  Medium (0.5-0.8): {medium_confidence} ({medium_confidence/len(df)*100:.1f}%)")
    logger.info(f"  Low (<0.5): {low_confidence} ({low_confidence/len(df)*100:.1f}%)")
    
    logger.info("=" * 60)

def save_cache_results(df: pd.DataFrame, filename: str = None):
    """Save cache results to CSV file."""
    if df.empty:
        logger.error("No results to save")
        return
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cache_results_{timestamp}.csv"
    
    df.to_csv(filename, index=False)
    logger.info(f"Results saved to {filename}")
    return filename

def main():
    """Main function to load and analyze cache results."""
    print("🔄 Loading cache results...")
    
    # Load cache results
    df = load_cache_results()
    
    if df.empty:
        print("❌ No cache results found or cache is empty")
        return
    
    # Analyze results
    analyze_cache_results(df)
    
    # Save results
    filename = save_cache_results(df)
    
    print(f"\n✅ Successfully loaded {len(df)} cached results!")
    print(f"📊 Results saved to: {filename}")
    
    # Show some example results
    print(f"\n📝 Example results:")
    system_feedback_examples = df[df['has_system_feedback'] == True].head(3)
    if not system_feedback_examples.empty:
        for i, (_, row) in enumerate(system_feedback_examples.iterrows(), 1):
            print(f"\n{i}. Confidence: {row['confidence_score']:.2f}")
            print(f"   Type: {row['feedback_type']}")
            print(f"   Reason: {row['reasoning'][:100]}...")

if __name__ == "__main__":
    main() 