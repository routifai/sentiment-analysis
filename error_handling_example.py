#!/usr/bin/env python3
"""
Example demonstrating the improved error handling in the LLM client.
This shows how the system handles content filtering errors (400) and other API errors.
"""

import logging
from llm_client import create_openai_client, GenericLLMClient
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example response model
class SystemFeedbackConfirmation(BaseModel):
    has_system_feedback: bool
    confidence_score: float
    reasoning: str
    feedback_type: str = "general"

def demonstrate_error_handling():
    """Demonstrate how error handling works."""
    print("🚀 Error Handling Demonstration")
    print("=" * 50)
    
    # Initialize client (replace with your API key)
    client = create_openai_client(
        api_key="your-openai-api-key-here",
        model="gpt-4o"
    )
    
    # Example messages that might trigger different errors
    test_messages = [
        {
            'text': "I'm not happy with this response, it's too vague",
            'keywords': ["response", "vague"],
            'score': 4.0,
            'has_indicators': True
        },
        {
            'text': "Your summary is incomplete, please add more details",
            'keywords': ["summary", "incomplete", "details"],
            'score': 5.0,
            'has_indicators': True
        },
        {
            'text': "Thank you for the help with my job application",
            'keywords': ["thank you", "help"],
            'score': 2.0,
            'has_indicators': False
        }
    ]
    
    print("\n📝 Testing Error Handling Scenarios:")
    print("-" * 40)
    
    # Test individual message analysis
    for i, msg_data in enumerate(test_messages, 1):
        print(f"\n🔍 Testing Message {i}:")
        print(f"Text: {msg_data['text'][:50]}...")
        
        try:
            result = client.analyze_single_message(
                text=msg_data['text'],
                keywords=msg_data['keywords'],
                score=msg_data['score'],
                has_indicators=msg_data['has_indicators'],
                response_model=SystemFeedbackConfirmation
            )
            
            print(f"✅ Result: {result.has_system_feedback}")
            print(f"   Confidence: {result.confidence_score:.2f}")
            print(f"   Type: {result.feedback_type}")
            print(f"   Reasoning: {result.reasoning}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Test batch analysis
    print(f"\n📦 Testing Batch Analysis:")
    print("-" * 40)
    
    try:
        batch_results = client.analyze_batch_messages(test_messages, SystemFeedbackConfirmation)
        
        print(f"✅ Batch analysis completed: {len(batch_results)} results")
        for i, result in enumerate(batch_results, 1):
            print(f"\nResult {i}:")
            print(f"  System feedback: {result.has_system_feedback}")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Type: {result.feedback_type}")
            print(f"  Reasoning: {result.reasoning}")
            
    except Exception as e:
        print(f"❌ Batch analysis error: {e}")

def explain_error_types():
    """Explain the different types of errors handled."""
    print("\n📚 Error Handling Types:")
    print("=" * 50)
    
    print("\n🔴 Content Filtering Errors (400):")
    print("- Triggered when API detects inappropriate content")
    print("- System continues processing other messages")
    print("- Returns 'content_filtered' feedback type")
    print("- Example: Sexual content, violence, etc.")
    
    print("\n🟡 Rate Limit Errors:")
    print("- Triggered when API rate limits are exceeded")
    print("- System waits longer before retrying")
    print("- Falls back to individual processing if needed")
    print("- Example: Too many requests per minute")
    
    print("\n🟠 Bad Request Errors:")
    print("- Triggered for malformed requests")
    print("- System retries with exponential backoff")
    print("- Falls back to individual processing")
    print("- Example: Invalid model name, malformed prompts")
    
    print("\n🔵 API Errors:")
    print("- General API-related errors")
    print("- System retries with backoff")
    print("- Falls back to individual processing")
    print("- Example: Network issues, server errors")
    
    print("\n⚫ Unexpected Errors:")
    print("- Any other unexpected errors")
    print("- System logs error and continues")
    print("- Returns error response type")
    print("- Example: Memory errors, import errors")

def show_error_response_types():
    """Show the different response types for errors."""
    print("\n📊 Error Response Types:")
    print("=" * 50)
    
    print("\n1. Content Filtered Response:")
    print("   - has_system_feedback: False")
    print("   - confidence_score: 0.0")
    print("   - reasoning: 'Content filtered by API - unable to analyze'")
    print("   - feedback_type: 'content_filtered'")
    
    print("\n2. Error Response:")
    print("   - has_system_feedback: False")
    print("   - confidence_score: 0.0")
    print("   - reasoning: 'Analysis failed: [error message]'")
    print("   - feedback_type: 'error'")
    
    print("\n3. Rate Limit Error Response:")
    print("   - has_system_feedback: False")
    print("   - confidence_score: 0.0")
    print("   - reasoning: 'Rate limit exceeded: [error message]'")
    print("   - feedback_type: 'error'")

def main():
    """Main function demonstrating error handling."""
    print("🚀 LLM Client Error Handling Examples")
    print("=" * 60)
    
    # Explain error types
    explain_error_types()
    
    # Show response types
    show_error_response_types()
    
    # Demonstrate error handling (commented out to avoid API calls)
    # demonstrate_error_handling()
    
    print("\n" + "=" * 60)
    print("💡 Key Benefits:")
    print("1. System continues processing even with API errors")
    print("2. Content filtering doesn't stop the entire batch")
    print("3. Detailed error logging for debugging")
    print("4. Graceful fallback to individual processing")
    print("5. Proper error categorization in results")

if __name__ == "__main__":
    main() 