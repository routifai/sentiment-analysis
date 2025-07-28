#!/usr/bin/env python3
"""
Example usage of the isolated LLM client for system feedback analysis.
This demonstrates how to initialize the client with different configurations.
"""

import os
from llm_client import (
    create_openai_client, 
    create_custom_client, 
    create_client_from_env,
    GenericLLMClient,
    LLMConfig
)
from pydantic import BaseModel

# Example response model
class SystemFeedbackConfirmation(BaseModel):
    has_system_feedback: bool
    confidence_score: float
    reasoning: str
    feedback_type: str = "general"

def example_openai_client():
    """Example: Using OpenAI with API key."""
    print("🔧 Example 1: OpenAI Client")
    print("-" * 40)
    
    # Initialize OpenAI client
    client = create_openai_client(
        api_key="your-openai-api-key-here",
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.2
    )
    
    # Test connection
    if client.test_connection():
        print("✅ OpenAI client connection successful")
        
        # Example analysis
        result = client.analyze_single_message(
            text="I'm not happy with this summarization, you should focus more on part 2",
            keywords=["summarization", "should"],
            score=3.5,
            has_indicators=True,
            response_model=SystemFeedbackConfirmation
        )
        
        print(f"Analysis result: {result.has_system_feedback}")
        print(f"Confidence: {result.confidence_score}")
        print(f"Reasoning: {result.reasoning}")
    else:
        print("❌ OpenAI client connection failed")

def example_custom_client():
    """Example: Using custom OpenAI-compatible endpoint."""
    print("\n🔧 Example 2: Custom Endpoint Client")
    print("-" * 40)
    
    # Initialize custom client
    client = create_custom_client(
        api_key="your-custom-api-key-here",
        base_url="https://your-custom-endpoint.com/v1",
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.2
    )
    
    # Test connection
    if client.test_connection():
        print("✅ Custom endpoint client connection successful")
    else:
        print("❌ Custom endpoint client connection failed")

def example_env_client():
    """Example: Using client from environment variables."""
    print("\n🔧 Example 3: Environment Variables Client")
    print("-" * 40)
    
    try:
        # Initialize from environment
        client = create_client_from_env()
        
        # Test connection
        if client.test_connection():
            print("✅ Environment client connection successful")
        else:
            print("❌ Environment client connection failed")
            
    except ValueError as e:
        print(f"❌ Environment setup error: {e}")
        print("💡 Set these environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - OPENAI_BASE_URL (optional)")
        print("  - OPENAI_MODEL (optional)")
        print("  - OPENAI_MAX_TOKENS (optional)")
        print("  - OPENAI_TEMPERATURE (optional)")

def example_direct_config():
    """Example: Using direct configuration."""
    print("\n🔧 Example 4: Direct Configuration")
    print("-" * 40)
    
    # Create configuration directly
    config = LLMConfig(
        api_key="your-api-key-here",
        base_url="https://your-endpoint.com/v1",  # Optional
        model="gpt-4o",
        max_tokens=4000,
        temperature=0.2,
        timeout=60,
        max_retries=3,
        retry_delay=1.0
    )
    
    # Create client with config
    client = GenericLLMClient(config)
    
    # Test connection
    if client.test_connection():
        print("✅ Direct config client connection successful")
    else:
        print("❌ Direct config client connection failed")

def example_batch_analysis():
    """Example: Batch analysis with LLM client."""
    print("\n🔧 Example 5: Batch Analysis")
    print("-" * 40)
    
    # Initialize client
    client = create_openai_client(
        api_key="your-openai-api-key-here",
        model="gpt-4o"
    )
    
    # Example batch data
    batch_data = [
        {
            'text': "I'm not happy with this response, it's too vague",
            'keywords': ["response", "vague"],
            'score': 4.0,
            'has_indicators': True
        },
        {
            'text': "Thank you for the help with my job application",
            'keywords': ["thank you", "help"],
            'score': 2.0,
            'has_indicators': False
        },
        {
            'text': "Your summary is incomplete, please add more details",
            'keywords': ["summary", "incomplete", "details"],
            'score': 5.0,
            'has_indicators': True
        }
    ]
    
    # Analyze batch
    try:
        results = client.analyze_batch_messages(batch_data, SystemFeedbackConfirmation)
        
        print(f"Batch analysis completed: {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  System feedback: {result.has_system_feedback}")
            print(f"  Confidence: {result.confidence_score:.2f}")
            print(f"  Type: {result.feedback_type}")
            print(f"  Reasoning: {result.reasoning}")
            
    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")

def main():
    """Main function demonstrating different client configurations."""
    print("🚀 LLM Client Examples")
    print("=" * 50)
    
    # Run examples
    example_openai_client()
    example_custom_client()
    example_env_client()
    example_direct_config()
    example_batch_analysis()
    
    print("\n" + "=" * 50)
    print("📝 Usage Notes:")
    print("1. Replace 'your-api-key-here' with actual API keys")
    print("2. Set environment variables for automatic configuration")
    print("3. Test connections before running analysis")
    print("4. Handle errors gracefully in production")

if __name__ == "__main__":
    main() 