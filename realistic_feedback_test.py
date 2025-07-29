# ============================================================================
# REALISTIC FEEDBACK TEST - Based on actual user patterns
# ============================================================================

import pandas as pd
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_realistic_test_cases():
    """Generate realistic test cases based on actual user feedback patterns."""
    
    # REAL FEEDBACK EXAMPLES (should pass score-aware)
    real_feedback = [
        "Your response was helpful",
        "That didn't work for me",
        "Good answer, thanks",
        "Not quite what I was looking for",
        "Thanks, that helped a lot",
        "Your suggestion was perfect",
        "I like this approach better",
        "This makes sense now",
        "Better than the previous response",
        "Still not working as expected",
        "Your explanation was clear",
        "This is much better",
        "I appreciate the help",
        "That's exactly what I needed",
        "Your answer was spot on",
        "This is working now",
        "Thanks for the clarification",
        "Your response was accurate",
        "I found this helpful",
        "This is exactly right"
    ]
    
    # INSTRUCTION/REQUEST EXAMPLES (should be filtered out)
    instructions = [
        "Please help me write a Python function",
        "Can you explain how machine learning works?",
        "I need you to analyze this dataset",
        "What are the best practices for software development?",
        "Generate a business plan for my startup",
        "Your task is to review this document",
        "Act as a financial advisor",
        "How do I implement a neural network?",
        "Create a marketing strategy",
        "Explain the differences between databases",
        "Please show me how to do this",
        "Can you tell me more about this topic?",
        "I want you to help me with this problem",
        "Generate a comprehensive analysis",
        "Write a detailed report on this subject",
        "Your job is to provide insights",
        "Please assist me with this task",
        "I need your expertise on this matter",
        "Can you guide me through this process?",
        "Help me understand this concept"
    ]
    
    # LONG COMPLEX PROMPTS (should be filtered out)
    long_prompts = [
        "Please help me write a comprehensive Python function that calculates fibonacci numbers and includes proper error handling, input validation, and performance optimization for large numbers",
        "Can you explain in detail how machine learning algorithms work, including supervised learning, unsupervised learning, and reinforcement learning, with specific examples for each type",
        "I need you to analyze this complex dataset and provide detailed insights about customer behavior patterns, purchasing trends, and demographic analysis over the past year",
        "What are the best practices for software development in large teams, including version control, code review processes, testing strategies, and deployment pipelines?",
        "Generate a comprehensive business plan for a startup company including market analysis, competitive landscape, financial projections, marketing strategy, and risk assessment",
        "Your task is to review this document and provide a detailed summary of the key findings, recommendations, and implementation strategies",
        "Act as a financial advisor and help me create an investment portfolio that balances risk and return while considering my long-term financial goals",
        "How do I implement a neural network from scratch using Python, including the mathematical foundations, backpropagation algorithm, and optimization techniques?",
        "Create a detailed marketing strategy for launching a new product in a competitive market, including target audience analysis, positioning, and promotional tactics",
        "Explain the differences between various database management systems and their specific use cases, performance characteristics, and scalability considerations"
    ]
    
    # AMBIGUOUS CASES (edge cases)
    ambiguous = [
        "This is working better now",
        "I like how you explained that",
        "Can you do this differently?",
        "Your approach is interesting",
        "This seems right",
        "I think this is good",
        "Maybe try another way",
        "This looks correct",
        "I'm not sure about this",
        "This could work"
    ]
    
    return {
        'real_feedback': real_feedback,
        'instructions': instructions,
        'long_prompts': long_prompts,
        'ambiguous': ambiguous
    }

def test_score_aware_preprocessing():
    """Test the score-aware preprocessing with realistic cases."""
    
    from score_aware_preprocessor import ScoreAwareSystemFeedbackPreprocessor
    
    preprocessor = ScoreAwareSystemFeedbackPreprocessor()
    test_cases = generate_realistic_test_cases()
    
    print("🧪 REALISTIC FEEDBACK TEST")
    print("=" * 80)
    
    results = {}
    
    for category, texts in test_cases.items():
        print(f"\n📋 {category.upper()}:")
        print("-" * 40)
        
        passed = 0
        total = len(texts)
        
        for text in texts:
            result = preprocessor.is_potential_system_feedback(text)
            if result:
                passed += 1
                status = "✅"
            else:
                status = "❌"
            
            # Get detailed analysis
            analysis = preprocessor.get_classification_reasoning(text)
            score = analysis['score']
            word_count = analysis['word_count']
            
            print(f"{status} Score={score:.1f}, Words={word_count}: {text[:60]}{'...' if len(text) > 60 else ''}")
        
        results[category] = {
            'passed': passed,
            'total': total,
            'rate': passed / total * 100
        }
        
        print(f"   Pass rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    # Summary
    print("\n📊 SUMMARY RESULTS:")
    print("=" * 50)
    
    total_passed = sum(r['passed'] for r in results.values())
    total_messages = sum(r['total'] for r in results.values())
    
    print(f"Overall pass rate: {total_passed}/{total_messages} ({total_passed/total_messages*100:.1f}%)")
    
    print(f"\nCategory breakdown:")
    for category, stats in results.items():
        print(f"  {category}: {stats['passed']}/{stats['total']} ({stats['rate']:.1f}%)")
    
    # Efficiency analysis
    real_feedback_rate = results['real_feedback']['rate']
    instruction_filter_rate = 100 - results['instructions']['rate']
    long_prompt_filter_rate = 100 - results['long_prompts']['rate']
    
    print(f"\n🎯 EFFICIENCY ANALYSIS:")
    print(f"  Real feedback detection: {real_feedback_rate:.1f}%")
    print(f"  Instruction filtering: {instruction_filter_rate:.1f}%")
    print(f"  Long prompt filtering: {long_prompt_filter_rate:.1f}%")
    
    # Expected LLM reduction
    if total_messages > 0:
        overall_rate = total_passed / total_messages
        print(f"  Overall LLM call rate: {overall_rate*100:.1f}%")
        print(f"  Expected LLM reduction: {(1-overall_rate)*100:.1f}%")

if __name__ == "__main__":
    test_score_aware_preprocessing() 