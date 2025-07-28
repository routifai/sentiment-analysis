import openai
import instructor
import logging
import time
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# LLM CONFIGURATION
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    api_key: str
    base_url: Optional[str] = None  # For custom endpoints
    model: str = "gpt-4o"
    max_tokens: int = 4000
    temperature: float = 0.2
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

# ============================================================================
# GENERIC LLM CLIENT
# ============================================================================

class GenericLLMClient:
    """Generic LLM client that can work with different OpenAI-compatible endpoints."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = self._initialize_client()
        self.instructor_client = instructor.from_openai(self.client)
        
        logger.info(f"Initialized LLM Client:")
        logger.info(f"  Model: {config.model}")
        logger.info(f"  Base URL: {config.base_url or 'OpenAI Default'}")
        logger.info(f"  Max Tokens: {config.max_tokens}")
        logger.info(f"  Temperature: {config.temperature}")
    
    def _initialize_client(self) -> openai.OpenAI:
        """Initialize OpenAI client with custom configuration."""
        client_kwargs = {
            "api_key": self.config.api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries
        }
        
        # Add base URL if provided (for custom endpoints)
        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url
            logger.info(f"Using custom base URL: {self.config.base_url}")
        
        return openai.OpenAI(**client_kwargs)
    
    def test_connection(self) -> bool:
        """Test the LLM connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            logger.info("✅ LLM connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ LLM connection test failed: {e}")
            return False
    
    def analyze_single_message(self, 
                             text: str, 
                             keywords: List[str], 
                             score: float, 
                             has_indicators: bool,
                             response_model: BaseModel) -> BaseModel:
        """Analyze a single message with the LLM."""
        prompt = self._create_single_message_prompt(text, keywords, score, has_indicators)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.instructor_client.chat.completions.create(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=self.config.temperature
                )
                return response
            
            except openai.BadRequestError as e:
                # Handle content filtering errors (400)
                if "content_filter" in str(e).lower() or "400" in str(e):
                    logger.warning(f"Content filtering error (attempt {attempt + 1}): {e}")
                    return self._create_content_filtered_response(response_model, text)
                else:
                    logger.error(f"Bad request error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        return self._create_error_response(response_model, f"Bad request: {str(e)}")
            
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1) * 2)  # Longer delay for rate limits
                else:
                    return self._create_error_response(response_model, f"Rate limit exceeded: {str(e)}")
            
            except openai.APIError as e:
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return self._create_error_response(response_model, f"API error: {str(e)}")
            
            except Exception as e:
                logger.error(f"Unexpected error analyzing single message (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return self._create_error_response(response_model, f"Unexpected error: {str(e)}")
    
    def analyze_batch_messages(self, 
                              messages_data: List[dict], 
                              response_model: BaseModel) -> List[BaseModel]:
        """Analyze batch of messages with the LLM."""
        if not messages_data:
            return []
        
        prompt = self._create_batch_prompt(messages_data)
        
        for attempt in range(self.config.max_retries):
            try:
                # Calculate dynamic token limit
                base_tokens = 800
                tokens_per_message = 50
                max_tokens = min(
                    base_tokens + (len(messages_data) * tokens_per_message),
                    self.config.max_tokens
                )
                
                logger.info(f"Processing batch of {len(messages_data)} messages with max_tokens={max_tokens}")
                
                response = self.instructor_client.chat.completions.create(
                    model=self.config.model,
                    response_model=response_model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=self.config.temperature
                )
                
                # Validate response count matches input count
                if hasattr(response, 'results') and len(response.results) != len(messages_data):
                    logger.warning(f"Batch result mismatch: expected {len(messages_data)}, got {len(response.results)}")
                    return self._fallback_to_individual(messages_data, response_model)
                
                return response.results if hasattr(response, 'results') else [response]
            
            except openai.BadRequestError as e:
                # Handle content filtering errors (400) in batch
                if "content_filter" in str(e).lower() or "400" in str(e):
                    logger.warning(f"Content filtering error in batch (attempt {attempt + 1}): {e}")
                    logger.info(f"Skipping batch of {len(messages_data)} messages due to content filtering")
                    return self._create_content_filtered_batch_response(messages_data, response_model)
                else:
                    logger.error(f"Bad request error in batch (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        return self._fallback_to_individual(messages_data, response_model)
            
            except openai.RateLimitError as e:
                logger.warning(f"Rate limit error in batch (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1) * 2)  # Longer delay for rate limits
                else:
                    return self._fallback_to_individual(messages_data, response_model)
            
            except openai.APIError as e:
                logger.error(f"API error in batch (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return self._fallback_to_individual(messages_data, response_model)
            
            except Exception as e:
                logger.error(f"Unexpected error in batch analysis (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    return self._fallback_to_individual(messages_data, response_model)
    
    def _fallback_to_individual(self, messages_data: List[dict], response_model: BaseModel) -> List[BaseModel]:
        """Fallback to individual message processing."""
        logger.info("Falling back to individual message processing")
        results = []
        
        for msg_data in messages_data:
            result = self.analyze_single_message(
                msg_data['text'],
                msg_data['keywords'],
                msg_data['score'],
                msg_data.get('has_indicators', False),
                response_model
            )
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def _create_error_response(self, response_model: BaseModel, error_message: str) -> BaseModel:
        """Create an error response based on the response model."""
        try:
            # Try to create error response with common fields
            if hasattr(response_model, 'has_system_feedback'):
                return response_model(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning=f"Analysis failed: {error_message}",
                    feedback_type="error"
                )
            elif hasattr(response_model, 'has_feedback_tone'):
                return response_model(
                    has_feedback_tone=False,
                    confidence_score=0.0,
                    reasoning=f"Analysis failed: {error_message}"
                )
            else:
                # Generic error response
                return response_model()
        except Exception as e:
            logger.error(f"Could not create error response: {e}")
            # Create a minimal response to avoid None
            try:
                return response_model(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning=f"Analysis failed: {error_message}",
                    feedback_type="error"
                )
            except Exception:
                # Last resort - create with minimal fields
                return response_model()
    
    def _create_content_filtered_response(self, response_model: BaseModel, text: str) -> BaseModel:
        """Create a response for content that was filtered by the API."""
        try:
            if hasattr(response_model, 'has_system_feedback'):
                return response_model(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Content filtered by API - unable to analyze",
                    feedback_type="content_filtered"
                )
            elif hasattr(response_model, 'has_feedback_tone'):
                return response_model(
                    has_feedback_tone=False,
                    confidence_score=0.0,
                    reasoning="Content filtered by API - unable to analyze"
                )
            else:
                return response_model()
        except Exception as e:
            logger.error(f"Could not create content filtered response: {e}")
            # Create a minimal response to avoid None
            try:
                return response_model(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Content filtered by API - unable to analyze",
                    feedback_type="content_filtered"
                )
            except Exception:
                # Last resort - create with minimal fields
                return response_model()
    
    def _create_content_filtered_batch_response(self, messages_data: List[dict], response_model: BaseModel) -> List[BaseModel]:
        """Create batch response for content that was filtered by the API."""
        logger.info(f"Creating content filtered responses for batch of {len(messages_data)} messages")
        
        results = []
        for msg_data in messages_data:
            result = self._create_content_filtered_response(response_model, msg_data['text'])
            if result:
                results.append(result)
            else:
                # Fallback error response
                results.append(self._create_error_response(response_model, "Content filtered"))
        
        return results
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for feedback analysis."""
        return """You are analyzing user inputs to determine if they contain feedback about the AI system/chatbot itself.

CRITICAL: Only classify as system feedback if the user is providing feedback about:
1. The AI's response quality, accuracy, or helpfulness
2. The system's performance, speed, or functionality
3. The AI's behavior, capabilities, or limitations
4. Suggestions for improving the AI system
5. Complaints about the AI's responses or behavior

DO NOT classify as system feedback if the user is:
- Providing feedback about other topics (jobs, products, services, etc.)
- Making general statements or opinions unrelated to the AI
- Asking questions or making requests
- Discussing personal matters or other subjects

Focus specifically on feedback directed at the AI system/chatbot, not general feedback or opinions."""
    
    def _create_single_message_prompt(self, text: str, keywords: List[str], score: float, has_indicators: bool) -> str:
        """Create prompt for single message analysis."""
        return f"""Analyze this user input for SYSTEM feedback (feedback about the AI/chatbot itself):

INPUT: "{text}"
System keywords found: {', '.join(keywords) if keywords else 'None'}
System feedback score: {score:.1f}
Has system indicators: {has_indicators}

Determine if this contains SYSTEM feedback based on:
1. Is the user providing feedback about the AI's response quality?
2. Are they commenting on the system's performance or behavior?
3. Are they suggesting improvements to the AI system?
4. Are they complaining about the AI's responses or capabilities?
5. Is this feedback specifically about the AI/chatbot, not other topics?

IMPORTANT: Only classify as system feedback if it's about the AI system itself.

Respond with has_system_feedback (True/False), confidence_score (0-1), reasoning, and feedback_type."""
    
    def _create_batch_prompt(self, messages_data: List[dict]) -> str:
        """Create prompt for batch analysis."""
        prompt = "Analyze each input for SYSTEM feedback (feedback about the AI/chatbot). Return results in the EXACT same order as inputs:\n\n"
        
        for i, msg_data in enumerate(messages_data, 1):
            prompt += f"INPUT {i}: \"{msg_data['text']}\"\n"
            prompt += f"System keywords: {', '.join(msg_data['keywords']) if msg_data['keywords'] else 'None'}\n"
            prompt += f"Score: {msg_data['score']:.1f}\n"
            prompt += f"Has system indicators: {msg_data.get('has_indicators', False)}\n\n"
        
        prompt += """CRITICAL: Return exactly one SystemFeedbackConfirmation per input, in the same order (1, 2, 3...).
        For each input, determine if it contains SYSTEM feedback about the AI/chatbot.
        Only classify as system feedback if it's specifically about the AI system itself."""
        
        return prompt

# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_openai_client(api_key: str, 
                        model: str = "gpt-4o",
                        max_tokens: int = 4000,
                        temperature: float = 0.2) -> GenericLLMClient:
    """Create LLM client for OpenAI."""
    config = LLMConfig(
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return GenericLLMClient(config)

def create_custom_client(api_key: str,
                        base_url: str,
                        model: str = "gpt-4o",
                        max_tokens: int = 4000,
                        temperature: float = 0.2) -> GenericLLMClient:
    """Create LLM client for custom OpenAI-compatible endpoint."""
    config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return GenericLLMClient(config)

def create_client_from_env() -> GenericLLMClient:
    """Create LLM client from environment variables."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4000"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    config = LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return GenericLLMClient(config)

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Example of how to use the LLM client."""
    
    # Example 1: OpenAI with API key
    client1 = create_openai_client(
        api_key="your-openai-api-key",
        model="gpt-4o"
    )
    
    # Example 2: Custom endpoint
    client2 = create_custom_client(
        api_key="your-api-key",
        base_url="https://your-custom-endpoint.com/v1",
        model="gpt-4o"
    )
    
    # Example 3: From environment variables
    client3 = create_client_from_env()
    
    # Test connection
    if client1.test_connection():
        print("✅ OpenAI client connection successful")
    
    if client2.test_connection():
        print("✅ Custom endpoint client connection successful")

if __name__ == "__main__":
    example_usage() 