import psycopg2
import pandas as pd
import re
import json
import logging
import pickle
import hashlib
import os
import time
import openai
import instructor
from typing import List, Dict, Tuple, Optional
from pydantic import BaseModel
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION CLASSES
# ============================================================================

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str

@dataclass
class AnalyzerConfig:
    """Analyzer configuration."""
    openai_api_key: str
    model: str = "gpt-4o"
    batch_size: int = 10  # Smaller for better accuracy
    cache_enabled: bool = True
    cache_file: str = "feedback_cache.pkl"
    max_tokens: int = 4000
    temperature: float = 0.2

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class FeedbackConfirmation(BaseModel):
    """Model for individual feedback confirmation results."""
    has_feedback_tone: bool
    confidence_score: float
    reasoning: str

class BatchFeedbackConfirmation(BaseModel):
    """Model for batch feedback confirmation results."""
    results: List[FeedbackConfirmation]

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Handles all database operations."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def get_connection(self):
        """Get database connection."""
        try:
            return psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def fetch_messages(self, limit: Optional[int] = None, offset: int = 0) -> pd.DataFrame:
        """Fetch messages from database with pagination."""
        query = """
            SELECT emp_id, session_id, input, output, chat_type, timestamp
            FROM chat_messages
            ORDER BY timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn)
            logger.info(f"Fetched {len(df)} messages (offset: {offset})")
            return df

# ============================================================================
# PREPROCESSOR
# ============================================================================

class FeedbackPreprocessor:
    """Handles text preprocessing and keyword analysis."""
    
    def __init__(self):
        self.feedback_keywords = self._load_feedback_keywords()
        self.feedback_patterns = self._load_feedback_patterns()
    
    def _load_feedback_keywords(self) -> List[str]:
        """Load feedback keywords."""
        return [
            # Core feedback terms
            'feedback', 'review', 'rating', 'opinion', 'suggestion', 'recommendation',
            'comment', 'evaluation', 'assessment', 'thought', 'experience',
            
            # Positive indicators
            'thank you', 'thanks', 'appreciate', 'helpful', 'useful', 'great',
            'excellent', 'good', 'amazing', 'love', 'like', 'pleased', 'satisfied',
            'worked', 'solved', 'improved', 'better', 'best', 'perfect',
            
            # Negative indicators
            'problem', 'issue', 'trouble', 'broken', 'failed', 'error',
            'bad', 'poor', 'terrible', 'disappointed', 'frustrated', 'confused',
            'difficult', 'hard', 'slow', 'doesn\'t work', 'not working',
            'useless', 'unhelpful', 'unclear', 'complicated', 'buggy',
            
            # Comparative terms
            'prefer', 'compare', 'versus', 'different', 'should', 'could', 
            'would', 'expect', 'hope', 'wish', 'want', 'need'
        ]
    
    def _load_feedback_patterns(self) -> List[str]:
        """Load regex patterns for feedback detection."""
        return [
            r'\b(thank you|thanks|appreciate)\b',
            r'\b(feedback|review|rating|opinion)\b',
            r'\b(good|bad|great|terrible|excellent|poor)\b',
            r'\b(like|love|hate|dislike)\b',
            r'\b(helpful|useful|useless|unhelpful)\b',
            r'\b(work|working|broken|failed)\b',
            r'\b(easy|difficult|hard|simple)\b',
            r'\b(should|could|would|might|need|want)\b',
            r'\b(better|worse|best|worst|improve)\b',
            r'\b(problem|issue|trouble|error)\b'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or pd.isna(text):
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\']', ' ', text)
        return text
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract feedback keywords from text."""
        if not text:
            return []
        
        clean_text = self.clean_text(text)
        found_keywords = []
        
        for keyword in self.feedback_keywords:
            if keyword in clean_text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def count_pattern_matches(self, text: str) -> int:
        """Count feedback pattern matches."""
        if not text:
            return 0
        
        matches = 0
        for pattern in self.feedback_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return matches
    
    def calculate_feedback_score(self, text: str) -> Tuple[List[str], int, float]:
        """Calculate comprehensive feedback score."""
        keywords = self.extract_keywords(text)
        pattern_matches = self.count_pattern_matches(text)
        
        # Calculate weighted score
        keyword_weight = 1.0
        pattern_weight = 0.5
        score = (len(keywords) * keyword_weight) + (pattern_matches * pattern_weight)
        
        return keywords, pattern_matches, score
    
    def is_potential_feedback(self, text: str) -> bool:
        """Determine if text is potential feedback."""
        keywords, pattern_matches, score = self.calculate_feedback_score(text)
        
        # Core feedback terms get priority
        has_core_terms = any(term in self.clean_text(text) for term in 
                           ['feedback', 'review', 'rating', 'opinion', 'suggestion'])
        
        return (
            len(keywords) >= 2 or
            pattern_matches >= 3 or
            score >= 3.0 or
            has_core_terms
        )

# ============================================================================
# CACHE MANAGER
# ============================================================================

class CacheManager:
    """Handles caching functionality."""
    
    def __init__(self, cache_file: str, enabled: bool = True):
        self.cache_file = cache_file
        self.enabled = enabled
        self.cache: Dict[str, FeedbackConfirmation] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, FeedbackConfirmation]:
        """Load cache from file."""
        if not self.enabled or not os.path.exists(self.cache_file):
            return {}
        
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache: {e}")
            return {}
    
    def save_cache(self):
        """Save cache to file."""
        if not self.enabled:
            return
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        if not text or pd.isna(text):
            text = "EMPTY_INPUT"
        return hashlib.md5(f"{text}:{len(text)}".encode()).hexdigest()
    
    def get(self, text: str) -> Optional[FeedbackConfirmation]:
        """Get cached result."""
        if not self.enabled:
            return None
        
        key = self.get_cache_key(text)
        return self.cache.get(key)
    
    def set(self, text: str, result: FeedbackConfirmation):
        """Cache result."""
        if not self.enabled:
            return
        
        key = self.get_cache_key(text)
        self.cache[key] = result

# ============================================================================
# LLM CLIENT
# ============================================================================

class LLMClient:
    """Handles all LLM interactions."""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.instructor_client = instructor.from_openai(self.client)
    
    def analyze_single_message(self, text: str, keywords: List[str], score: float) -> FeedbackConfirmation:
        """Analyze a single message for feedback tone."""
        prompt = self._create_single_message_prompt(text, keywords, score)
        
        try:
            response = self.instructor_client.chat.completions.create(
                model=self.config.model,
                response_model=FeedbackConfirmation,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=self.config.temperature
            )
            return response
        
        except Exception as e:
            logger.error(f"Error analyzing single message: {e}")
            return FeedbackConfirmation(
                has_feedback_tone=False,
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}"
            )
    
    def analyze_batch_messages(self, messages_data: List[dict]) -> List[FeedbackConfirmation]:
        """Analyze batch of messages for feedback tone."""
        if not messages_data:
            return []
        
        prompt = self._create_batch_prompt(messages_data)
        
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
                response_model=BatchFeedbackConfirmation,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.config.temperature
            )
            
            # Validate response count matches input count
            if len(response.results) != len(messages_data):
                logger.warning(f"Batch result mismatch: expected {len(messages_data)}, got {len(response.results)}")
                return self._fallback_to_individual(messages_data)
            
            return response.results
        
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return self._fallback_to_individual(messages_data)
    
    def _fallback_to_individual(self, messages_data: List[dict]) -> List[FeedbackConfirmation]:
        """Fallback to individual message processing."""
        logger.info("Falling back to individual message processing")
        results = []
        
        for msg_data in messages_data:
            result = self.analyze_single_message(
                msg_data['text'],
                msg_data['keywords'],
                msg_data['score']
            )
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for feedback analysis."""
        return """You are analyzing user inputs to determine if they contain feedback tone. 
        Focus on whether the user is providing feedback, opinions, or evaluations - not just sentiment.
        Be precise and consistent in your analysis."""
    
    def _create_single_message_prompt(self, text: str, keywords: List[str], score: float) -> str:
        """Create prompt for single message analysis."""
        return f"""Analyze this user input for feedback tone:

INPUT: "{text}"
Keywords found: {', '.join(keywords) if keywords else 'None'}
Feedback score: {score:.1f}

Determine if this contains feedback tone based on:
1. Does the user express opinions, evaluations, or reactions?
2. Are they providing thoughts, suggestions, or assessments?
3. Is this more than just a factual question or request?

Respond with has_feedback_tone (True/False), confidence_score (0-1), and brief reasoning."""
    
    def _create_batch_prompt(self, messages_data: List[dict]) -> str:
        """Create prompt for batch analysis."""
        prompt = "Analyze each input for feedback tone. Return results in the EXACT same order as inputs:\n\n"
        
        for i, msg_data in enumerate(messages_data, 1):
            prompt += f"INPUT {i}: \"{msg_data['text']}\"\n"
            prompt += f"Keywords: {', '.join(msg_data['keywords']) if msg_data['keywords'] else 'None'}\n"
            prompt += f"Score: {msg_data['score']:.1f}\n\n"
        
        prompt += """CRITICAL: Return exactly one FeedbackConfirmation per input, in the same order (1, 2, 3...).
        For each input, determine if it contains feedback tone.
        Focus on feedback detection, not sentiment analysis."""
        
        return prompt

# ============================================================================
# MAIN FEEDBACK ANALYZER
# ============================================================================

class FeedbackAnalyzer:
    """Main feedback analysis orchestrator."""
    
    def __init__(self, db_config: DatabaseConfig, analyzer_config: AnalyzerConfig):
        self.config = analyzer_config
        self.db_manager = DatabaseManager(db_config)
        self.preprocessor = FeedbackPreprocessor()
        self.cache_manager = CacheManager(
            analyzer_config.cache_file, 
            analyzer_config.cache_enabled
        )
        self.llm_client = LLMClient(analyzer_config)
        
        logger.info(f"Initialized Feedback Analyzer with:")
        logger.info(f"  Model: {analyzer_config.model}")
        logger.info(f"  Batch size: {analyzer_config.batch_size}")
        logger.info(f"  Cache enabled: {analyzer_config.cache_enabled}")
    
    def analyze_messages(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Main method to analyze messages for feedback tone."""
        start_time = time.time()
        
        try:
            # Step 1: Fetch messages
            messages_df = self.db_manager.fetch_messages(limit)
            if messages_df.empty:
                logger.warning("No messages found")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(messages_df)} messages")
            
            # Step 2: Preprocess to identify potential feedback
            potential_feedback, non_feedback = self._preprocess_messages(messages_df)
            
            # Step 3: Analyze potential feedback with LLM
            feedback_results = self._analyze_potential_feedback(potential_feedback)
            
            # Step 4: Combine results
            final_results = self._combine_results(potential_feedback, non_feedback, feedback_results)
            
            # Step 5: Save results
            results_df = pd.DataFrame(final_results)
            filename = self._save_results(results_df)
            
            # Step 6: Log summary
            self._log_summary(results_df, start_time, len(non_feedback), filename)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            # Always save cache
            self.cache_manager.save_cache()
    
    def _preprocess_messages(self, messages_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
        """Preprocess messages to identify potential feedback."""
        potential_feedback = []
        non_feedback = []
        
        for _, row in messages_df.iterrows():
            keywords, pattern_matches, score = self.preprocessor.calculate_feedback_score(row['input'])
            
            message_data = {
                'row': row,
                'keywords': keywords,
                'pattern_matches': pattern_matches,
                'score': score
            }
            
            if self.preprocessor.is_potential_feedback(row['input']):
                potential_feedback.append(message_data)
            else:
                non_feedback.append(message_data)
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Potential feedback: {len(potential_feedback)}")
        logger.info(f"  Non-feedback: {len(non_feedback)}")
        logger.info(f"  LLM call reduction: {len(non_feedback)/len(messages_df)*100:.1f}%")
        
        return potential_feedback, non_feedback
    
    def _analyze_potential_feedback(self, potential_feedback: List[dict]) -> List[FeedbackConfirmation]:
        """Analyze potential feedback messages with LLM."""
        if not potential_feedback:
            return []
        
        logger.info(f"Analyzing {len(potential_feedback)} potential feedback messages with LLM")
        all_results = []
        
        # Process in batches
        for i in range(0, len(potential_feedback), self.config.batch_size):
            batch = potential_feedback[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (len(potential_feedback) + self.config.batch_size - 1) // self.config.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)")
            
            # Check cache first
            uncached_batch, cached_results = self._check_cache(batch)
            
            # Process uncached messages
            if uncached_batch:
                batch_data = [
                    {
                        'text': msg['row']['input'],
                        'keywords': msg['keywords'],
                        'score': msg['score']
                    }
                    for msg in uncached_batch
                ]
                
                new_results = self.llm_client.analyze_batch_messages(batch_data)
                
                # Validate results
                if len(new_results) != len(uncached_batch):
                    logger.error(f"Result count mismatch in batch {batch_num}")
                    # Pad missing results
                    while len(new_results) < len(uncached_batch):
                        new_results.append(FeedbackConfirmation(
                            has_feedback_tone=False,
                            confidence_score=0.0,
                            reasoning="Missing result - batch processing error"
                        ))
                
                # Cache new results
                for msg, result in zip(uncached_batch, new_results):
                    self.cache_manager.set(msg['row']['input'], result)
                
                # Merge results maintaining order
                batch_results = []
                cached_idx = 0
                new_idx = 0
                
                for msg in batch:
                    if any(msg is uncached_msg for uncached_msg in uncached_batch):
                        batch_results.append(new_results[new_idx])
                        new_idx += 1
                    else:
                        batch_results.append(cached_results[cached_idx])
                        cached_idx += 1
                
                all_results.extend(batch_results)
            else:
                all_results.extend(cached_results)
            
            # Save cache periodically
            if batch_num % 5 == 0:
                self.cache_manager.save_cache()
            
            # Rate limiting between batches
            if batch_num < total_batches:
                time.sleep(0.5)
        
        return all_results
    
    def _check_cache(self, batch: List[dict]) -> Tuple[List[dict], List[FeedbackConfirmation]]:
        """Check cache for batch messages."""
        uncached_batch = []
        cached_results = []
        
        for msg in batch:
            cached_result = self.cache_manager.get(msg['row']['input'])
            if cached_result:
                cached_results.append(cached_result)
            else:
                uncached_batch.append(msg)
        
        logger.info(f"Cache stats: {len(cached_results)} cached, {len(uncached_batch)} need processing")
        return uncached_batch, cached_results
    
    def _combine_results(self, potential_feedback: List[dict], 
                        non_feedback: List[dict], 
                        feedback_results: List[FeedbackConfirmation]) -> List[dict]:
        """Combine all analysis results."""
        final_results = []
        
        # Add confirmed feedback results
        for i, msg in enumerate(potential_feedback):
            if i < len(feedback_results):
                result = feedback_results[i]
            else:
                result = FeedbackConfirmation(
                    has_feedback_tone=False,
                    confidence_score=0.0,
                    reasoning="Missing LLM result"
                )
            
            final_results.append(self._create_result_dict(msg, result, 'potential_feedback'))
        
        # Add non-feedback results
        for msg in non_feedback:
            result = FeedbackConfirmation(
                has_feedback_tone=False,
                confidence_score=1.0,
                reasoning="Preprocessing: No feedback indicators detected"
            )
            final_results.append(self._create_result_dict(msg, result, 'non_feedback'))
        
        return final_results
    
    def _create_result_dict(self, msg_data: dict, 
                           result: FeedbackConfirmation, 
                           preprocessing_result: str) -> dict:
        """Create result dictionary."""
        return {
            'emp_id': msg_data['row']['emp_id'],
            'session_id': msg_data['row']['session_id'],
            'input_text': msg_data['row']['input'],
            'chat_type': msg_data['row']['chat_type'],
            'timestamp': msg_data['row']['timestamp'],
            'has_feedback_tone': result.has_feedback_tone,
            'confidence_score': result.confidence_score,
            'reasoning': result.reasoning,
            'feedback_keywords': ', '.join(msg_data['keywords']),
            'keyword_count': len(msg_data['keywords']),
            'pattern_matches': msg_data['pattern_matches'],
            'feedback_score': msg_data['score'],
            'preprocessing_result': preprocessing_result
        }
    
    def _save_results(self, results_df: pd.DataFrame) -> str:
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_analysis_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        return filename
    
    def _log_summary(self, results_df: pd.DataFrame, start_time: float, 
                    non_feedback_count: int, filename: str):
        """Log analysis summary."""
        processing_time = time.time() - start_time
        feedback_count = len(results_df[results_df['has_feedback_tone'] == True])
        
        logger.info("=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Total messages: {len(results_df)}")
        logger.info(f"Feedback messages: {feedback_count} ({feedback_count/len(results_df)*100:.1f}%)")
        logger.info(f"LLM calls saved: {non_feedback_count} ({non_feedback_count/len(results_df)*100:.1f}%)")
        logger.info(f"Cache size: {len(self.cache_manager.cache)} entries")
        logger.info(f"Results file: {filename}")
        logger.info("=" * 60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the feedback analyzer."""
    try:
        # Configuration - Replace with your actual values
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="your_database",
            user="your_username",
            password="your_password"
        )
        
        analyzer_config = AnalyzerConfig(
            openai_api_key="your_openai_api_key",  # Or use os.getenv("OPENAI_API_KEY")
            model="gpt-4o",
            batch_size=10,  # Smaller batches for better accuracy
            cache_enabled=True
        )
        
        # Initialize and run analyzer
        analyzer = FeedbackAnalyzer(db_config, analyzer_config)
        results = analyzer.analyze_messages(limit=1000)  # Process first 1000 messages
        
        # Display results summary
        if not results.empty:
            feedback_count = len(results[results['has_feedback_tone'] == True])
            print(f"\n🎯 FEEDBACK ANALYSIS COMPLETE!")
            print(f"📊 Total messages analyzed: {len(results)}")
            print(f"💬 Messages with feedback: {feedback_count}")
            print(f"📈 Feedback rate: {feedback_count/len(results)*100:.1f}%")
            
            # Show example feedback messages
            feedback_examples = results[results['has_feedback_tone'] == True].head(3)
            if not feedback_examples.empty:
                print(f"\n📝 Example feedback messages:")
                for i, (_, row) in enumerate(feedback_examples.iterrows(), 1):
                    print(f"\n{i}. Confidence: {row['confidence_score']:.2f}")
                    print(f"   Text: {row['input_text'][:100]}...")
                    print(f"   Reason: {row['reasoning']}")
        else:
            print("No results generated. Check your database connection and data.")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        print("💡 Make sure to:")
        print("  - Set up your database connection")
        print("  - Configure your OpenAI API key")
        print("  - Install required packages: pip install openai instructor psycopg2-binary pandas")

if __name__ == "__main__":
    main()