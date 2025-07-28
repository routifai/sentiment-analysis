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
    batch_size: int = 10
    cache_enabled: bool = True
    cache_file: str = "system_feedback_cache.pkl"
    max_tokens: int = 4000
    temperature: float = 0.2

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SystemFeedbackConfirmation(BaseModel):
    """Model for system feedback confirmation results."""
    has_system_feedback: bool
    confidence_score: float
    reasoning: str
    feedback_type: str = "general"  # system, response, accuracy, etc.

class BatchSystemFeedbackConfirmation(BaseModel):
    """Model for batch system feedback confirmation results."""
    results: List[SystemFeedbackConfirmation]

# ============================================================================
# IMPROVED PREPROCESSOR FOR SYSTEM FEEDBACK
# ============================================================================

class SystemFeedbackPreprocessor:
    """Handles text preprocessing specifically for system/chatbot feedback."""
    
    def __init__(self):
        self.system_feedback_keywords = self._load_system_feedback_keywords()
        self.system_feedback_patterns = self._load_system_feedback_patterns()
        self.system_indicators = self._load_system_indicators()
    
    def _load_system_feedback_keywords(self) -> List[str]:
        """Load keywords specifically for system/chatbot feedback."""
        return [
            # System-specific feedback terms
            'response', 'answer', 'reply', 'output', 'result', 'summary', 'summarization',
            'generation', 'generated', 'ai', 'chatbot', 'assistant', 'system',
            'model', 'algorithm', 'processing', 'analysis', 'interpretation',
            
            # Quality indicators for system responses
            'accurate', 'inaccurate', 'correct', 'incorrect', 'wrong', 'right',
            'complete', 'incomplete', 'detailed', 'brief', 'comprehensive',
            'relevant', 'irrelevant', 'helpful', 'unhelpful', 'useful', 'useless',
            'clear', 'unclear', 'confusing', 'understandable', 'readable',
            
            # System performance terms
            'slow', 'fast', 'quick', 'responsive', 'lag', 'delay', 'timeout',
            'error', 'bug', 'glitch', 'crash', 'freeze', 'hang', 'broken',
            'working', 'not working', 'doesn\'t work', 'failed', 'successful',
            
            # Response-specific feedback
            'missing', 'included', 'excluded', 'focused', 'unfocused',
            'detailed', 'vague', 'specific', 'general', 'precise', 'imprecise',
            'thorough', 'superficial', 'deep', 'shallow', 'comprehensive',
            
            # Content quality terms
            'quality', 'poor', 'excellent', 'good', 'bad', 'terrible',
            'satisfied', 'dissatisfied', 'happy', 'unhappy', 'pleased', 'displeased',
            'meets expectations', 'exceeds expectations', 'below expectations',
            
            # Improvement suggestions
            'should', 'could', 'would', 'might', 'need', 'want', 'expect',
            'improve', 'better', 'worse', 'enhance', 'fix', 'correct',
            'add', 'remove', 'include', 'exclude', 'focus', 'emphasize'
        ]
    
    def _load_system_feedback_patterns(self) -> List[str]:
        """Load regex patterns specifically for system feedback detection."""
        return [
            # System response patterns
            r'\b(response|answer|reply|output|result)\b',
            r'\b(summary|summarization|generation|generated)\b',
            r'\b(ai|chatbot|assistant|system|model)\b',
            
            # Quality assessment patterns
            r'\b(accurate|inaccurate|correct|incorrect|wrong|right)\b',
            r'\b(complete|incomplete|detailed|brief|comprehensive)\b',
            r'\b(relevant|irrelevant|helpful|unhelpful|useful|useless)\b',
            r'\b(clear|unclear|confusing|understandable|readable)\b',
            
            # Performance patterns
            r'\b(slow|fast|quick|responsive|lag|delay|timeout)\b',
            r'\b(error|bug|glitch|crash|freeze|hang|broken)\b',
            r'\b(working|not working|doesn\'t work|failed|successful)\b',
            
            # Content feedback patterns
            r'\b(missing|included|excluded|focused|unfocused)\b',
            r'\b(detailed|vague|specific|general|precise|imprecise)\b',
            r'\b(thorough|superficial|deep|shallow|comprehensive)\b',
            
            # Quality assessment patterns
            r'\b(quality|poor|excellent|good|bad|terrible)\b',
            r'\b(satisfied|dissatisfied|happy|unhappy|pleased|displeased)\b',
            r'\b(meets expectations|exceeds expectations|below expectations)\b',
            
            # Improvement patterns
            r'\b(should|could|would|might|need|want|expect)\b',
            r'\b(improve|better|worse|enhance|fix|correct)\b',
            r'\b(add|remove|include|exclude|focus|emphasize)\b'
        ]
    
    def _load_system_indicators(self) -> List[str]:
        """Load indicators that suggest the feedback is about the system."""
        return [
            # Direct system references
            'you', 'your', 'this', 'that', 'it', 'the system', 'the ai',
            'the chatbot', 'the assistant', 'the model', 'the algorithm',
            
            # Response-specific references
            'the response', 'the answer', 'the output', 'the result',
            'the summary', 'the generation', 'the analysis',
            
            # Action-based indicators
            'should', 'could', 'would', 'need to', 'must', 'have to',
            'improve', 'fix', 'change', 'modify', 'adjust', 'update',
            
            # Comparison indicators
            'instead of', 'rather than', 'instead', 'better', 'worse',
            'more', 'less', 'too much', 'too little', 'not enough'
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text or pd.isna(text):
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\']', ' ', text)
        return text
    
    def extract_system_keywords(self, text: str) -> List[str]:
        """Extract system-specific feedback keywords from text."""
        if not text:
            return []
        
        clean_text = self.clean_text(text)
        found_keywords = []
        
        for keyword in self.system_feedback_keywords:
            if keyword in clean_text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def count_system_pattern_matches(self, text: str) -> int:
        """Count system feedback pattern matches."""
        if not text:
            return 0
        
        matches = 0
        for pattern in self.system_feedback_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        
        return matches
    
    def has_system_indicators(self, text: str) -> bool:
        """Check if text contains indicators that suggest system feedback."""
        if not text:
            return False
        
        clean_text = self.clean_text(text)
        
        # Check for system indicators
        for indicator in self.system_indicators:
            if indicator in clean_text:
                return True
        
        # Check for direct system references
        system_refs = ['you', 'your', 'this response', 'that answer', 'the system']
        for ref in system_refs:
            if ref in clean_text:
                return True
        
        return False
    
    def calculate_system_feedback_score(self, text: str) -> Tuple[List[str], int, float, bool]:
        """Calculate comprehensive system feedback score."""
        keywords = self.extract_system_keywords(text)
        pattern_matches = self.count_system_pattern_matches(text)
        has_indicators = self.has_system_indicators(text)
        
        # Calculate weighted score with system indicator bonus
        keyword_weight = 1.0
        pattern_weight = 0.5
        indicator_bonus = 2.0 if has_indicators else 0.0
        
        score = (len(keywords) * keyword_weight) + (pattern_matches * pattern_weight) + indicator_bonus
        
        return keywords, pattern_matches, score, has_indicators
    
    def is_potential_system_feedback(self, text: str) -> bool:
        """Determine if text is potential system feedback."""
        keywords, pattern_matches, score, has_indicators = self.calculate_system_feedback_score(text)
        
        # Must have system indicators to be considered system feedback
        if not has_indicators:
            return False
        
        # Core system feedback terms get priority
        has_core_system_terms = any(term in self.clean_text(text) for term in 
                                  ['response', 'answer', 'output', 'result', 'summary', 'generation'])
        
        return (
            has_indicators and (
                len(keywords) >= 2 or
                pattern_matches >= 2 or
                score >= 3.0 or
                has_core_system_terms
            )
        )

# ============================================================================
# IMPROVED LLM CLIENT
# ============================================================================

class SystemFeedbackLLMClient:
    """Handles LLM interactions specifically for system feedback analysis."""
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.client = openai.OpenAI(api_key=config.openai_api_key)
        self.instructor_client = instructor.from_openai(self.client)
    
    def analyze_single_message(self, text: str, keywords: List[str], score: float, has_indicators: bool) -> SystemFeedbackConfirmation:
        """Analyze a single message for system feedback tone."""
        prompt = self._create_single_message_prompt(text, keywords, score, has_indicators)
        
        try:
            response = self.instructor_client.chat.completions.create(
                model=self.config.model,
                response_model=SystemFeedbackConfirmation,
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
            return SystemFeedbackConfirmation(
                has_system_feedback=False,
                confidence_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                feedback_type="error"
            )
    
    def analyze_batch_messages(self, messages_data: List[dict]) -> List[SystemFeedbackConfirmation]:
        """Analyze batch of messages for system feedback tone."""
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
                response_model=BatchSystemFeedbackConfirmation,
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
    
    def _fallback_to_individual(self, messages_data: List[dict]) -> List[SystemFeedbackConfirmation]:
        """Fallback to individual message processing."""
        logger.info("Falling back to individual message processing")
        results = []
        
        for msg_data in messages_data:
            result = self.analyze_single_message(
                msg_data['text'],
                msg_data['keywords'],
                msg_data['score'],
                msg_data['has_indicators']
            )
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for system feedback analysis."""
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
            prompt += f"Has system indicators: {msg_data['has_indicators']}\n\n"
        
        prompt += """CRITICAL: Return exactly one SystemFeedbackConfirmation per input, in the same order (1, 2, 3...).
        For each input, determine if it contains SYSTEM feedback about the AI/chatbot.
        Only classify as system feedback if it's specifically about the AI system itself."""
        
        return prompt

# ============================================================================
# IMPROVED CACHE MANAGER
# ============================================================================

class SystemFeedbackCacheManager:
    """Handles caching functionality for system feedback."""
    
    def __init__(self, cache_file: str, enabled: bool = True):
        self.cache_file = cache_file
        self.enabled = enabled
        self.cache: Dict[str, SystemFeedbackConfirmation] = self._load_cache()
    
    def _load_cache(self) -> Dict[str, SystemFeedbackConfirmation]:
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
    
    def get(self, text: str) -> Optional[SystemFeedbackConfirmation]:
        """Get cached result."""
        if not self.enabled:
            return None
        
        key = self.get_cache_key(text)
        return self.cache.get(key)
    
    def set(self, text: str, result: SystemFeedbackConfirmation):
        """Cache result."""
        if not self.enabled:
            return
        
        key = self.get_cache_key(text)
        self.cache[key] = result

# ============================================================================
# IMPROVED DATABASE MANAGER
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
# MAIN IMPROVED FEEDBACK ANALYZER
# ============================================================================

class ImprovedSystemFeedbackAnalyzer:
    """Main system feedback analysis orchestrator."""
    
    def __init__(self, db_config: DatabaseConfig, analyzer_config: AnalyzerConfig):
        self.config = analyzer_config
        self.db_manager = DatabaseManager(db_config)
        self.preprocessor = SystemFeedbackPreprocessor()
        self.cache_manager = SystemFeedbackCacheManager(
            analyzer_config.cache_file, 
            analyzer_config.cache_enabled
        )
        self.llm_client = SystemFeedbackLLMClient(analyzer_config)
        
        logger.info(f"Initialized Improved System Feedback Analyzer with:")
        logger.info(f"  Model: {analyzer_config.model}")
        logger.info(f"  Batch size: {analyzer_config.batch_size}")
        logger.info(f"  Cache enabled: {analyzer_config.cache_enabled}")
    
    def analyze_messages(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Main method to analyze messages for system feedback tone."""
        start_time = time.time()
        
        try:
            # Step 1: Fetch messages
            messages_df = self.db_manager.fetch_messages(limit)
            if messages_df.empty:
                logger.warning("No messages found")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(messages_df)} messages")
            
            # Step 2: Preprocess to identify potential system feedback
            potential_system_feedback, non_system_feedback = self._preprocess_messages(messages_df)
            
            # Step 3: Analyze potential system feedback with LLM
            system_feedback_results = self._analyze_potential_system_feedback(potential_system_feedback)
            
            # Step 4: Combine results
            final_results = self._combine_results(potential_system_feedback, non_system_feedback, system_feedback_results)
            
            # Step 5: Save results
            results_df = pd.DataFrame(final_results)
            filename = self._save_results(results_df)
            
            # Step 6: Log summary
            self._log_summary(results_df, start_time, len(non_system_feedback), filename)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
        finally:
            # Always save cache
            self.cache_manager.save_cache()
    
    def _preprocess_messages(self, messages_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
        """Preprocess messages to identify potential system feedback."""
        potential_system_feedback = []
        non_system_feedback = []
        
        for _, row in messages_df.iterrows():
            keywords, pattern_matches, score, has_indicators = self.preprocessor.calculate_system_feedback_score(row['input'])
            
            message_data = {
                'row': row,
                'keywords': keywords,
                'pattern_matches': pattern_matches,
                'score': score,
                'has_indicators': has_indicators
            }
            
            if self.preprocessor.is_potential_system_feedback(row['input']):
                potential_system_feedback.append(message_data)
            else:
                non_system_feedback.append(message_data)
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Potential system feedback: {len(potential_system_feedback)}")
        logger.info(f"  Non-system feedback: {len(non_system_feedback)}")
        logger.info(f"  LLM call reduction: {len(non_system_feedback)/len(messages_df)*100:.1f}%")
        
        return potential_system_feedback, non_system_feedback
    
    def _analyze_potential_system_feedback(self, potential_system_feedback: List[dict]) -> List[SystemFeedbackConfirmation]:
        """Analyze potential system feedback messages with LLM."""
        if not potential_system_feedback:
            return []
        
        logger.info(f"Analyzing {len(potential_system_feedback)} potential system feedback messages with LLM")
        all_results = []
        
        # Process in batches
        for i in range(0, len(potential_system_feedback), self.config.batch_size):
            batch = potential_system_feedback[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            total_batches = (len(potential_system_feedback) + self.config.batch_size - 1) // self.config.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} messages)")
            
            # Check cache first
            uncached_batch, cached_results = self._check_cache(batch)
            
            # Process uncached messages
            if uncached_batch:
                batch_data = [
                    {
                        'text': msg['row']['input'],
                        'keywords': msg['keywords'],
                        'score': msg['score'],
                        'has_indicators': msg['has_indicators']
                    }
                    for msg in uncached_batch
                ]
                
                new_results = self.llm_client.analyze_batch_messages(batch_data)
                
                # Validate results
                if len(new_results) != len(uncached_batch):
                    logger.error(f"Result count mismatch in batch {batch_num}")
                    # Pad missing results
                    while len(new_results) < len(uncached_batch):
                        new_results.append(SystemFeedbackConfirmation(
                            has_system_feedback=False,
                            confidence_score=0.0,
                            reasoning="Missing result - batch processing error",
                            feedback_type="error"
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
    
    def _check_cache(self, batch: List[dict]) -> Tuple[List[dict], List[SystemFeedbackConfirmation]]:
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
    
    def _combine_results(self, potential_system_feedback: List[dict], 
                        non_system_feedback: List[dict], 
                        system_feedback_results: List[SystemFeedbackConfirmation]) -> List[dict]:
        """Combine all analysis results."""
        final_results = []
        
        # Add confirmed system feedback results
        for i, msg in enumerate(potential_system_feedback):
            if i < len(system_feedback_results):
                result = system_feedback_results[i]
            else:
                result = SystemFeedbackConfirmation(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Missing LLM result",
                    feedback_type="error"
                )
            
            final_results.append(self._create_result_dict(msg, result, 'potential_system_feedback'))
        
        # Add non-system feedback results
        for msg in non_system_feedback:
            result = SystemFeedbackConfirmation(
                has_system_feedback=False,
                confidence_score=1.0,
                reasoning="Preprocessing: No system feedback indicators detected",
                feedback_type="non_system"
            )
            final_results.append(self._create_result_dict(msg, result, 'non_system_feedback'))
        
        return final_results
    
    def _create_result_dict(self, msg_data: dict, 
                           result: SystemFeedbackConfirmation, 
                           preprocessing_result: str) -> dict:
        """Create result dictionary."""
        return {
            'emp_id': msg_data['row']['emp_id'],
            'session_id': msg_data['row']['session_id'],
            'input_text': msg_data['row']['input'],
            'chat_type': msg_data['row']['chat_type'],
            'timestamp': msg_data['row']['timestamp'],
            'has_system_feedback': result.has_system_feedback,
            'confidence_score': result.confidence_score,
            'reasoning': result.reasoning,
            'feedback_type': result.feedback_type,
            'system_keywords': ', '.join(msg_data['keywords']),
            'keyword_count': len(msg_data['keywords']),
            'pattern_matches': msg_data['pattern_matches'],
            'system_feedback_score': msg_data['score'],
            'has_system_indicators': msg_data['has_indicators'],
            'preprocessing_result': preprocessing_result
        }
    
    def _save_results(self, results_df: pd.DataFrame) -> str:
        """Save results to CSV file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_feedback_analysis_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        return filename
    
    def _log_summary(self, results_df: pd.DataFrame, start_time: float, 
                    non_system_feedback_count: int, filename: str):
        """Log analysis summary."""
        processing_time = time.time() - start_time
        system_feedback_count = len(results_df[results_df['has_system_feedback'] == True])
        
        logger.info("=" * 60)
        logger.info("SYSTEM FEEDBACK ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        logger.info(f"Total messages: {len(results_df)}")
        logger.info(f"System feedback messages: {system_feedback_count} ({system_feedback_count/len(results_df)*100:.1f}%)")
        logger.info(f"LLM calls saved: {non_system_feedback_count} ({non_system_feedback_count/len(results_df)*100:.1f}%)")
        logger.info(f"Cache size: {len(self.cache_manager.cache)} entries")
        logger.info(f"Results file: {filename}")
        logger.info("=" * 60)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for the improved system feedback analyzer."""
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
            batch_size=10,
            cache_enabled=True
        )
        
        # Initialize and run analyzer
        analyzer = ImprovedSystemFeedbackAnalyzer(db_config, analyzer_config)
        results = analyzer.analyze_messages(limit=1000)  # Process first 1000 messages
        
        # Display results summary
        if not results.empty:
            system_feedback_count = len(results[results['has_system_feedback'] == True])
            print(f"\n🎯 SYSTEM FEEDBACK ANALYSIS COMPLETE!")
            print(f"📊 Total messages analyzed: {len(results)}")
            print(f"💬 Messages with system feedback: {system_feedback_count}")
            print(f"📈 System feedback rate: {system_feedback_count/len(results)*100:.1f}%")
            
            # Show example system feedback messages
            system_feedback_examples = results[results['has_system_feedback'] == True].head(3)
            if not system_feedback_examples.empty:
                print(f"\n📝 Example system feedback messages:")
                for i, (_, row) in enumerate(system_feedback_examples.iterrows(), 1):
                    print(f"\n{i}. Confidence: {row['confidence_score']:.2f}")
                    print(f"   Type: {row['feedback_type']}")
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