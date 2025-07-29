import psycopg2
import pandas as pd
import re
import json
import logging
import pickle
import hashlib
import os
import time
import asyncio
import concurrent.futures
import threading
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
from enum import Enum

# Import the isolated LLM client
from llm_client import GenericLLMClient, LLMConfig, create_openai_client, create_custom_client, create_client_from_env
from pydantic import BaseModel

class SystemFeedbackConfirmation(BaseModel):
    """Model for system feedback confirmation results."""
    has_system_feedback: bool
    confidence_score: float
    reasoning: str
    feedback_type: str = "general"

class BatchSystemFeedbackConfirmation(BaseModel):
    """Model for batch system feedback confirmation results."""
    results: List[SystemFeedbackConfirmation]

class ClassificationLevel(Enum):
    """Multi-stage classification levels."""
    DEFINITELY_FEEDBACK = "definitely_feedback"
    PROBABLY_FEEDBACK = "probably_feedback" 
    UNCERTAIN_LEAN_FEEDBACK = "uncertain_lean_feedback"
    UNCERTAIN_LEAN_INSTRUCTION = "uncertain_lean_instruction"
    PROBABLY_INSTRUCTION = "probably_instruction"
    DEFINITELY_INSTRUCTION = "definitely_instruction"

@dataclass
class MessageSignals:
    """Lightweight signal container optimized for speed."""
    # Core signals (fast to compute)
    word_count: int
    has_question: bool
    has_past_ref: bool  # "your response was", "that didn't work"
    has_imperative: bool  # "please help", "can you"
    has_evaluation: bool  # "good", "bad", "wrong", "helpful"
    
    # Advanced signals (computed on demand)
    temporal_score: float = 0.0
    instruction_score: float = 0.0
    feedback_score: float = 0.0
    confidence: float = 0.0

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str

@dataclass
class EnhancedAnalyzerConfig:
    """Enhanced analyzer configuration."""
    llm_client: GenericLLMClient
    batch_size: int = 15  # Slightly larger batches
    parallel_batches: int = 8  # More parallelism
    cache_enabled: bool = True
    cache_file: str = "enhanced_feedback_cache.pkl"
    cache_save_frequency: int = 25  # Less frequent saves for performance
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Multi-stage filtering thresholds
    stage1_target_rate: float = 0.15  # Select ~15% in stage 1 (75k from 500k)
    stage2_target_rate: float = 0.40  # Select ~40% in stage 2 (30k from 75k)
    uncertain_threshold: float = 0.3   # Send uncertain cases to LLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltraFastSignalExtractor:
    """Optimized signal extraction with precompiled patterns and caching."""
    
    def __init__(self):
        # Precompile all patterns for performance
        self._compile_patterns()
        self._init_word_sets()
        self._pattern_cache = {}  # Thread-local pattern cache
        self._cache_lock = threading.Lock()
    
    def _compile_patterns(self):
        """Precompile regex patterns for maximum performance."""
        self.patterns = {
            # FEEDBACK PATTERNS - optimized for common cases
            'past_reference': re.compile(r'\b(your|the)\s+(response|answer|output|suggestion|advice|reply)\s+(was|were|seemed|didn\'t|helped|worked)\b', re.IGNORECASE),
            'past_action': re.compile(r'\b(you|that|this)\s+(did|didn\'t|was|were|seemed|appeared|helped|worked|failed|gave|provided)\b', re.IGNORECASE),
            'evaluation_quick': re.compile(r'\b(good|bad|great|poor|excellent|terrible|wrong|right|correct|helpful|useless|perfect|awful)\b', re.IGNORECASE),
            'satisfaction': re.compile(r'\b(thanks|thank\s+you|satisfied|disappointed|impressed|confused|unclear)\b', re.IGNORECASE),
            
            # INSTRUCTION PATTERNS - catch obvious commands
            'imperative_strong': re.compile(r'^\s*(can|could|would|will|please)\s+you\s+(help|assist|show|tell|explain|do|make|create|write|analyze|review)\b', re.IGNORECASE),
            'question_command': re.compile(r'^\s*(what|how|where|when|why|who|which)\s+.{10,}\?', re.IGNORECASE),
            'task_assignment': re.compile(r'\b(your\s+)?(task|job|role|assignment)\s+(is|will\s+be)\b', re.IGNORECASE),
            'direct_request': re.compile(r'\b(i\s+need|i\s+want|i\s+would\s+like)\s+you\s+to\b', re.IGNORECASE),
            
            # MIXED SIGNALS
            'continuation': re.compile(r'\b(also|but|however|additionally|furthermore)\b', re.IGNORECASE),
            'comparison': re.compile(r'\b(better|worse|more|less)\s+(than|helpful|clear|accurate)\b', re.IGNORECASE)
        }
    
    def _init_word_sets(self):
        """Initialize word sets for fast lookup."""
        self.feedback_words = {
            'positive': {'good', 'great', 'excellent', 'perfect', 'helpful', 'clear', 'accurate', 'useful', 'right', 'correct'},
            'negative': {'bad', 'poor', 'terrible', 'wrong', 'incorrect', 'unhelpful', 'unclear', 'useless', 'awful'},
            'satisfaction': {'thanks', 'satisfied', 'disappointed', 'impressed', 'appreciate', 'grateful'},
            'temporal': {'was', 'were', 'did', 'didn\'t', 'had', 'gave', 'provided', 'seemed', 'appeared'}
        }
        
        self.instruction_words = {
            'requests': {'can', 'could', 'would', 'will', 'please', 'help', 'assist', 'show', 'tell', 'explain'},
            'tasks': {'analyze', 'review', 'examine', 'create', 'write', 'generate', 'make', 'build', 'design'},
            'questions': {'what', 'how', 'where', 'when', 'why', 'who', 'which', 'is', 'are', 'do', 'does'}
        }
    
    def extract_signals_ultra_fast(self, text: str) -> MessageSignals:
        """Ultra-fast signal extraction optimized for 500k messages."""
        if not text or pd.isna(text):
            return MessageSignals(0, False, False, False, False)
        
        # Basic metrics (fastest operations first)
        words = text.split()
        word_count = len(words)
        has_question = '?' in text
        
        # Early termination for obvious cases
        if word_count > 150:  # Very long = likely instruction
            return MessageSignals(word_count, has_question, False, True, False, 0.0, 5.0, 0.0, 0.9)
        
        if word_count < 3:  # Too short = likely not meaningful
            return MessageSignals(word_count, has_question, False, False, False, 0.0, 0.0, 0.0, 0.1)
        
        # Convert to lowercase once for all operations
        text_lower = text.lower()
        
        # Fast pattern matching (most discriminative patterns first)
        has_past_ref = bool(self.patterns['past_reference'].search(text) or self.patterns['past_action'].search(text))
        has_imperative = bool(self.patterns['imperative_strong'].search(text) or self.patterns['direct_request'].search(text))
        has_evaluation = bool(self.patterns['evaluation_quick'].search(text) or self.patterns['satisfaction'].search(text))
        
        # Fast word-based scoring using sets
        words_lower = set(word.strip('.,!?()[]') for word in words[:20])  # Only check first 20 words
        
        feedback_score = self._calculate_feedback_score_fast(words_lower, has_past_ref, has_evaluation)
        instruction_score = self._calculate_instruction_score_fast(words_lower, has_imperative, has_question, word_count)
        
        return MessageSignals(
            word_count=word_count,
            has_question=has_question,
            has_past_ref=has_past_ref,
            has_imperative=has_imperative,
            has_evaluation=has_evaluation,
            feedback_score=feedback_score,
            instruction_score=instruction_score,
            confidence=abs(feedback_score - instruction_score) / max(feedback_score + instruction_score, 1.0)
        )
    
    def _calculate_feedback_score_fast(self, words: Set[str], has_past_ref: bool, has_evaluation: bool) -> float:
        """Fast feedback scoring using set intersections."""
        score = 0.0
        
        # Word-based signals
        score += len(words & self.feedback_words['positive']) * 2.0
        score += len(words & self.feedback_words['negative']) * 2.0
        score += len(words & self.feedback_words['satisfaction']) * 1.5
        score += len(words & self.feedback_words['temporal']) * 1.0
        
        # Pattern-based signals
        if has_past_ref:
            score += 3.0
        if has_evaluation:
            score += 2.0
        
        return score
    
    def _calculate_instruction_score_fast(self, words: Set[str], has_imperative: bool, has_question: bool, word_count: int) -> float:
        """Fast instruction scoring using set intersections."""
        score = 0.0
        
        # Word-based signals
        score += len(words & self.instruction_words['requests']) * 2.0
        score += len(words & self.instruction_words['tasks']) * 2.5
        score += len(words & self.instruction_words['questions']) * 1.0
        
        # Pattern-based signals
        if has_imperative:
            score += 4.0
        if has_question and word_count > 15:
            score += 2.0
        
        # Length penalty for instructions
        if word_count > 50:
            score += 1.0
        if word_count > 100:
            score += 2.0
        
        return score

class MultiStageClassifier:
    """Multi-stage classifier with optimized thresholds."""
    
    def __init__(self, config: EnhancedAnalyzerConfig):
        self.config = config
        self.signal_extractor = UltraFastSignalExtractor()
        self.stats = {
            'stage1_processed': 0,
            'stage1_selected': 0,
            'stage2_processed': 0,
            'stage2_selected': 0,
            'definitely_feedback': 0,
            'probably_feedback': 0,
            'uncertain': 0,
            'filtered_out': 0
        }
        self.stats_lock = threading.Lock()
    
    def classify_message(self, text: str) -> Tuple[ClassificationLevel, MessageSignals, str]:
        """Multi-stage classification with detailed reasoning."""
        signals = self.signal_extractor.extract_signals_ultra_fast(text)
        
        # STAGE 1: Quick elimination of obvious cases
        level, reasoning = self._stage1_classification(signals)
        
        # STAGE 2: Refined analysis for uncertain cases
        if level in [ClassificationLevel.UNCERTAIN_LEAN_FEEDBACK, ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION]:
            level, reasoning = self._stage2_classification(signals, text, reasoning)
        
        # Update stats (thread-safe)
        with self.stats_lock:
            if level == ClassificationLevel.DEFINITELY_FEEDBACK:
                self.stats['definitely_feedback'] += 1
            elif level == ClassificationLevel.PROBABLY_FEEDBACK:
                self.stats['probably_feedback'] += 1
            elif level in [ClassificationLevel.UNCERTAIN_LEAN_FEEDBACK, ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION]:
                self.stats['uncertain'] += 1
            else:
                self.stats['filtered_out'] += 1
        
        return level, signals, reasoning
    
    def _stage1_classification(self, signals: MessageSignals) -> Tuple[ClassificationLevel, str]:
        """Stage 1: Fast classification based on strong signals."""
        with self.stats_lock:
            self.stats['stage1_processed'] += 1
        
        # DEFINITE FEEDBACK - strong positive signals
        if (signals.feedback_score >= 4.0 and signals.instruction_score <= 2.0 and 
            signals.word_count <= 60 and signals.has_past_ref):
            return ClassificationLevel.DEFINITELY_FEEDBACK, f"Strong feedback signals (score: {signals.feedback_score:.1f}, past ref: True)"
        
        # DEFINITE INSTRUCTION - strong negative signals
        if (signals.instruction_score >= 6.0 and signals.feedback_score <= 1.0):
            return ClassificationLevel.DEFINITELY_INSTRUCTION, f"Strong instruction signals (score: {signals.instruction_score:.1f})"
        
        if (signals.word_count > 120 and signals.has_question):
            return ClassificationLevel.DEFINITELY_INSTRUCTION, f"Long question ({signals.word_count} words)"
        
        if (signals.has_imperative and signals.word_count > 40 and not signals.has_past_ref):
            return ClassificationLevel.DEFINITELY_INSTRUCTION, f"Clear imperative command ({signals.word_count} words)"
        
        # PROBABLE FEEDBACK - good signals but not overwhelming
        if (signals.feedback_score >= 2.0 and signals.feedback_score > signals.instruction_score * 1.5 and 
            signals.word_count <= 80):
            return ClassificationLevel.PROBABLY_FEEDBACK, f"Good feedback signals (f: {signals.feedback_score:.1f} vs i: {signals.instruction_score:.1f})"
        
        # PROBABLE INSTRUCTION - clear but not overwhelming
        if (signals.instruction_score >= 3.0 and signals.instruction_score > signals.feedback_score * 2.0):
            return ClassificationLevel.PROBABLY_INSTRUCTION, f"Clear instruction signals (i: {signals.instruction_score:.1f} vs f: {signals.feedback_score:.1f})"
        
        # UNCERTAIN CASES - need more analysis
        if signals.feedback_score > signals.instruction_score:
            return ClassificationLevel.UNCERTAIN_LEAN_FEEDBACK, f"Lean feedback (f: {signals.feedback_score:.1f} vs i: {signals.instruction_score:.1f})"
        else:
            return ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION, f"Lean instruction (i: {signals.instruction_score:.1f} vs f: {signals.feedback_score:.1f})"
    
    def _stage2_classification(self, signals: MessageSignals, text: str, stage1_reasoning: str) -> Tuple[ClassificationLevel, str]:
        """Stage 2: Refined analysis for uncertain cases."""
        with self.stats_lock:
            self.stats['stage2_processed'] += 1
        
        # Additional pattern analysis for edge cases
        text_lower = text.lower()
        
        # Look for mixed signals (feedback + new request)
        has_continuation = any(word in text_lower for word in ['also', 'but', 'however', 'additionally'])
        has_thank_plus_request = 'thank' in text_lower and any(word in text_lower for word in ['can you', 'could you', 'please'])
        
        # Feedback with continuation (still feedback)
        if (signals.has_evaluation and has_continuation and signals.word_count <= 100):
            return ClassificationLevel.PROBABLY_FEEDBACK, f"Feedback with continuation | {stage1_reasoning}"
        
        # Thank you + request (lean instruction)
        if has_thank_plus_request:
            return ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION, f"Thanks + request pattern | {stage1_reasoning}"
        
        # Short evaluation without clear instruction
        if (signals.has_evaluation and signals.word_count <= 30 and not signals.has_imperative):
            return ClassificationLevel.PROBABLY_FEEDBACK, f"Short evaluation | {stage1_reasoning}"
        
        # Medium length with question
        if (signals.has_question and 20 <= signals.word_count <= 60 and signals.feedback_score < 1.0):
            return ClassificationLevel.PROBABLY_INSTRUCTION, f"Medium question | {stage1_reasoning}"
        
        # Keep original classification if no new evidence
        return ClassificationLevel.UNCERTAIN_LEAN_FEEDBACK if signals.feedback_score >= signals.instruction_score else ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION, f"Stage 2 uncertain | {stage1_reasoning}"
    
    def should_send_to_llm(self, level: ClassificationLevel) -> bool:
        """Determine if message should be sent to LLM."""
        return level in [
            ClassificationLevel.DEFINITELY_FEEDBACK,
            ClassificationLevel.PROBABLY_FEEDBACK,
            ClassificationLevel.UNCERTAIN_LEAN_FEEDBACK
        ]
    
    def get_stats(self) -> Dict[str, int]:
        """Get classification statistics."""
        with self.stats_lock:
            total = self.stats['stage1_processed']
            if total == 0:
                return self.stats.copy()
            
            stats = self.stats.copy()
            selected = stats['definitely_feedback'] + stats['probably_feedback'] + stats['uncertain']
            stats['total_selected'] = selected
            stats['selection_rate'] = selected / total if total > 0 else 0
            return stats

class ThreadSafeCacheManager:
    """Thread-safe cache manager optimized for concurrent access."""
    
    def __init__(self, cache_file: str, enabled: bool = True, save_frequency: int = 25):
        self.cache_file = cache_file
        self.enabled = enabled
        self.save_frequency = save_frequency
        self.cache: Dict[str, SystemFeedbackConfirmation] = self._load_cache()
        self.cache_lock = threading.RLock()  # Re-entrant lock
        self.unsaved_entries = 0
        self.stats = {'hits': 0, 'misses': 0, 'sets': 0}
        
    def _load_cache(self) -> Dict[str, SystemFeedbackConfirmation]:
        """Load cache with error recovery."""
        if not self.enabled or not os.path.exists(self.cache_file):
            return {}
        
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                converted_cache = {}
                
                for key, value in cached_data.items():
                    try:
                        if isinstance(value, dict):
                            converted_cache[key] = SystemFeedbackConfirmation(**value)
                        else:
                            converted_cache[key] = value
                    except Exception:
                        continue
                
                logger.info(f"💾 Cache loaded: {len(converted_cache)} entries")
                return converted_cache
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")
            return {}
    
    def save_cache(self, force: bool = False):
        """Thread-safe cache saving."""
        if not self.enabled or (not force and self.unsaved_entries < self.save_frequency):
            return
        
        with self.cache_lock:
            try:
                cache_dict = {}
                for key, value in self.cache.items():
                    if hasattr(value, 'model_dump'):
                        cache_dict[key] = value.model_dump()
                    elif hasattr(value, 'dict'):
                        cache_dict[key] = value.dict()
                    else:
                        cache_dict[key] = value
                
                temp_file = self.cache_file + '.tmp'
                with open(temp_file, 'wb') as f:
                    pickle.dump(cache_dict, f)
                os.replace(temp_file, self.cache_file)
                
                logger.info(f"💾 Cache saved: {len(cache_dict)} entries")
                self.unsaved_entries = 0
            except Exception as e:
                logger.warning(f"Cache save failed: {e}")
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{text}:{len(text)}".encode()).hexdigest()
    
    def get(self, text: str) -> Optional[SystemFeedbackConfirmation]:
        """Thread-safe cache get."""
        if not self.enabled:
            return None
        
        key = self.get_cache_key(text)
        with self.cache_lock:
            result = self.cache.get(key)
            if result:
                self.stats['hits'] += 1
            else:
                self.stats['misses'] += 1
            return result
    
    def set(self, text: str, result: SystemFeedbackConfirmation):
        """Thread-safe cache set."""
        if not self.enabled:
            return
        
        key = self.get_cache_key(text)
        with self.cache_lock:
            if key not in self.cache:
                self.unsaved_entries += 1
            self.cache[key] = result
            self.stats['sets'] += 1
            
            if self.unsaved_entries >= self.save_frequency:
                self.save_cache()
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            return {
                'cache_size': len(self.cache),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'sets': self.stats['sets'],
                'hit_rate': self.stats['hits'] / max(1, total_requests),
                'unsaved_entries': self.unsaved_entries
            }

class DatabaseManager:
    """Database manager with connection pooling and months-based filtering."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
    
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password
        )
    
    def fetch_messages(self, months_back: Optional[int] = None, limit: Optional[int] = None, 
                      offset: int = 0) -> pd.DataFrame:
        """
        Fetch messages with months-based filtering from current date.
        
        Args:
            months_back: Number of months to go back from today (1, 2, 3, etc.)
            limit: Maximum number of messages (optional, for testing)
            offset: Number of messages to skip (for pagination)
        """
        
        # Base query
        query = """
            SELECT emp_id, session_id, input, output, chat_type, timestamp
            FROM chat_messages
        """
        
        params = []
        
        # Add time filtering if specified
        if months_back is not None:
            query += " WHERE timestamp >= NOW() - INTERVAL '%s months'"
            params.append(months_back)
            time_filter_desc = f"last {months_back} month{'s' if months_back != 1 else ''}"
        else:
            time_filter_desc = "all time"
        
        # Always order by timestamp descending (newest first)
        query += " ORDER BY timestamp DESC"
        
        # Add pagination if specified
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        
        logger.info(f"🗓️  Fetching messages from {time_filter_desc}")
        if limit:
            logger.info(f"📊 Query limit: {limit:,} messages, offset: {offset:,}")
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                # Log time range of actual data
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                logger.info(f"📥 Fetched {len(df):,} messages")
                logger.info(f"📅 Date range: {min_date} to {max_date}")
                
                # Show monthly distribution if we have multiple months
                if months_back and months_back > 1:
                    monthly_counts = df.groupby(df['timestamp'].dt.to_period('M')).size()
                    logger.info(f"📊 Monthly distribution:")
                    for period, count in monthly_counts.head(months_back).items():
                        logger.info(f"    {period}: {count:,} messages")
            else:
                logger.warning(f"❌ No messages found for the specified time period")
            
            return df
    
    def get_date_range_info(self) -> Dict[str, any]:
        """Get information about available date ranges in the database."""
        
        query = """
            SELECT 
                MIN(timestamp) as earliest_message,
                MAX(timestamp) as latest_message,
                COUNT(*) as total_messages,
                COUNT(DISTINCT DATE_TRUNC('month', timestamp)) as months_available
            FROM chat_messages
        """
        
        with self.get_connection() as conn:
            result = pd.read_sql_query(query, conn)
            
            if not result.empty and result.iloc[0]['total_messages'] > 0:
                info = {
                    'earliest_message': result.iloc[0]['earliest_message'],
                    'latest_message': result.iloc[0]['latest_message'],
                    'total_messages': int(result.iloc[0]['total_messages']),
                    'months_available': int(result.iloc[0]['months_available'])
                }
                
                # Calculate how many months back we can go
                if info['earliest_message'] and info['latest_message']:
                    earliest = pd.to_datetime(info['earliest_message'])
                    latest = pd.to_datetime(info['latest_message'])
                    months_span = (latest.year - earliest.year) * 12 + (latest.month - earliest.month)
                    info['max_months_back'] = months_span
                
                logger.info(f"📊 Database date range info:")
                logger.info(f"    Total messages: {info['total_messages']:,}")
                logger.info(f"    Date range: {info['earliest_message']} to {info['latest_message']}")
                logger.info(f"    Months available: {info['months_available']}")
                logger.info(f"    Max months back: {info.get('max_months_back', 'unknown')}")
                
                return info
            else:
                logger.warning("❌ No messages found in database")
                return {}

class OptimizedBatchProcessor:
    """Optimized batch processor with better error handling."""
    
    def __init__(self, llm_client: GenericLLMClient, max_workers: int = 8):
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
    
    def process_batch_with_retry(self, batch_data: List[dict], batch_id: int, 
                               cache_manager: ThreadSafeCacheManager) -> Tuple[int, List[SystemFeedbackConfirmation]]:
        """Process batch with improved retry logic."""
        
        for attempt in range(3):
            try:
                logger.info(f"🔄 Processing batch {batch_id} ({len(batch_data)} messages, attempt {attempt + 1})")
                
                prompt = self._create_optimized_prompt(batch_data)
                
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.config.model,
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=self.llm_client.config.max_tokens,
                    temperature=0.1,  # Lower temperature for consistency
                    timeout=90.0
                )
                
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response")
                
                # Improved JSON parsing
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                    else:
                        raise ValueError("No valid JSON found")
                
                if 'results' not in result_data:
                    raise ValueError("Missing 'results' field")
                
                results = []
                for i, result in enumerate(result_data['results']):
                    try:
                        results.append(SystemFeedbackConfirmation(**result))
                    except Exception as e:
                        logger.warning(f"Result {i} parse error: {e}")
                        results.append(SystemFeedbackConfirmation(
                            has_system_feedback=False,
                            confidence_score=0.0,
                            reasoning=f"Parse error: {str(e)}",
                            feedback_type="error"
                        ))
                
                # Ensure correct number of results
                while len(results) < len(batch_data):
                    results.append(SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning="Missing result",
                        feedback_type="error"
                    ))
                
                # Cache results immediately
                for i, result in enumerate(results[:len(batch_data)]):
                    cache_manager.set(batch_data[i]['text'], result)
                
                logger.info(f"✅ Batch {batch_id} complete")
                return batch_id, results[:len(batch_data)]
                
            except Exception as e:
                logger.warning(f"❌ Batch {batch_id} attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep((attempt + 1) * 2)
                else:
                    logger.error(f"💥 Batch {batch_id} failed completely")
                    fallback_results = [
                        SystemFeedbackConfirmation(
                            has_system_feedback=False,
                            confidence_score=0.0,
                            reasoning=f"Batch failed: {str(e)}",
                            feedback_type="error"
                        ) for _ in batch_data
                    ]
                    return batch_id, fallback_results
    
    def _create_optimized_prompt(self, batch_data: List[dict]) -> str:
        """Create optimized prompt with better examples."""
        
        prompt = """You are an expert at identifying FEEDBACK about previous AI responses vs NEW INSTRUCTIONS to AI systems.

SYSTEM FEEDBACK = User commenting on/evaluating a PREVIOUS AI response
USER INSTRUCTION = User asking AI to perform a NEW task

FEEDBACK Examples (TRUE):
✅ "Your response was helpful"
✅ "That answer seems wrong" 
✅ "Good explanation, thanks"
✅ "This didn't address my question"
✅ "Better than last time"
✅ "Thanks, but can you also check X" (feedback + new request)

INSTRUCTION Examples (FALSE):
❌ "Can you help me with this problem?"
❌ "Please analyze this document"
❌ "What is the capital of France?"
❌ "Your task is to summarize..."
❌ "I need you to verify this information"

DECISION CRITERIA:
- Past tense about AI output = FEEDBACK
- Quality judgments (good/bad/helpful) = FEEDBACK  
- Future requests/questions = INSTRUCTION
- Task assignments = INSTRUCTION

For each message, determine:
1. has_system_feedback: true/false
2. confidence_score: 0.0-1.0
3. reasoning: Brief explanation
4. feedback_type: "system_evaluation", "user_instruction", "mixed", or "ambiguous"

Return JSON:
{
  "results": [
    {
      "has_system_feedback": boolean,
      "confidence_score": float,
      "reasoning": "string",
      "feedback_type": "system_evaluation|user_instruction|mixed|ambiguous"
    }
  ]
}

Messages to analyze:
"""
        
        for i, msg in enumerate(batch_data):
            # Include preprocessing info for better LLM decision making
            preprocessing_info = f" [preprocessing: {msg.get('preprocessing_level', 'unknown')}]"
            prompt += f"\n{i+1}. \"{msg['text']}\"{preprocessing_info}"
        
        return prompt
    
    def process_batches_parallel(self, all_batches: List[List[dict]], 
                               cache_manager: ThreadSafeCacheManager) -> List[List[SystemFeedbackConfirmation]]:
        """Process batches in parallel with progress tracking."""
        
        logger.info(f"🚀 Processing {len(all_batches)} batches with {self.max_workers} workers")
        
        future_to_batch = {}
        for batch_id, batch_data in enumerate(all_batches):
            future = self.executor.submit(self.process_batch_with_retry, batch_data, batch_id, cache_manager)
            future_to_batch[future] = batch_id
        
        results = [None] * len(all_batches)
        completed = 0
        
        for future in concurrent.futures.as_completed(future_to_batch, timeout=600):
            try:
                batch_id, batch_results = future.result()
                results[batch_id] = batch_results
                completed += 1
                
                # Progress logging
                if completed % 5 == 0 or completed == len(all_batches):
                    progress = completed / len(all_batches) * 100
                    cache_stats = cache_manager.get_stats()
                    logger.info(f"📊 Progress: {completed}/{len(all_batches)} ({progress:.1f}%) | "
                              f"Cache: {cache_stats['cache_size']} entries, {cache_stats['hit_rate']:.1%} hit rate")
                
            except Exception as e:
                batch_id = future_to_batch[future]
                logger.error(f"💥 Batch {batch_id} execution failed: {e}")
                
                batch_size = len(all_batches[batch_id])
                fallback_results = [
                    SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning=f"Execution failed: {str(e)}",
                        feedback_type="error"
                    ) for _ in range(batch_size)
                ]
                results[batch_id] = fallback_results
                completed += 1
        
        # Final cache save
        cache_manager.save_cache(force=True)
        
        # Ensure no None results
        for i, result in enumerate(results):
            if result is None:
                batch_size = len(all_batches[i])
                results[i] = [
                    SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning="Missing batch result",
                        feedback_type="error"
                    ) for _ in range(batch_size)
                ]
        
        return results

class EnhancedSystemFeedbackAnalyzer:
    """Main analyzer with multi-stage classification and optimized processing."""
    
    def __init__(self, db_config: DatabaseConfig, analyzer_config: EnhancedAnalyzerConfig):
        self.config = analyzer_config
        self.db_manager = DatabaseManager(db_config)
        self.classifier = MultiStageClassifier(analyzer_config)
        self.cache_manager = ThreadSafeCacheManager(
            analyzer_config.cache_file, 
            analyzer_config.cache_enabled,
            analyzer_config.cache_save_frequency
        )
        self.llm_client = analyzer_config.llm_client
        
        logger.info(f"🚀 Enhanced System Feedback Analyzer Initialized:")
        logger.info(f"  Model: {analyzer_config.llm_client.config.model}")
        logger.info(f"  Batch size: {analyzer_config.batch_size}")
        logger.info(f"  Parallel batches: {analyzer_config.parallel_batches}")
        logger.info(f"  Multi-stage classification enabled")
        logger.info(f"  Target selection rate: {analyzer_config.stage1_target_rate*100:.1f}%")
        
    def analyze_messages(self, months_back: Optional[int] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Main analysis method with months-based filtering.
        
        Args:
            months_back: Number of months to analyze (1, 2, 3, etc.) from current date
            limit: Optional message limit (mainly for testing)
        """
        start_time = time.time()
        
        try:
            # Step 0: Get database info and validate months parameter
            date_info = self.db_manager.get_date_range_info()
            if not date_info:
                logger.error("❌ No data available in database")
                return pd.DataFrame()
            
            # Validate months_back parameter
            if months_back is not None:
                max_months = date_info.get('max_months_back', 0)
                if months_back > max_months:
                    logger.warning(f"⚠️  Requested {months_back} months, but only {max_months} months available")
                    logger.info(f"📅 Available date range: {date_info['earliest_message']} to {date_info['latest_message']}")
            
            # Step 1: Fetch messages with months filtering
            messages_df = self.db_manager.fetch_messages(months_back=months_back, limit=limit)
            
            if messages_df.empty:
                logger.warning("❌ No messages found for the specified time period")
                return pd.DataFrame()
            
            total_messages = len(messages_df)
            
            # Log time period being analyzed
            if months_back:
                time_desc = f"last {months_back} month{'s' if months_back != 1 else ''}"
            elif limit:
                time_desc = f"latest {limit:,} messages"
            else:
                time_desc = "all available messages"
            
            logger.info(f"🎯 Processing {total_messages:,} messages from {time_desc}")
            
            # Step 2: Multi-stage preprocessing
            selected_messages, filtered_messages = self._multi_stage_preprocessing(messages_df)
            
            # Step 3: Parallel LLM analysis  
            llm_results = self._analyze_selected_messages_parallel(selected_messages)
            
            # Step 4: Combine and save results
            final_results = self._combine_all_results(selected_messages, filtered_messages, llm_results)
            results_df = pd.DataFrame(final_results)
            filename = self._save_results(results_df, months_back)
            
            # Step 5: Enhanced summary with time period info
            self._log_enhanced_summary(results_df, start_time, total_messages, filename, 
                                     time_desc, months_back)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            self.cache_manager.save_cache(force=True)
            raise
        finally:
            self.cache_manager.save_cache(force=True)
    
    def _multi_stage_preprocessing(self, messages_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
        """Enhanced multi-stage preprocessing with detailed tracking."""
        
        logger.info(f"🔍 Starting multi-stage preprocessing on {len(messages_df):,} messages")
        preprocessing_start = time.time()
        
        selected_messages = []
        filtered_messages = []
        
        # Process messages in chunks for memory efficiency
        chunk_size = 10000
        total_chunks = (len(messages_df) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(messages_df))
            chunk = messages_df.iloc[start_idx:end_idx]
            
            logger.info(f"📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk):,} messages)")
            
            for _, row in chunk.iterrows():
                level, signals, reasoning = self.classifier.classify_message(row['input'])
                
                message_data = {
                    'row': row,
                    'classification_level': level,
                    'signals': signals,
                    'reasoning': reasoning,
                    'preprocessing_confidence': signals.confidence,
                    'preprocessing_score': signals.feedback_score - signals.instruction_score
                }
                
                if self.classifier.should_send_to_llm(level):
                    message_data['text'] = row['input']
                    message_data['preprocessing_level'] = level.value
                    selected_messages.append(message_data)
                else:
                    filtered_messages.append(message_data)
            
            # Log progress every few chunks
            if (chunk_idx + 1) % 5 == 0 or chunk_idx + 1 == total_chunks:
                elapsed = time.time() - preprocessing_start
                rate = (end_idx) / elapsed
                logger.info(f"⏱️  Processed {end_idx:,}/{len(messages_df):,} messages ({rate:.0f}/sec)")
        
        # Get final classification stats
        classifier_stats = self.classifier.get_stats()
        preprocessing_time = time.time() - preprocessing_start
        
        logger.info(f"✅ Multi-stage preprocessing complete in {preprocessing_time:.2f}s:")
        logger.info(f"  📊 Classification breakdown:")
        logger.info(f"    Definitely feedback: {classifier_stats['definitely_feedback']:,}")
        logger.info(f"    Probably feedback: {classifier_stats['probably_feedback']:,}")  
        logger.info(f"    Uncertain: {classifier_stats['uncertain']:,}")
        logger.info(f"    Filtered out: {classifier_stats['filtered_out']:,}")
        logger.info(f"  🎯 Selection results:")
        logger.info(f"    Selected for LLM: {len(selected_messages):,} ({len(selected_messages)/len(messages_df)*100:.1f}%)")
        logger.info(f"    Filtered out: {len(filtered_messages):,} ({len(filtered_messages)/len(messages_df)*100:.1f}%)")
        logger.info(f"  ⚡ Performance: {len(messages_df)/preprocessing_time:.0f} messages/second")
        
        return selected_messages, filtered_messages
    
    def _analyze_selected_messages_parallel(self, selected_messages: List[dict]) -> List[SystemFeedbackConfirmation]:
        """Analyze selected messages with parallel processing and caching."""
        
        if not selected_messages:
            return []
        
        logger.info(f"🤖 Starting LLM analysis on {len(selected_messages):,} selected messages")
        
        # Check cache first
        uncached_messages = []
        cached_results = {}
        
        for i, msg in enumerate(selected_messages):
            cached_result = self.cache_manager.get(msg['text'])
            if cached_result:
                cached_results[i] = cached_result
            else:
                uncached_messages.append((i, msg))
        
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"💾 Cache check: {len(cached_results):,} cached, {len(uncached_messages):,} need processing")
        logger.info(f"📊 Cache stats: {cache_stats['cache_size']:,} entries, {cache_stats['hit_rate']:.1%} hit rate")
        
        # Initialize results array
        all_results = [None] * len(selected_messages)
        
        # Fill cached results
        for i, result in cached_results.items():
            all_results[i] = result
        
        # Process uncached messages in batches
        if uncached_messages:
            batches = []
            current_batch = []
            
            for i, msg in uncached_messages:
                current_batch.append({
                    'original_index': i,
                    'text': msg['text'],
                    'preprocessing_level': msg['preprocessing_level'],
                    'preprocessing_confidence': msg['preprocessing_confidence']
                })
                
                if len(current_batch) >= self.config.batch_size:
                    batches.append(current_batch)
                    current_batch = []
            
            if current_batch:
                batches.append(current_batch)
            
            logger.info(f"🔄 LLM processing: {len(batches):,} batches for {len(uncached_messages):,} messages")
            
            # Process batches in parallel
            with OptimizedBatchProcessor(self.llm_client, self.config.parallel_batches) as processor:
                batch_results = processor.process_batches_parallel(batches, self.cache_manager)
            
            # Merge batch results back
            for batch_idx, batch_result in enumerate(batch_results):
                batch = batches[batch_idx]
                for msg_idx, result in enumerate(batch_result):
                    if msg_idx < len(batch):
                        original_index = batch[msg_idx]['original_index']
                        all_results[original_index] = result
            
            final_cache_stats = self.cache_manager.get_stats()
            logger.info(f"✅ LLM analysis complete | Final cache: {final_cache_stats['cache_size']:,} entries")
        
        else:
            logger.info(f"✅ All messages cached - no LLM processing needed!")
        
        # Ensure no None results
        final_results = []
        for i, result in enumerate(all_results):
            if result is None:
                logger.warning(f"Missing result for index {i}")
                result = SystemFeedbackConfirmation(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Missing LLM result",
                    feedback_type="error"
                )
            final_results.append(result)
        
        return final_results
    
    def _combine_all_results(self, selected_messages: List[dict], filtered_messages: List[dict], 
                           llm_results: List[SystemFeedbackConfirmation]) -> List[dict]:
        """Combine all results with enhanced metadata."""
        
        final_results = []
        
        # Add LLM-analyzed results
        for i, msg in enumerate(selected_messages):
            if i < len(llm_results):
                result = llm_results[i]
            else:
                result = SystemFeedbackConfirmation(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Missing LLM result",
                    feedback_type="error"
                )
            
            final_results.append(self._create_enhanced_result_dict(msg, result, 'llm_analyzed'))
        
        # Add filtered results
        for msg in filtered_messages:
            level = msg['classification_level']
            
            # Determine final classification for filtered messages
            if level == ClassificationLevel.DEFINITELY_INSTRUCTION:
                has_feedback = False
                feedback_type = "user_instruction"
            elif level == ClassificationLevel.PROBABLY_INSTRUCTION:
                has_feedback = False
                feedback_type = "user_instruction"
            else:
                has_feedback = False
                feedback_type = "filtered_out"
            
            result = SystemFeedbackConfirmation(
                has_system_feedback=has_feedback,
                confidence_score=msg['preprocessing_confidence'],
                reasoning=f"Multi-stage filtering: {msg['reasoning']}",
                feedback_type=feedback_type
            )
            
            final_results.append(self._create_enhanced_result_dict(msg, result, 'filtered_out'))
        
        return final_results
    
    def _create_enhanced_result_dict(self, msg_data: dict, result: SystemFeedbackConfirmation, 
                                   processing_type: str) -> dict:
        """Create enhanced result dictionary with all metadata."""
        
        signals = msg_data.get('signals')
        
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
            'processing_type': processing_type,
            'classification_level': msg_data.get('classification_level', ClassificationLevel.UNCERTAIN_LEAN_INSTRUCTION).value,
            'preprocessing_confidence': msg_data.get('preprocessing_confidence', 0.0),
            'preprocessing_score': msg_data.get('preprocessing_score', 0.0),
            'preprocessing_reasoning': msg_data.get('reasoning', ''),
            'word_count': signals.word_count if signals else 0,
            'has_question': signals.has_question if signals else False,
            'has_past_ref': signals.has_past_ref if signals else False,
            'has_imperative': signals.has_imperative if signals else False,
            'has_evaluation': signals.has_evaluation if signals else False,
            'feedback_score': signals.feedback_score if signals else 0.0,
            'instruction_score': signals.instruction_score if signals else 0.0
        }
    
    def _save_results(self, results_df: pd.DataFrame, months_back: Optional[int] = None) -> str:
        """Save results with descriptive filename based on months."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create descriptive filename based on months
        if months_back:
            time_suffix = f"{months_back}months"
        else:
            time_suffix = "all_time"
        
        filename = f"enhanced_system_feedback_analysis_{time_suffix}_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"💾 Results saved to {filename}")
        return filename
    
    def _log_enhanced_summary(self, results_df: pd.DataFrame, start_time: float, 
                            total_messages: int, filename: str, time_desc: str,
                            months_back: Optional[int] = None):
        """Comprehensive analysis summary with time period information."""
        
        processing_time = time.time() - start_time
        
        # Core metrics
        system_feedback_count = len(results_df[results_df['has_system_feedback'] == True])
        llm_analyzed_count = len(results_df[results_df['processing_type'] == 'llm_analyzed'])
        filtered_count = len(results_df[results_df['processing_type'] == 'filtered_out'])
        
        # Efficiency metrics  
        selection_rate = llm_analyzed_count / total_messages
        efficiency_gain = filtered_count / total_messages
        feedback_yield = system_feedback_count / llm_analyzed_count if llm_analyzed_count > 0 else 0
        
        # Performance metrics
        messages_per_second = total_messages / processing_time
        estimated_cost = llm_analyzed_count * 0.01  # Assuming $0.01 per LLM call
        estimated_savings = filtered_count * 0.01
        
        # Time-based metrics
        if not results_df.empty and 'timestamp' in results_df.columns:
            # Calculate daily/monthly averages
            results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
            date_range = (results_df['timestamp'].max() - results_df['timestamp'].min()).days
            
            if date_range > 0:
                daily_avg_messages = total_messages / date_range
                daily_avg_feedback = system_feedback_count / date_range
            else:
                daily_avg_messages = total_messages
                daily_avg_feedback = system_feedback_count
        else:
            daily_avg_messages = 0
            daily_avg_feedback = 0
        
        # Quality metrics by classification level
        definitely_feedback = len(results_df[results_df['classification_level'] == 'definitely_feedback'])
        probably_feedback = len(results_df[results_df['classification_level'] == 'probably_feedback'])
        uncertain_feedback = len(results_df[results_df['classification_level'].str.contains('uncertain_lean_feedback', na=False)])
        
        # Cache performance
        cache_stats = self.cache_manager.get_stats()
        
        logger.info("=" * 100)
        logger.info("🚀 ENHANCED MULTI-STAGE SYSTEM FEEDBACK ANALYSIS - COMPREHENSIVE SUMMARY")
        logger.info("=" * 100)
        
        # Time Period Analysis
        logger.info("📅 TIME PERIOD ANALYSIS:")
        logger.info(f"  Analysis period: {time_desc}")
        if months_back:
            logger.info(f"  Months analyzed: {months_back}")
        logger.info(f"  Total messages: {total_messages:,}")
        if daily_avg_messages > 0:
            logger.info(f"  Daily average messages: {daily_avg_messages:.1f}")
            logger.info(f"  Daily average feedback: {daily_avg_feedback:.1f}")
        
        # Performance Overview
        logger.info("📊 PERFORMANCE OVERVIEW:")
        logger.info(f"  Total processing time: {processing_time:.2f} seconds")
        logger.info(f"  Processing rate: {messages_per_second:.1f} messages/second")
        
        # Multi-Stage Classification Results
        logger.info("🎯 MULTI-STAGE CLASSIFICATION RESULTS:")
        logger.info(f"  Messages sent to LLM: {llm_analyzed_count:,} ({selection_rate*100:.1f}%)")
        logger.info(f"  Messages filtered out: {filtered_count:,} ({efficiency_gain*100:.1f}%)")
        logger.info(f"  System feedback found: {system_feedback_count:,}")
        logger.info(f"  Feedback yield from LLM: {feedback_yield*100:.1f}%")
        logger.info(f"  Overall feedback rate: {system_feedback_count/total_messages*100:.2f}%")
        
        # Classification Level Breakdown
        logger.info("📋 CLASSIFICATION LEVEL BREAKDOWN:")
        logger.info(f"  Definitely feedback: {definitely_feedback:,} ({definitely_feedback/total_messages*100:.1f}%)")
        logger.info(f"  Probably feedback: {probably_feedback:,} ({probably_feedback/total_messages*100:.1f}%)")
        logger.info(f"  Uncertain (lean feedback): {uncertain_feedback:,} ({uncertain_feedback/total_messages*100:.1f}%)")
        
        # Time Period Efficiency Analysis
        if months_back:
            messages_per_month = total_messages / months_back
            feedback_per_month = system_feedback_count / months_back
            logger.info("📈 MONTHLY AVERAGES:")
            logger.info(f"  Messages per month: {messages_per_month:.0f}")
            logger.info(f"  Feedback per month: {feedback_per_month:.0f}")
            logger.info(f"  LLM calls per month: {llm_analyzed_count / months_back:.0f}")
        
        # Efficiency & Cost Analysis
        logger.info("💰 EFFICIENCY & COST ANALYSIS:")
        logger.info(f"  Preprocessing efficiency: {efficiency_gain*100:.1f}% filtered before LLM")
        logger.info(f"  Estimated LLM cost: ${estimated_cost:.2f}")
        logger.info(f"  Estimated savings: ${estimated_savings:.2f}")
        logger.info(f"  Cost efficiency: {estimated_savings/(estimated_cost + estimated_savings)*100:.1f}% saved")
        
        # Cache Performance
        logger.info("💾 CACHE PERFORMANCE:")
        logger.info(f"  Cache size: {cache_stats['cache_size']:,} entries")
        logger.info(f"  Cache hit rate: {cache_stats['hit_rate']*100:.1f}%")
        
        # System Configuration
        logger.info("⚙️  SYSTEM CONFIGURATION:")
        logger.info(f"  Model: {self.llm_client.config.model}")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Parallel workers: {self.config.parallel_batches}")
        
        # Top Examples
        logger.info("📝 EXAMPLE RESULTS:")
        
        # High confidence feedback examples
        high_conf_feedback = results_df[
            (results_df['has_system_feedback'] == True) & 
            (results_df['confidence_score'] >= 0.8)
        ].head(2)
        
        if not high_conf_feedback.empty:
            logger.info("  🎯 High-confidence feedback examples:")
            for i, (_, row) in enumerate(high_conf_feedback.iterrows(), 1):
                logger.info(f"    {i}. Confidence: {row['confidence_score']:.2f} | Level: {row['classification_level']}")
                logger.info(f"       Text: \"{row['input_text'][:80]}...\"")
                logger.info(f"       Reasoning: {row['reasoning'][:100]}...")
        
        logger.info("📁 OUTPUT:")
        logger.info(f"  Results saved to: {filename}")
        logger.info(f"  Total rows: {len(results_df):,}")
        logger.info(f"  Time period: {time_desc}")
        
        # SUCCESS METRICS with time-based context
        original_selection_rate = 0.022  # 11k/500k = 2.2
        original_feedback_found = 1300
        
        improvement_factor = selection_rate / original_selection_rate
        potential_feedback_increase = (system_feedback_count / original_feedback_found) if original_feedback_found > 0 else 0
        
        logger.info("🏆 IMPROVEMENT OVER ORIGINAL SYSTEM:")
        logger.info(f"  Selection rate change: {original_selection_rate*100:.1f}% → {selection_rate*100:.1f}% ({improvement_factor:.1f}x)")
        logger.info(f"  Feedback found change: {original_feedback_found:,} → {system_feedback_count:,} ({potential_feedback_increase:.1f}x)")
        logger.info(f"  Estimated recall improvement: {(potential_feedback_increase - 1) * 100:.0f}% more feedback captured")
        
        logger.info("=" * 100)
        logger.info("🎉 TIME-BASED ANALYSIS COMPLETE - PRODUCTION READY!")
        logger.info("=" * 100)

# ============================================================================
# INITIALIZATION AND FACTORY FUNCTIONS
# ============================================================================

def create_enhanced_analyzer(db_config: DatabaseConfig, analyzer_config: EnhancedAnalyzerConfig) -> EnhancedSystemFeedbackAnalyzer:
    """Factory function to create enhanced analyzer."""
    return EnhancedSystemFeedbackAnalyzer(db_config, analyzer_config)

def initialize_openai_client(api_key: str, model: str = "gpt-4o", max_tokens: int = 4000, temperature: float = 0.1) -> GenericLLMClient:
    """Initialize OpenAI client with optimized settings."""
    return create_openai_client(api_key, model, max_tokens, temperature)

def initialize_custom_client(api_key: str, base_url: str, model: str = "gpt-4o", max_tokens: int = 4000, temperature: float = 0.1) -> GenericLLMClient:
    """Initialize custom client."""
    return create_custom_client(api_key, base_url, model, max_tokens, temperature)

def initialize_client_from_env() -> GenericLLMClient:
    """Initialize from environment."""
    return create_client_from_env()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main(months_back: Optional[int] = None):
    """
    Main entry point with simple months-based filtering.
    
    Args:
        months_back: Number of months to analyze from today (1, 2, 3, etc.)
                    If None, analyzes all available data
    
    Examples:
        main(1)   # Last month
        main(3)   # Last 3 months
        main(6)   # Last 6 months  
        main()    # All data
    """
    try:
        # Database configuration
        db_config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="your_database",
            user="your_username", 
            password="your_password"
        )
        
        # Initialize LLM client
        llm_client = initialize_openai_client(
            api_key="your-openai-api-key",
            model="gpt-4o",
            temperature=0.1
        )
        
        # Test connection
        if not llm_client.test_connection():
            raise Exception("LLM connection test failed")
        
        # Enhanced analyzer configuration
        analyzer_config = EnhancedAnalyzerConfig(
            llm_client=llm_client,
            batch_size=15,
            parallel_batches=8,
            cache_enabled=True,
            cache_file="enhanced_feedback_cache.pkl",
            cache_save_frequency=25,
            max_retries=3,
            retry_delay=1.0,
            stage1_target_rate=0.15,
            stage2_target_rate=0.40,
            uncertain_threshold=0.3
        )
        
        # Log analysis parameters
        if months_back:
            logger.info(f"🕐 Starting analysis for last {months_back} month{'s' if months_back != 1 else ''}")
        else:
            logger.info(f"🕐 Starting analysis for all available data")
        
        logger.info(f"🚀 Enhanced multi-stage classification enabled")
        logger.info(f"🎯 Target: Select ~{analyzer_config.stage1_target_rate*100:.1f}% of messages for LLM analysis")
        
        # Initialize and run enhanced analyzer
        analyzer = create_enhanced_analyzer(db_config, analyzer_config)
        
        # Get database info first
        date_info = analyzer.db_manager.get_date_range_info()
        if date_info:
            logger.info(f"📊 Database contains {date_info['total_messages']:,} messages")
            logger.info(f"📅 Date range: {date_info['earliest_message']} to {date_info['latest_message']}")
            logger.info(f"📈 {date_info['months_available']} months of data available")
        
        # Run analysis with months-based filtering
        results = analyzer.analyze_messages(months_back=months_back)
        
        # Display final results summary
        if not results.empty:
            total_messages = len(results)
            system_feedback_count = len(results[results['has_system_feedback'] == True])
            llm_analyzed = len(results[results['processing_type'] == 'llm_analyzed'])
            filtered_out = len(results[results['processing_type'] == 'filtered_out'])
            
            # Key performance metrics
            selection_rate = llm_analyzed / total_messages
            feedback_yield = system_feedback_count / llm_analyzed if llm_analyzed > 0 else 0
            efficiency_gain = filtered_out / total_messages
            
            # Time-based insights
            results['timestamp'] = pd.to_datetime(results['timestamp'])
            date_range = (results['timestamp'].max() - results['timestamp'].min()).days
            
            print(f"\n🎉 ENHANCED ANALYSIS COMPLETE!")
            print(f"=" * 60)
            print(f"📊 FINAL RESULTS:")
            print(f"  Total messages processed: {total_messages:,}")
            if months_back:
                print(f"  Time period: Last {months_back} month{'s' if months_back != 1 else ''} ({date_range} days)")
            else:
                print(f"  Time period: All available data ({date_range} days)")
            print(f"  Messages sent to LLM: {llm_analyzed:,} ({selection_rate*100:.1f}%)")
            print(f"  System feedback found: {system_feedback_count:,}")
            print(f"  Feedback yield: {feedback_yield*100:.1f}%")
            print(f"  Processing efficiency: {efficiency_gain*100:.1f}% filtered")
            
            # Monthly breakdown if available
            if months_back and months_back > 1:
                monthly_feedback = results[results['has_system_feedback'] == True].groupby(
                    results['timestamp'].dt.to_period('M')
                ).size()
                
                print(f"\n📈 MONTHLY FEEDBACK BREAKDOWN:")
                for period, count in monthly_feedback.items():
                    print(f"  {period}: {count:,} feedback messages")
            
            # Improvement estimates
            original_selection = 11000  # Original system selected 11k
            original_feedback = 1300    # Original system found 1.3k feedback
            
            selection_improvement = llm_analyzed / original_selection
            feedback_improvement = system_feedback_count / original_feedback
            
            print(f"\n🏆 IMPROVEMENT ESTIMATES:")
            print(f"  Selection increase: {selection_improvement:.1f}x ({llm_analyzed:,} vs {original_selection:,})")
            print(f"  Feedback increase: {feedback_improvement:.1f}x ({system_feedback_count:,} vs {original_feedback:,})")
            print(f"  Estimated recall improvement: +{(feedback_improvement-1)*100:.0f}%")
            
            # Show classification breakdown
            level_counts = results['classification_level'].value_counts()
            print(f"\n📋 CLASSIFICATION BREAKDOWN:")
            for level, count in level_counts.head(5).items():
                print(f"  {level}: {count:,} ({count/total_messages*100:.1f}%)")
            
            # Example high-confidence feedback
            high_conf = results[
                (results['has_system_feedback'] == True) & 
                (results['confidence_score'] >= 0.8)
            ].head(3)
            
            if not high_conf.empty:
                print(f"\n💎 HIGH-CONFIDENCE FEEDBACK EXAMPLES:")
                for i, (_, row) in enumerate(high_conf.iterrows(), 1):
                    date_str = row['timestamp'].strftime('%Y-%m-%d')
                    print(f"  {i}. [{date_str}] Confidence: {row['confidence_score']:.2f}")
                    print(f"     Text: \"{row['input_text'][:100]}...\"")
                    print(f"     Level: {row['classification_level']}")
            
            print(f"=" * 60)
            print(f"🎯 ANALYSIS READY FOR PRODUCTION!")
            
        else:
            print("❌ No results generated. Check configuration.")
            
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        print(f"❌ Error: {e}")
        print("💡 Setup checklist:")
        print("  ✓ Database connection configured")
        print("  ✓ OpenAI API key set")
        print("  ✓ Required packages installed")
        print("  ✓ months_back parameter is valid integer")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Run the main analysis - just change the number for different months
    main(months_back=3)  # Analyze last 3 months