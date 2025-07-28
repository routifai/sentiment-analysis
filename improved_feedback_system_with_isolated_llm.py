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
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# Import the isolated LLM client
from llm_client import GenericLLMClient, LLMConfig, create_openai_client, create_custom_client, create_client_from_env
# Define models here to avoid pickle warnings
from pydantic import BaseModel

class SystemFeedbackConfirmation(BaseModel):
    """Model for system feedback confirmation results."""
    has_system_feedback: bool
    confidence_score: float
    reasoning: str
    feedback_type: str = "general"  # system, response, accuracy, content_filtered, error

class BatchSystemFeedbackConfirmation(BaseModel):
    """Model for batch system feedback confirmation results."""
    results: List[SystemFeedbackConfirmation]

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
    """Analyzer configuration with periodic cache saving."""
    llm_client: GenericLLMClient
    batch_size: int = 10
    parallel_batches: int = 5  # Number of batches to process in parallel
    cache_enabled: bool = True
    cache_file: str = "refined_system_feedback_cache.pkl"
    cache_save_frequency: int = 10  # Save cache after every N new entries
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class FeedbackSignals:
    """Container for different types of feedback signals detected."""
    temporal_signals: List[str]
    evaluation_signals: List[str]
    reference_signals: List[str]
    instruction_signals: List[str]
    context_signals: List[str]
    score: float
    confidence: float

# ============================================================================
# SCORE-AWARE PREPROCESSING - Based on real feedback patterns
# ============================================================================

class ScoreAwareSystemFeedbackPreprocessor:
    """
    Preprocessing based on the insight that real feedback is typically:
    - Short and concise (not long prompts)
    - Score range 2-10 (not too low, not too high)
    - Direct and to the point
    """
    
    def __init__(self):
        self.feedback_patterns = self._initialize_feedback_patterns()
        self.instruction_patterns = self._initialize_instruction_patterns()
        self.context_analyzers = self._initialize_context_analyzers()
    
    def _initialize_feedback_patterns(self) -> Dict[str, List[str]]:
        """Feedback patterns optimized for short, direct feedback."""
        return {
            # TEMPORAL FEEDBACK - focused on short expressions
            'temporal_indicators': [
                # Short past tense responses
                r'\b(response|answer|output|result|summary)\s+(was|were|seemed|appeared)\b',
                r'\b(you|it)\s+(did|didn\'t|worked|failed|helped)\b',
                r'\b(that|this)\s+(was|wasn\'t|seemed|didn\'t|helped|worked)\b',
                
                # Short past experience references
                r'\b(last time|before|earlier)\s.*(you|response)\b',
                r'\b(you\s+)?(gave|provided|said|told|suggested)\b',
                r'\b(got|received)\s.*(response|answer)\b',
                
                # Quick acknowledgments
                r'\b(just|now)\s+(you|that)\b',
                r'\b(when\s+you|after\s+you)\s+(said|told)\b'
            ],
            
            # EVALUATION LANGUAGE - short evaluative phrases
            'evaluation_indicators': [
                # Simple quality assessments
                r'\b(good|bad|great|poor|excellent|terrible|awful|perfect)\b',
                r'\b(right|wrong|correct|incorrect|accurate|inaccurate)\b',
                r'\b(helpful|unhelpful|useful|useless|clear|unclear)\b',
                r'\b(better|worse|improved|degraded)\b',
                
                # Short satisfaction expressions
                r'\b(thanks|thank\s+you|appreciate|disappointed|satisfied)\b',
                r'\b(like|love|hate|dislike|enjoy)\b',
                r'\b(makes\s+sense|doesn\'t\s+make\s+sense|confusing)\b',
                
                # Simple comparisons
                r'\b(more|less)\s+(helpful|clear|useful)\b',
                r'\b(not\s+quite|almost|close|exactly)\b'
            ],
            
            # REFERENCE TO SYSTEM OUTPUT - short references
            'reference_indicators': [
                # Simple references to AI output
                r'\b(your|the)\s+(response|answer|output|suggestion|advice)\b',
                r'\b(what\s+you\s+)?(said|told|mentioned|suggested)\b',
                r'\b(your\s+)?(suggestion|advice|idea|solution)\b',
                
                # Short behavior references
                r'\b(how\s+you|way\s+you)\s+(responded|answered|handled)\b',
                r'\b(according\s+to\s+you|like\s+you\s+said)\b'
            ],
            
            # SHORT CONTINUATION SIGNALS
            'continuation_indicators': [
                r'\b(also|but|however|though|still)\b',
                r'\b(and|plus|besides)\b',
                r'\b(by\s+the\s+way|anyway)\b'
            ]
        }
    
    def _initialize_instruction_patterns(self) -> Dict[str, List[str]]:
        """Instruction patterns that clearly indicate non-feedback."""
        return {
            # CLEAR REQUESTS/COMMANDS
            'request_patterns': [
                r'^\s*(can|could|would|will|please)\s+you\s+(help|assist|show|tell|explain|do|make|create|write)\b',
                r'^\s*(help\s+me|assist\s+me|show\s+me|tell\s+me|explain)\s+(with|how|what|why)\b',
                r'^\s*(i\s+need|i\s+want|i\s+would\s+like)\s+you\s+to\b',
                r'^\s*(generate|create|write|make|build|design|develop)\b'
            ],
            
            # CLEAR QUESTIONS
            'question_patterns': [
                r'^\s*(what|how|where|when|why|who|which)\s+.{15,}\?$',  # Substantial questions
                r'^\s*(is|are|was|were|will|would|can|could|should|do|does|did)\s+.{15,}\?$',
                r'^\s*(can\s+you\s+tell\s+me|do\s+you\s+know)\b.*\?'
            ],
            
            # TASK ASSIGNMENTS
            'task_patterns': [
                r'\b(your\s+)?(task|job|role|mission|assignment)\s+(is|will\s+be)\b',
                r'\b(you\s+)?(should|must|need\s+to|have\s+to)\s+(analyze|review|examine|check)\b',
                r'\b(act\s+as|pretend\s+to\s+be|imagine\s+you\s+are)\b'
            ]
        }
    
    def _initialize_context_analyzers(self) -> Dict[str, any]:
        """Initialize context analysis functions."""
        return {
            'pronoun_analysis': self._analyze_pronouns,
            'tense_analysis': self._analyze_tense,
            'sentiment_flow': self._analyze_sentiment_flow,
            'conversation_context': self._analyze_conversation_context
        }
    
    def _analyze_pronouns(self, text: str) -> Dict[str, any]:
        """Analyze pronoun usage patterns."""
        clean_text = text.lower()
        
        # Feedback pronouns (referring to system/output)
        feedback_pronouns = len(re.findall(r'\b(your|you|it|that|this)\s+(response|answer|output|was|did)\b', clean_text))
        
        # Instruction pronouns (directing system)
        instruction_pronouns = len(re.findall(r'\b(you\s+)?(can|should|could|would|will|need|must)\b', clean_text))
        
        return {
            'feedback_pronoun_count': feedback_pronouns,
            'instruction_pronoun_count': instruction_pronouns,
            'pronoun_ratio': feedback_pronouns / max(1, instruction_pronouns)
        }
    
    def _analyze_tense(self, text: str) -> Dict[str, any]:
        """Analyze verb tense patterns."""
        clean_text = text.lower()
        
        # Past tense (indicates feedback about completed action)
        past_tense = len(re.findall(r'\b(was|were|did|didn\'t|had|gave|provided|generated|created|seemed|appeared)\b', clean_text))
        
        # Future/imperative tense (indicates instruction)
        future_tense = len(re.findall(r'\b(will|would|can|could|should|please|help|show|tell|explain|do|make|create|generate)\b', clean_text))
        
        return {
            'past_tense_count': past_tense,
            'future_tense_count': future_tense,
            'tense_ratio': past_tense / max(1, future_tense)
        }
    
    def _analyze_sentiment_flow(self, text: str) -> Dict[str, any]:
        """Analyze sentiment and evaluative language."""
        clean_text = text.lower()
        
        # Evaluative language
        positive_eval = len(re.findall(r'\b(good|great|excellent|helpful|accurate|clear|useful|satisfied|impressed)\b', clean_text))
        negative_eval = len(re.findall(r'\b(bad|poor|wrong|unhelpful|unclear|useless|disappointed|confused)\b', clean_text))
        
        # Improvement suggestions (can be feedback or instruction)
        improvement = len(re.findall(r'\b(should|could|would|better|improve|enhance|fix|correct|add|include)\b', clean_text))
        
        return {
            'positive_evaluation': positive_eval,
            'negative_evaluation': negative_eval,
            'improvement_suggestions': improvement,
            'total_evaluation': positive_eval + negative_eval
        }
    
    def _analyze_conversation_context(self, text: str) -> Dict[str, any]:
        """Analyze conversational context clues."""
        clean_text = text.lower()
        
        # Conversation continuation (suggests ongoing dialogue)
        continuation = len(re.findall(r'\b(also|additionally|furthermore|moreover|however|but|although)\b', clean_text))
        
        # Meta-conversation (talking about the conversation itself)
        meta_conv = len(re.findall(r'\b(conversation|chat|discussion|interaction|dialogue|exchange)\b', clean_text))
        
        # Reference to previous exchange
        previous_ref = len(re.findall(r'\b(previous|earlier|before|last\s+time|above|prior)\b', clean_text))
        
        return {
            'continuation_markers': continuation,
            'meta_conversation': meta_conv,
            'previous_references': previous_ref
        }
    
    def _analyze_message_characteristics(self, text: str) -> Dict[str, any]:
        """Analyze key characteristics that distinguish feedback from instructions."""
        
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentence_count = len(re.split(r'[.!?]+', text.strip()))
        
        # Analyze sentence structure
        avg_sentence_length = word_count / max(1, sentence_count)
        has_question_mark = '?' in text
        has_multiple_sentences = sentence_count > 2
        
        # Analyze complexity
        complex_words = len([w for w in words if len(w) > 8])  # Long words
        complex_ratio = complex_words / max(1, word_count)
        
        # Analyze punctuation patterns
        exclamation_count = text.count('!')
        question_count = text.count('?')
        period_count = text.count('.')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_sentence_length': avg_sentence_length,
            'has_question_mark': has_question_mark,
            'has_multiple_sentences': has_multiple_sentences,
            'complex_ratio': complex_ratio,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'period_count': period_count
        }
    
    def analyze_feedback_signals(self, text: str) -> FeedbackSignals:
        """Comprehensive analysis of feedback signals in text."""
        if not text or pd.isna(text):
            return FeedbackSignals([], [], [], [], [], 0.0, 0.0)
        
        clean_text = text.lower().strip()
        
        # Detect different types of signals
        temporal_signals = []
        evaluation_signals = []
        reference_signals = []
        instruction_signals = []
        context_signals = []
        
        # Check temporal patterns
        for pattern in self.feedback_patterns['temporal_indicators']:
            matches = re.findall(pattern, clean_text)
            temporal_signals.extend([str(m) for m in matches])
        
        # Check evaluation patterns
        for pattern in self.feedback_patterns['evaluation_indicators']:
            matches = re.findall(pattern, clean_text)
            evaluation_signals.extend([str(m) for m in matches])
        
        # Check reference patterns
        for pattern in self.feedback_patterns['reference_indicators']:
            matches = re.findall(pattern, clean_text)
            reference_signals.extend([str(m) for m in matches])
        
        # Check instruction patterns (negative signals)
        for pattern_type in self.instruction_patterns.values():
            for pattern in pattern_type:
                matches = re.findall(pattern, clean_text)
                instruction_signals.extend([str(m) for m in matches])
        
        # Context analysis
        pronoun_analysis = self._analyze_pronouns(clean_text)
        tense_analysis = self._analyze_tense(clean_text)
        sentiment_analysis = self._analyze_sentiment_flow(clean_text)
        context_analysis = self._analyze_conversation_context(clean_text)
        
        # Calculate weighted score
        score = self._calculate_weighted_score(
            temporal_signals, evaluation_signals, reference_signals,
            instruction_signals, pronoun_analysis, tense_analysis,
            sentiment_analysis, context_analysis
        )
        
        # Calculate confidence based on signal strength and clarity
        confidence = self._calculate_confidence(
            temporal_signals, evaluation_signals, reference_signals,
            instruction_signals, text
        )
        
        context_signals = [
            f"pronouns: {pronoun_analysis}",
            f"tense: {tense_analysis}",
            f"sentiment: {sentiment_analysis}",
            f"context: {context_analysis}"
        ]
        
        return FeedbackSignals(
            temporal_signals=temporal_signals[:5],  # Limit for readability
            evaluation_signals=evaluation_signals[:5],
            reference_signals=reference_signals[:5],
            instruction_signals=instruction_signals[:5],
            context_signals=context_signals,
            score=score,
            confidence=confidence
        )
    
    def _calculate_weighted_score(self, temporal_signals, evaluation_signals, 
                                reference_signals, instruction_signals,
                                pronoun_analysis, tense_analysis, 
                                sentiment_analysis, context_analysis) -> float:
        """Calculate weighted feedback score based on multiple signals."""
        
        score = 0.0
        
        # Positive indicators (feedback signals) - OPTIMIZED FOR 2-10 RANGE
        score += len(temporal_signals) * 2.5        
        score += len(evaluation_signals) * 2.0      
        score += len(reference_signals) * 1.5       
        score += pronoun_analysis['feedback_pronoun_count'] * 1.0  
        score += tense_analysis['past_tense_count'] * 0.8  
        score += sentiment_analysis['total_evaluation'] * 1.5  
        score += context_analysis['previous_references'] * 1.0  
        
        # Negative indicators (instruction signals) - MORE PENALTY
        score -= len(instruction_signals) * 3.0     
        score -= pronoun_analysis['instruction_pronoun_count'] * 2.0  
        score -= tense_analysis['future_tense_count'] * 1.5  
        
        # Ratio bonuses
        if pronoun_analysis['pronoun_ratio'] > 2.0:
            score += 1.5
        if tense_analysis['tense_ratio'] > 2.0:
            score += 1.5
        
        return max(0.0, score)  # Ensure non-negative
    
    def _calculate_confidence(self, temporal_signals, evaluation_signals,
                            reference_signals, instruction_signals, text) -> float:
        """Calculate confidence in the classification."""
        
        # Base confidence from signal strength
        feedback_strength = len(temporal_signals) + len(evaluation_signals) + len(reference_signals)
        instruction_strength = len(instruction_signals)
        
        if feedback_strength == 0 and instruction_strength == 0:
            return 0.1  # Very low confidence for ambiguous text
        
        # High confidence requirements
        if feedback_strength >= 4 and instruction_strength == 0:
            return 0.95
        if instruction_strength >= 3 and feedback_strength == 0:
            return 0.9
        
        # Medium confidence
        if feedback_strength > instruction_strength:
            return 0.6 + min(0.2, (feedback_strength - instruction_strength) * 0.08)
        elif instruction_strength > feedback_strength:
            return 0.5 + min(0.2, (instruction_strength - feedback_strength) * 0.08)
        else:
            return 0.4
    
    def is_potential_system_feedback(self, text: str) -> bool:
        """Score-aware classification based on real feedback patterns."""
        
        if not text or pd.isna(text):
            return False
        
        # Get signal analysis
        signals = self.analyze_feedback_signals(text)
        characteristics = self._analyze_message_characteristics(text)
        
        # RULE 1: Length-based filtering (feedback is typically short)
        if characteristics['word_count'] > 80:  # Very long messages are usually instructions
            return False
        
        if characteristics['word_count'] > 50 and signals.score > 8:  # Medium-long with high score
            return False
        
        # RULE 2: Score-based filtering (KEY INSIGHT!)
        if signals.score > 15:  # Very high score = likely complex instruction
            return False
        
        if signals.score < 1.5:  # Very low score = no feedback signals
            return False
        
        # RULE 3: Question filtering
        if characteristics['has_question_mark'] and characteristics['word_count'] > 20:
            # Long questions are usually asking for help, not feedback
            return False
        
        # RULE 4: Complex instruction filtering
        if (len(signals.instruction_signals) >= 2 and 
            characteristics['word_count'] > 30 and
            len(signals.temporal_signals) == 0):
            return False
        
        # POSITIVE INDICATORS for feedback (optimized for score range 2-10)
        positive_score = 0
        
        # Score in the sweet spot (2-10)
        if 2.0 <= signals.score <= 10.0:
            positive_score += 3  # Strong positive signal
        elif 1.5 <= signals.score <= 12.0:
            positive_score += 2  # Medium positive signal
        elif signals.score > 12.0:
            positive_score -= 2  # Penalty for very high scores
        
        # Feedback signals
        if len(signals.temporal_signals) >= 1:
            positive_score += 2
        if len(signals.evaluation_signals) >= 1:
            positive_score += 2
        if len(signals.reference_signals) >= 1:
            positive_score += 1
        
        # Length bonuses (shorter is better for feedback)
        if characteristics['word_count'] <= 20:  # Very short
            positive_score += 2
        elif characteristics['word_count'] <= 40:  # Medium short
            positive_score += 1
        
        # Simple structure bonus
        if characteristics['sentence_count'] <= 2:  # Simple structure
            positive_score += 1
        
        # Confidence bonus
        if signals.confidence >= 0.7:
            positive_score += 1
        
        # NEGATIVE INDICATORS
        negative_score = 0
        
        # Instruction signals
        if len(signals.instruction_signals) >= 1:
            negative_score += 1
        if len(signals.instruction_signals) >= 2:
            negative_score += 2
        
        # Complexity penalties
        if characteristics['complex_ratio'] > 0.3:  # Many complex words
            negative_score += 1
        if characteristics['avg_sentence_length'] > 25:  # Very long sentences
            negative_score += 1
        
        # FINAL DECISION
        final_score = positive_score - negative_score
        
        # Multiple pathways to qualify:
        
        # 1. Strong evidence in the sweet spot
        if final_score >= 5 and 2.0 <= signals.score <= 10.0:
            return True
        
        # 2. Clear feedback signals with good characteristics
        if (len(signals.temporal_signals) >= 1 or len(signals.evaluation_signals) >= 1) and \
           characteristics['word_count'] <= 50 and negative_score <= 1:
            return True
        
        # 3. High confidence with reasonable score
        if signals.confidence >= 0.8 and 1.5 <= signals.score <= 12.0 and final_score >= 2:
            return True
        
        # 4. Short message with any feedback signals
        if characteristics['word_count'] <= 25 and \
           (len(signals.temporal_signals) >= 1 or len(signals.evaluation_signals) >= 1) and \
           negative_score == 0:
            return True
        
        return False
    
    def get_classification_reasoning(self, text: str) -> Dict[str, any]:
        """Enhanced reasoning with score and length awareness."""
        signals = self.analyze_feedback_signals(text)
        characteristics = self._analyze_message_characteristics(text)
        is_feedback = self.is_potential_system_feedback(text)
        
        # Build detailed reasoning
        reasoning_parts = []
        
        if is_feedback:
            reasoning_parts.append(f"Score in good range: {signals.score:.1f}")
            if characteristics['word_count'] <= 50:
                reasoning_parts.append(f"Appropriate length: {characteristics['word_count']} words")
            if len(signals.temporal_signals) >= 1:
                reasoning_parts.append(f"Temporal signals: {len(signals.temporal_signals)}")
            if len(signals.evaluation_signals) >= 1:
                reasoning_parts.append(f"Evaluation language: {len(signals.evaluation_signals)}")
            
            reason = f"Likely feedback: {', '.join(reasoning_parts)}"
        else:
            if signals.score > 15:
                reason = f"Score too high ({signals.score:.1f}) - likely complex instruction"
            elif characteristics['word_count'] > 80:
                reason = f"Too long ({characteristics['word_count']} words) for typical feedback"
            elif signals.score < 1.5:
                reason = f"Score too low ({signals.score:.1f}) - insufficient feedback signals"
            elif len(signals.instruction_signals) >= 2:
                reason = f"Multiple instruction patterns ({len(signals.instruction_signals)})"
            elif characteristics['has_question_mark'] and characteristics['word_count'] > 20:
                reason = f"Long question ({characteristics['word_count']} words) - likely seeking help"
            else:
                reason = f"Doesn't meet feedback criteria (score: {signals.score:.1f}, words: {characteristics['word_count']})"
        
        return {
            'classification': 'feedback' if is_feedback else 'not_feedback',
            'confidence': signals.confidence,
            'score': signals.score,
            'word_count': characteristics['word_count'],
            'characteristics': characteristics,
            'signals_detected': {
                'temporal': signals.temporal_signals,
                'evaluation': signals.evaluation_signals,
                'reference': signals.reference_signals,
                'instruction': signals.instruction_signals
            },
            'reasoning': reason
        }

# ============================================================================
# IMPROVED CACHE MANAGER WITH PERIODIC SAVING (unchanged)
# ============================================================================

class ImprovedCacheManager:
    """Enhanced cache manager with periodic saving and better error handling."""
    
    def __init__(self, cache_file: str, enabled: bool = True, save_frequency: int = 10):
        self.cache_file = cache_file
        self.enabled = enabled
        self.save_frequency = save_frequency  # Save after every N new entries
        self.cache: Dict[str, SystemFeedbackConfirmation] = self._load_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        self.unsaved_entries = 0  # Track entries that haven't been saved yet
        self.total_entries_processed = 0
        
    def _load_cache(self) -> Dict[str, SystemFeedbackConfirmation]:
        """Load cache from file with error recovery."""
        if not self.enabled or not os.path.exists(self.cache_file):
            logger.info(f"No existing cache file found at {self.cache_file}")
            return {}
        
        try:
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                converted_cache = {}
                
                for key, value in cached_data.items():
                    try:
                        if isinstance(value, dict):
                            if 'results' in value:
                                # Handle old batch format
                                if value['results'] and len(value['results']) > 0:
                                    first_result = value['results'][0]
                                    if isinstance(first_result, dict):
                                        converted_cache[key] = SystemFeedbackConfirmation(**first_result)
                                    else:
                                        converted_cache[key] = first_result
                            else:
                                converted_cache[key] = SystemFeedbackConfirmation(**value)
                        else:
                            converted_cache[key] = value
                    except Exception as e:
                        logger.warning(f"Could not convert cache entry for key {key}: {e}")
                        continue
                
                logger.info(f"💾 CACHE LOADED: {len(converted_cache)} entries from {self.cache_file}")
                return converted_cache
                
        except Exception as e:
            logger.warning(f"Could not load cache: {e}. Starting with empty cache.")
            return {}
    
    def save_cache(self, force: bool = False, reason: str = "manual"):
        """Save cache to file with periodic saving logic."""
        if not self.enabled:
            return
        
        # Only save if we have unsaved entries or if forced
        if not force and self.unsaved_entries == 0:
            return
        
        try:
            # Convert to dict for pickling
            cache_dict = {}
            for key, value in self.cache.items():
                try:
                    if hasattr(value, 'model_dump'):
                        cache_dict[key] = value.model_dump()
                    elif hasattr(value, 'dict'):
                        cache_dict[key] = value.dict()
                    else:
                        cache_dict[key] = value
                except Exception as e:
                    logger.warning(f"Could not serialize cache entry {key}: {e}")
                    continue
            
            # Atomic write
            temp_file = self.cache_file + '.tmp'
            with open(temp_file, 'wb') as f:
                pickle.dump(cache_dict, f)
            
            os.replace(temp_file, self.cache_file)
            
            logger.info(f"💾 CACHE SAVED ({reason}): {len(cache_dict)} total entries, {self.unsaved_entries} new entries saved to {self.cache_file}")
            self.unsaved_entries = 0  # Reset unsaved counter
            
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
        result = self.cache.get(key)
        
        if result:
            self.cache_hits += 1
            logger.debug(f"✅ Cache HIT for text: {text[:50]}...")
        else:
            self.cache_misses += 1
            logger.debug(f"🔄 Cache MISS for text: {text[:50]}...")
            
        return result
    
    def set(self, text: str, result: SystemFeedbackConfirmation):
        """Cache result with periodic saving."""
        if not self.enabled or not result:
            return
        
        try:
            # Validate result before caching
            if not isinstance(result, SystemFeedbackConfirmation):
                logger.warning(f"Attempted to cache invalid result type: {type(result)}")
                return
            
            key = self.get_cache_key(text)
            
            # Only increment if this is a new entry
            if key not in self.cache:
                self.unsaved_entries += 1
                self.total_entries_processed += 1
            
            self.cache[key] = result
            
            # Periodic save based on unsaved entries
            if self.unsaved_entries >= self.save_frequency:
                self.save_cache(reason=f"periodic save ({self.save_frequency} new entries)")
            
        except Exception as e:
            logger.warning(f"Could not cache result for text '{text[:50]}...': {e}")
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / max(1, total_requests)
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'unsaved_entries': self.unsaved_entries,
            'total_entries_processed': self.total_entries_processed,
            'save_frequency': self.save_frequency
        }

# ============================================================================
# DATABASE MANAGER (unchanged)
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
# PARALLEL BATCH PROCESSOR WITH IMPROVED CACHE INTEGRATION
# ============================================================================

class ParallelBatchProcessor:
    """Handles parallel processing of multiple batches with better cache integration."""
    
    def __init__(self, llm_client: GenericLLMClient, max_workers: int = 5, max_retries: int = 3):
        self.llm_client = llm_client
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(wait=True)
    
    def process_batch_with_retry(self, batch_data: List[dict], batch_id: int, 
                               cache_manager: ImprovedCacheManager) -> Tuple[int, List[SystemFeedbackConfirmation]]:
        """Process a single batch with retry logic and immediate caching."""
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"🔄 Processing batch {batch_id} (attempt {attempt + 1}, {len(batch_data)} messages)")
                
                # Create improved prompt
                prompt = self._create_improved_prompt(batch_data)
                
                # Call LLM
                response = self.llm_client.client.chat.completions.create(
                    model=self.llm_client.config.model,
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=self.llm_client.config.max_tokens,
                    temperature=self.llm_client.config.temperature,
                    timeout=60.0  # 60 second timeout
                )
                
                # Parse response
                content = response.choices[0].message.content
                if not content or content.strip() == "":
                    raise ValueError("Empty response from LLM")
                
                # Parse JSON
                try:
                    result_data = json.loads(content)
                except json.JSONDecodeError as e:
                    # Try to extract JSON from response if it's wrapped in other text
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                    else:
                        raise ValueError(f"Could not parse JSON from response: {e}")
                
                # Convert to objects
                if 'results' not in result_data:
                    raise ValueError("Response missing 'results' field")
                
                results = []
                for i, result in enumerate(result_data['results']):
                    try:
                        results.append(SystemFeedbackConfirmation(**result))
                    except Exception as e:
                        logger.warning(f"Could not parse result {i} in batch {batch_id}: {e}")
                        # Create fallback result
                        results.append(SystemFeedbackConfirmation(
                            has_system_feedback=False,
                            confidence_score=0.0,
                            reasoning=f"Parse error: {str(e)}",
                            feedback_type="error"
                        ))
                
                # Ensure we have the right number of results
                while len(results) < len(batch_data):
                    results.append(SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning="Missing result - padding with default",
                        feedback_type="error"
                    ))
                
                # IMMEDIATELY cache all results from this batch
                for i, result in enumerate(results[:len(batch_data)]):
                    if i < len(batch_data):
                        text = batch_data[i]['text']
                        cache_manager.set(text, result)
                        logger.debug(f"💾 Cached result for batch {batch_id}, message {i+1}")
                
                logger.info(f"✅ Successfully processed batch {batch_id} with {len(results)} results and cached immediately")
                return batch_id, results[:len(batch_data)]  # Ensure exact match
                
            except Exception as e:
                logger.warning(f"❌ Batch {batch_id} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"⏳ Retrying batch {batch_id} in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"💥 Batch {batch_id} failed after {self.max_retries + 1} attempts")
                    # Return fallback results
                    fallback_results = [
                        SystemFeedbackConfirmation(
                            has_system_feedback=False,
                            confidence_score=0.0,
                            reasoning=f"Batch processing failed: {str(e)}",
                            feedback_type="error"
                        ) for _ in batch_data
                    ]
                    return batch_id, fallback_results
    
    def _create_improved_prompt(self, batch_data: List[dict]) -> str:
        """Create improved system prompt for batch analysis."""
        
        prompt = """You are an expert at identifying FEEDBACK about AI system responses vs USER INSTRUCTIONS to AI systems.

CRITICAL DISTINCTION:
- SYSTEM FEEDBACK = User commenting on/evaluating a PREVIOUS AI response they already received
- USER INSTRUCTION = User asking the AI to DO something or perform a task

SYSTEM FEEDBACK Examples (classify as TRUE):
✅ "Your response was helpful"
✅ "That answer was wrong" 
✅ "The summary you provided was incomplete"
✅ "I'm satisfied with your output"
✅ "Your response seemed inaccurate"
✅ "The answer was too vague"
✅ "That response didn't address my question"

USER INSTRUCTIONS Examples (classify as FALSE):
❌ "Can you help me with this?"
❌ "Please verify this information"
❌ "You are a helpful assistant who..."
❌ "What is the capital of France?"
❌ "Summarize this document"
❌ "Your task is to analyze..."
❌ "I need you to check if this is correct"
❌ "Act as a financial advisor"

KEY INDICATORS for TRUE (System Feedback):
- Past tense verbs about AI responses ("was", "were", "did", "provided")
- Quality judgments ("helpful", "wrong", "accurate", "incomplete")
- References to previous outputs ("your response", "that answer", "the summary")
- Satisfaction expressions ("satisfied", "disappointed", "expected")

KEY INDICATORS for FALSE (User Instructions):
- Future/imperative verbs ("can you", "please", "help me", "do this")
- Questions to the AI ("what is", "how do", "where is")
- Task assignments ("your task", "you should", "I need you to")
- Role definitions ("you are", "act as", "pretend to be")
- Verification requests ("check this", "verify", "confirm")

For each message, determine:
1. has_system_feedback: true/false
2. confidence_score: 0.0-1.0 (how certain you are)
3. reasoning: Brief explanation of your decision
4. feedback_type: "system_evaluation", "user_instruction", "ambiguous", or "other"

Be VERY strict - when in doubt, classify as FALSE (user instruction).

Analyze these messages and return JSON with this exact structure:
{
  "results": [
    {
      "has_system_feedback": boolean,
      "confidence_score": float,
      "reasoning": "string explaining decision",
      "feedback_type": "system_evaluation|user_instruction|ambiguous|other"
    }
  ]
}

Messages to analyze:
"""
        
        for i, msg in enumerate(batch_data):
            prompt += f"\n{i+1}. \"{msg['text']}\""
        
        prompt += "\n\nReturn JSON array with one result per message in the same order."
        
        return prompt
    
    def process_batches_parallel(self, all_batches: List[List[dict]], 
                               cache_manager: ImprovedCacheManager) -> List[List[SystemFeedbackConfirmation]]:
        """Process multiple batches in parallel with immediate caching."""
        
        logger.info(f"🚀 Processing {len(all_batches)} batches in parallel (max {self.max_workers} workers)")
        
        # Submit all batches to executor
        future_to_batch = {}
        for batch_id, batch_data in enumerate(all_batches):
            future = self.executor.submit(self.process_batch_with_retry, batch_data, batch_id, cache_manager)
            future_to_batch[future] = batch_id
        
        # Collect results as they complete
        results = [None] * len(all_batches)
        completed = 0
        
        for future in concurrent.futures.as_completed(future_to_batch, timeout=300):
            try:
                batch_id, batch_results = future.result()
                results[batch_id] = batch_results
                completed += 1
                
                # Show cache stats every few batches
                if completed % 3 == 0 or completed == len(all_batches):
                    cache_stats = cache_manager.get_stats()
                    logger.info(f"📊 Progress: {completed}/{len(all_batches)} batches complete | "
                              f"Cache: {cache_stats['cache_size']} entries, "
                              f"{cache_stats['unsaved_entries']} unsaved, "
                              f"{cache_stats['hit_rate']:.1%} hit rate")
                
            except Exception as e:
                batch_id = future_to_batch[future]
                logger.error(f"💥 Batch {batch_id} failed completely: {e}")
                
                # Create fallback results for failed batch
                batch_size = len(all_batches[batch_id])
                fallback_results = [
                    SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning=f"Batch execution failed: {str(e)}",
                        feedback_type="error"
                    ) for _ in range(batch_size)
                ]
                results[batch_id] = fallback_results
                completed += 1
        
        # Final cache save to ensure everything is persisted
        cache_manager.save_cache(force=True, reason="batch processing complete")
        
        # Ensure no None results
        for i, result in enumerate(results):
            if result is None:
                logger.error(f"Batch {i} returned None, creating fallback")
                batch_size = len(all_batches[i])
                results[i] = [
                    SystemFeedbackConfirmation(
                        has_system_feedback=False,
                        confidence_score=0.0,
                        reasoning="Batch returned None - using fallback",
                        feedback_type="error"
                    ) for _ in range(batch_size)
                ]
        
        return results

# ============================================================================
# MAIN ENHANCED ANALYZER WITH SCORE-AWARE PREPROCESSING
# ============================================================================

class EnhancedSystemFeedbackAnalyzer:
    """Main system feedback analysis orchestrator with score-aware preprocessing."""
    
    def __init__(self, db_config: DatabaseConfig, analyzer_config: AnalyzerConfig):
        self.config = analyzer_config
        self.db_manager = DatabaseManager(db_config)
        
        # Use the new score-aware preprocessor
        self.preprocessor = ScoreAwareSystemFeedbackPreprocessor()
        
        # Initialize cache manager with periodic saving
        cache_save_frequency = getattr(analyzer_config, 'cache_save_frequency', 10)
        self.cache_manager = ImprovedCacheManager(
            analyzer_config.cache_file, 
            analyzer_config.cache_enabled,
            save_frequency=cache_save_frequency
        )
        
        self.llm_client = analyzer_config.llm_client
        
        logger.info(f"🚀 Initialized ENHANCED System Feedback Analyzer with SCORE-AWARE preprocessing:")
        logger.info(f"  Model: {analyzer_config.llm_client.config.model}")
        logger.info(f"  Base URL: {analyzer_config.llm_client.config.base_url or 'OpenAI Default'}")
        logger.info(f"  Batch size: {analyzer_config.batch_size}")
        logger.info(f"  Parallel batches: {analyzer_config.parallel_batches}")
        logger.info(f"  Cache enabled: {analyzer_config.cache_enabled}")
        logger.info(f"  Cache save frequency: every {cache_save_frequency} entries")
        logger.info(f"  🎯 SCORE-AWARE preprocessing optimized for 2-10 feedback score range")
        
        # Add cache status logging
        if analyzer_config.cache_enabled:
            cache_stats = self.cache_manager.get_stats()
            logger.info(f"  Cache status: {cache_stats['cache_size']} entries loaded")
            if cache_stats['cache_size'] > 0:
                logger.info(f"  Cache file: {analyzer_config.cache_file}")
        else:
            logger.info("  Cache disabled")
    
    def analyze_messages(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Main method to analyze messages for system feedback tone with score-aware preprocessing."""
        start_time = time.time()
        
        try:
            # Step 1: Fetch messages
            messages_df = self.db_manager.fetch_messages(limit)
            if messages_df.empty:
                logger.warning("No messages found")
                return pd.DataFrame()
            
            logger.info(f"📈 Processing {len(messages_df)} messages with SCORE-AWARE preprocessing and periodic cache saving")
            
            # Step 2: Score-aware preprocessing
            potential_system_feedback, non_system_feedback = self._score_aware_preprocess_messages(messages_df)
            
            # Step 3: Parallel LLM analysis with periodic caching
            system_feedback_results = self._analyze_potential_system_feedback_parallel(potential_system_feedback)
            
            # Step 4: Combine results
            final_results = self._combine_results(potential_system_feedback, non_system_feedback, system_feedback_results)
            
            # Step 5: Save results
            results_df = pd.DataFrame(final_results)
            filename = self._save_results(results_df)
            
            # Step 6: Final cache save and log summary
            self.cache_manager.save_cache(force=True, reason="analysis complete")
            self._log_comprehensive_summary(results_df, start_time, len(non_system_feedback), filename)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            # Always try to save cache even if analysis fails
            self.cache_manager.save_cache(force=True, reason="analysis failed - emergency save")
            raise
        finally:
            # Ensure cache is always saved
            self.cache_manager.save_cache(force=True, reason="final cleanup")
    
    def _score_aware_preprocess_messages(self, messages_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
        """Score-aware preprocessing with detailed signal analysis optimized for 2-10 range."""
        potential_system_feedback = []
        non_system_feedback = []
        
        preprocessing_stats = {
            'total_analyzed': 0,
            'feedback_candidates': 0,
            'non_feedback': 0,
            'high_confidence': 0,
            'medium_confidence': 0,
            'low_confidence': 0,
            'score_2_10_range': 0,  # Messages in the sweet spot
            'score_too_high': 0,    # Messages with score > 15
            'score_too_low': 0,     # Messages with score < 1.5
            'length_filtered': 0,   # Messages too long
            'question_filtered': 0  # Long questions filtered out
        }
        
        for _, row in messages_df.iterrows():
            preprocessing_stats['total_analyzed'] += 1
            
            # Get detailed analysis with score-aware classification
            classification = self.preprocessor.get_classification_reasoning(row['input'])
            
            # Track score distribution
            score = classification['score']
            if 2.0 <= score <= 10.0:
                preprocessing_stats['score_2_10_range'] += 1
            elif score > 15.0:
                preprocessing_stats['score_too_high'] += 1
            elif score < 1.5:
                preprocessing_stats['score_too_low'] += 1
            
            # Track length filtering
            word_count = classification['word_count']
            if word_count > 80:
                preprocessing_stats['length_filtered'] += 1
            
            # Track question filtering
            characteristics = classification.get('characteristics', {})
            if characteristics.get('has_question_mark') and word_count > 20:
                preprocessing_stats['question_filtered'] += 1
            
            message_data = {
                'row': row,
                'classification': classification['classification'],
                'confidence': classification['confidence'],
                'score': classification['score'],
                'word_count': classification['word_count'],
                'signals': classification['signals_detected'],
                'reasoning': classification['reasoning']
            }
            
            if classification['classification'] == 'feedback':
                potential_system_feedback.append(message_data)
                preprocessing_stats['feedback_candidates'] += 1
                
                # Track confidence levels
                if classification['confidence'] >= 0.9:
                    preprocessing_stats['high_confidence'] += 1
                elif classification['confidence'] >= 0.7:
                    preprocessing_stats['medium_confidence'] += 1
                else:
                    preprocessing_stats['low_confidence'] += 1
            else:
                non_system_feedback.append(message_data)
                preprocessing_stats['non_feedback'] += 1
        
        # Log detailed score-aware preprocessing results
        logger.info(f"🎯 SCORE-AWARE preprocessing complete:")
        logger.info(f"  Total messages: {preprocessing_stats['total_analyzed']}")
        logger.info(f"  Feedback candidates: {preprocessing_stats['feedback_candidates']}")
        logger.info(f"  Non-feedback: {preprocessing_stats['non_feedback']}")
        logger.info(f"  📊 Score distribution:")
        logger.info(f"    Sweet spot (2-10): {preprocessing_stats['score_2_10_range']} ({preprocessing_stats['score_2_10_range']/preprocessing_stats['total_analyzed']*100:.1f}%)")
        logger.info(f"    Too high (>15): {preprocessing_stats['score_too_high']} ({preprocessing_stats['score_too_high']/preprocessing_stats['total_analyzed']*100:.1f}%)")
        logger.info(f"    Too low (<1.5): {preprocessing_stats['score_too_low']} ({preprocessing_stats['score_too_low']/preprocessing_stats['total_analyzed']*100:.1f}%)")
        logger.info(f"  📏 Filtering effectiveness:")
        logger.info(f"    Length filtered (>80 words): {preprocessing_stats['length_filtered']}")
        logger.info(f"    Question filtered (long ?): {preprocessing_stats['question_filtered']}")
        logger.info(f"  🎯 Confidence distribution:")
        logger.info(f"    High confidence (≥0.9): {preprocessing_stats['high_confidence']}")
        logger.info(f"    Medium confidence (≥0.7): {preprocessing_stats['medium_confidence']}")
        logger.info(f"    Low confidence (<0.7): {preprocessing_stats['low_confidence']}")
        logger.info(f"  ⚡ Efficiency: {preprocessing_stats['non_feedback']/preprocessing_stats['total_analyzed']*100:.1f}% filtered before LLM")
        
        return potential_system_feedback, non_system_feedback
    
    def _analyze_potential_system_feedback_parallel(self, potential_system_feedback: List[dict]) -> List[SystemFeedbackConfirmation]:
        """Analyze potential system feedback messages with parallel LLM processing and immediate caching."""
        if not potential_system_feedback:
            return []
        
        logger.info(f"🔍 Analyzing {len(potential_system_feedback)} potential system feedback messages with parallel processing and periodic caching")
        
        # Check cache first and separate cached vs uncached
        uncached_messages = []
        cached_results = {}
        
        logger.info(f"💾 Checking cache for {len(potential_system_feedback)} messages...")
        
        for i, msg in enumerate(potential_system_feedback):
            cached_result = self.cache_manager.get(msg['row']['input'])
            if cached_result:
                cached_results[i] = cached_result
            else:
                uncached_messages.append((i, msg))
        
        cache_stats = self.cache_manager.get_stats()
        logger.info(f"📊 Cache check complete: {len(cached_results)} cached, {len(uncached_messages)} need processing")
        logger.info(f"📊 Current cache: {cache_stats['cache_size']} entries, {cache_stats['hit_rate']:.1%} hit rate")
        
        # Process uncached messages in parallel batches
        all_results = [None] * len(potential_system_feedback)
        
        # Fill in cached results
        for i, result in cached_results.items():
            all_results[i] = result
        
        if uncached_messages:
            # Create batches for parallel processing
            batches = []
            current_batch = []
            
            for i, msg in uncached_messages:
                current_batch.append({
                    'original_index': i,
                    'text': msg['row']['input'],
                    'preprocessing_confidence': msg['confidence'],
                    'preprocessing_score': msg['score'],
                    'word_count': msg['word_count']
                })
                
                if len(current_batch) >= self.config.batch_size:
                    batches.append(current_batch)
                    current_batch = []
            
            if current_batch:  # Don't forget the last batch
                batches.append(current_batch)
            
            logger.info(f"🔄 LLM PROCESSING: Created {len(batches)} batches for parallel processing ({len(uncached_messages)} total messages)")
            
            # Process batches in parallel with immediate caching
            with ParallelBatchProcessor(self.llm_client, self.config.parallel_batches) as processor:
                batch_results = processor.process_batches_parallel(batches, self.cache_manager)
            
            logger.info(f"✅ LLM PROCESSING: Completed {len(batches)} batches with periodic caching")
            
            # Merge results back
            for batch_idx, batch_result in enumerate(batch_results):
                batch = batches[batch_idx]
                
                for msg_idx, result in enumerate(batch_result):
                    if msg_idx < len(batch):
                        original_index = batch[msg_idx]['original_index']
                        all_results[original_index] = result
            
            # Final cache save to ensure everything is persisted
            final_stats = self.cache_manager.get_stats()
            logger.info(f"💾 Final cache status: {final_stats['cache_size']} total entries, "
                       f"{final_stats['total_entries_processed']} processed this session")
            
        else:
            logger.info(f"✅ ALL CACHED: No LLM processing needed - all messages found in cache")
        
        # Filter out any None results
        final_results = []
        for i, result in enumerate(all_results):
            if result is None:
                logger.warning(f"Missing result for index {i}, creating fallback")
                result = SystemFeedbackConfirmation(
                    has_system_feedback=False,
                    confidence_score=0.0,
                    reasoning="Missing result - created fallback",
                    feedback_type="error"
                )
            final_results.append(result)
        
        return final_results
    
    def _combine_results(self, potential_system_feedback: List[dict], 
                        non_system_feedback: List[dict], 
                        system_feedback_results: List[SystemFeedbackConfirmation]) -> List[dict]:
        """Combine all analysis results with enhanced metadata including score-aware info."""
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
            
            final_results.append(self._create_enhanced_result_dict(msg, result, 'potential_system_feedback'))
        
        # Add non-system feedback results
        for msg in non_system_feedback:
            result = SystemFeedbackConfirmation(
                has_system_feedback=False,
                confidence_score=msg['confidence'],
                reasoning=f"Score-aware preprocessing: {msg['reasoning']}",
                feedback_type="non_system"
            )
            final_results.append(self._create_enhanced_result_dict(msg, result, 'non_system_feedback'))
        
        return final_results
    
    def _create_enhanced_result_dict(self, msg_data: dict, 
                                   result: SystemFeedbackConfirmation, 
                                   preprocessing_result: str) -> dict:
        """Create enhanced result dictionary with score-aware preprocessing insights."""
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
            'preprocessing_result': preprocessing_result,
            'preprocessing_confidence': msg_data.get('confidence', 0.0),
            'preprocessing_score': msg_data.get('score', 0.0),
            'preprocessing_word_count': msg_data.get('word_count', 0),
            'preprocessing_reasoning': msg_data.get('reasoning', ''),
            'temporal_signals': ', '.join(msg_data.get('signals', {}).get('temporal', [])),
            'evaluation_signals': ', '.join(msg_data.get('signals', {}).get('evaluation', [])),
            'reference_signals': ', '.join(msg_data.get('signals', {}).get('reference', [])),
            'instruction_signals': ', '.join(msg_data.get('signals', {}).get('instruction', []))
        }
    
    def _save_results(self, results_df: pd.DataFrame) -> str:
        """Save results to CSV file with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"score_aware_system_feedback_analysis_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        return filename
    
    def _log_comprehensive_summary(self, results_df: pd.DataFrame, start_time: float, 
                                 non_system_feedback_count: int, filename: str):
        """Log comprehensive analysis summary with score-aware performance metrics."""
        processing_time = time.time() - start_time
        
        # Analysis results
        system_feedback_count = len(results_df[results_df['has_system_feedback'] == True])
        total_messages = len(results_df)
        
        # Count different types of results
        content_filtered_count = len(results_df[results_df['feedback_type'] == 'content_filtered'])
        error_count = len(results_df[results_df['feedback_type'] == 'error'])
        non_system_count = len(results_df[results_df['feedback_type'] == 'non_system'])
        system_evaluation_count = len(results_df[results_df['feedback_type'] == 'system_evaluation'])
        
        # Score-aware statistics
        score_2_10_feedback = len(results_df[(results_df['preprocessing_score'] >= 2.0) & 
                                           (results_df['preprocessing_score'] <= 10.0) & 
                                           (results_df['has_system_feedback'] == True)])
        avg_feedback_score = results_df[results_df['has_system_feedback'] == True]['preprocessing_score'].mean()
        avg_non_feedback_score = results_df[results_df['has_system_feedback'] == False]['preprocessing_score'].mean()
        
        # Cache performance
        cache_stats = self.cache_manager.get_stats()
        
        # LLM efficiency calculations
        llm_calls_made = total_messages - non_system_feedback_count
        llm_calls_saved = non_system_feedback_count
        efficiency_percent = (llm_calls_saved / total_messages) * 100
        
        # Accuracy calculations
        if llm_calls_made > 0:
            precision = system_feedback_count / llm_calls_made
            false_positive_rate = (llm_calls_made - system_feedback_count) / llm_calls_made
        else:
            precision = 0.0
            false_positive_rate = 0.0
        
        # Cost estimation (assuming $0.01 per LLM call)
        estimated_cost = llm_calls_made * 0.01
        estimated_savings = llm_calls_saved * 0.01
        
        logger.info("=" * 80)
        logger.info("🎯 SCORE-AWARE SYSTEM FEEDBACK ANALYSIS COMPREHENSIVE SUMMARY")
        logger.info("=" * 80)
        
        # Performance metrics
        logger.info("📊 PERFORMANCE METRICS:")
        logger.info(f"  Processing time: {processing_time:.2f} seconds")
        logger.info(f"  Messages per second: {total_messages / processing_time:.1f}")
        logger.info(f"  Total messages processed: {total_messages:,}")
        
        # Score-aware efficiency metrics
        logger.info("🎯 SCORE-AWARE EFFICIENCY METRICS:")
        logger.info(f"  Preprocessing efficiency: {efficiency_percent:.1f}% filtered")
        logger.info(f"  LLM calls made: {llm_calls_made:,}")
        logger.info(f"  LLM calls saved: {llm_calls_saved:,}")
        logger.info(f"  Estimated cost: ${estimated_cost:.2f}")
        logger.info(f"  Estimated savings: ${estimated_savings:.2f}")
        
        # Score distribution analysis
        logger.info("📈 SCORE DISTRIBUTION ANALYSIS:")
        logger.info(f"  Feedback in optimal range (2-10): {score_2_10_feedback:,}")
        logger.info(f"  Average feedback score: {avg_feedback_score:.2f}")
        logger.info(f"  Average non-feedback score: {avg_non_feedback_score:.2f}")
        
        # Accuracy metrics
        logger.info("🎯 ACCURACY METRICS:")
        logger.info(f"  System feedback detected: {system_feedback_count:,} ({system_feedback_count/total_messages*100:.1f}%)")
        logger.info(f"  Precision: {precision:.1%}")
        logger.info(f"  False positive rate: {false_positive_rate:.1%}")
        
        # Result breakdown
        logger.info("📋 RESULT BREAKDOWN:")
        logger.info(f"  System evaluation: {system_evaluation_count:,} ({system_evaluation_count/total_messages*100:.1f}%)")
        logger.info(f"  Non-system feedback: {non_system_count:,} ({non_system_count/total_messages*100:.1f}%)")
        logger.info(f"  Content filtered: {content_filtered_count:,} ({content_filtered_count/total_messages*100:.1f}%)")
        logger.info(f"  Processing errors: {error_count:,} ({error_count/total_messages*100:.1f}%)")
        
        # Cache performance with new metrics
        logger.info("💾 CACHE PERFORMANCE:")
        logger.info(f"  Cache size: {cache_stats['cache_size']:,} entries")
        logger.info(f"  Cache hit rate: {cache_stats['hit_rate']:.1%}")
        logger.info(f"  Cache hits: {cache_stats['cache_hits']:,}")
        logger.info(f"  Cache misses: {cache_stats['cache_misses']:,}")
        logger.info(f"  Session entries processed: {cache_stats['total_entries_processed']:,}")
        logger.info(f"  Save frequency: every {cache_stats['save_frequency']} entries")
        
        # Configuration used
        logger.info("⚙️  CONFIGURATION:")
        logger.info(f"  Batch size: {self.config.batch_size}")
        logger.info(f"  Parallel batches: {self.config.parallel_batches}")
        logger.info(f"  Model: {self.llm_client.config.model}")
        logger.info(f"  Max retries: {self.config.max_retries}")
        logger.info(f"  Cache save frequency: {self.cache_manager.save_frequency}")
        logger.info(f"  🎯 Score-aware preprocessing: 2-10 optimal range")
        
        logger.info("📁 OUTPUT:")
        logger.info(f"  Results file: {filename}")
        logger.info("=" * 80)

# ============================================================================
# INITIALIZATION FUNCTIONS (unchanged)
# ============================================================================

def initialize_openai_client(api_key: str, 
                           model: str = "gpt-4o",
                           max_tokens: int = 4000,
                           temperature: float = 0.2) -> GenericLLMClient:
    """Initialize OpenAI client."""
    return create_openai_client(api_key, model, max_tokens, temperature)

def initialize_custom_client(api_key: str,
                           base_url: str,
                           model: str = "gpt-4o",
                           max_tokens: int = 4000,
                           temperature: float = 0.2) -> GenericLLMClient:
    """Initialize custom OpenAI-compatible client."""
    return create_custom_client(api_key, base_url, model, max_tokens, temperature)

def initialize_client_from_env() -> GenericLLMClient:
    """Initialize client from environment variables."""
    return create_client_from_env()

# ============================================================================
# TESTING FUNCTION FOR SCORE-AWARE PREPROCESSING
# ============================================================================

def test_score_aware_preprocessing():
    """Test the score-aware preprocessing with various examples."""
    
    preprocessor = ScoreAwareSystemFeedbackPreprocessor()
    
    # Test cases with expected score ranges
    test_cases = [
        # EXPECTED FEEDBACK (score 2-10, short)
        ("Your response was helpful", "feedback"),
        ("That didn't work", "feedback"),
        ("Good answer", "feedback"),
        ("Not quite right", "feedback"),
        ("Thanks, that helped", "feedback"),
        ("Your suggestion was perfect", "feedback"),
        ("I like this approach", "feedback"),
        ("This makes sense now", "feedback"),
        ("Better than before", "feedback"),
        ("Still not working", "feedback"),
        
        # EXPECTED NON-FEEDBACK (various reasons)
        ("Please help me write a Python function to calculate fibonacci numbers", "instruction"),
        ("Can you explain how machine learning algorithms work in detail?", "instruction"),
        ("I need you to analyze this complex dataset and provide insights about customer behavior patterns", "instruction"),
        ("What are the best practices for software development in large teams?", "instruction"),
        ("Generate a comprehensive business plan for a startup company", "instruction"),
        ("Your task is to review this document and summarize the key findings", "instruction"),
        ("Act as a financial advisor and help me create an investment portfolio", "instruction"),
        ("How do I implement a neural network from scratch using Python?", "instruction"),
        ("Create a detailed marketing strategy for launching a new product", "instruction"),
        ("Explain the differences between various database management systems", "instruction")
    ]
    
    print("🧪 TESTING SCORE-AWARE PREPROCESSING:")
    print("=" * 80)
    
    feedback_correct = 0
    feedback_total = 0
    instruction_correct = 0
    instruction_total = 0
    
    for text, expected in test_cases:
        result = preprocessor.get_classification_reasoning(text)
        classification = result['classification']
        score = result['score']
        word_count = result['word_count']
        reasoning = result['reasoning']
        
        is_correct = classification == expected
        status = "✅" if is_correct else "❌"
        
        print(f"{status} {expected.upper()}: Score={score:.1f}, Words={word_count}")
        print(f"   Text: '{text}'")
        print(f"   Result: {classification}")
        print(f"   Reason: {reasoning}")
        print()
        
        if expected == "feedback":
            feedback_total += 1
            if is_correct:
                feedback_correct += 1
        else:
            instruction_total += 1
            if is_correct:
                instruction_correct += 1
    
    print("📊 RESULTS:")
    print(f"Feedback Detection: {feedback_correct}/{feedback_total} ({feedback_correct/feedback_total*100:.1f}%)")
    print(f"Instruction Filtering: {instruction_correct}/{instruction_total} ({instruction_correct/instruction_total*100:.1f}%)")
    print(f"Overall Accuracy: {(feedback_correct + instruction_correct)/(feedback_total + instruction_total)*100:.1f}%")
    
    # Score distribution analysis
    print("\n📈 SCORE DISTRIBUTION:")
    feedback_scores = []
    instruction_scores = []
    
    for text, expected in test_cases:
        result = preprocessor.get_classification_reasoning(text)
        if expected == "feedback":
            feedback_scores.append(result['score'])
        else:
            instruction_scores.append(result['score'])
    
    if feedback_scores:
        print(f"Feedback scores: {min(feedback_scores):.1f} - {max(feedback_scores):.1f} (avg: {sum(feedback_scores)/len(feedback_scores):.1f})")
    if instruction_scores:
        print(f"Instruction scores: {min(instruction_scores):.1f} - {max(instruction_scores):.1f} (avg: {sum(instruction_scores)/len(instruction_scores):.1f})")

# ============================================================================
# FACTORY FUNCTION FOR EASY INITIALIZATION
# ============================================================================

def create_score_aware_analyzer(db_config: DatabaseConfig, analyzer_config: AnalyzerConfig):
    """Create analyzer with score-aware preprocessing."""
    
    analyzer = EnhancedSystemFeedbackAnalyzer(db_config, analyzer_config)
    
    logger.info("🎯 Using SCORE-AWARE preprocessor (optimized for 2-10 score range)")
    
    return analyzer

# ============================================================================
# MAIN ENTRY POINT WITH SCORE-AWARE PREPROCESSING
# ============================================================================

def main():
    """Main entry point with score-aware preprocessing configuration."""
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
            model="gpt-4o"
        )
        
        # Test LLM connection
        if not llm_client.test_connection():
            raise Exception("LLM connection test failed")
        
        # Analyzer configuration with score-aware preprocessing
        analyzer_config = AnalyzerConfig(
            llm_client=llm_client,
            batch_size=10,                    # Messages per batch
            parallel_batches=5,               # Number of parallel batches
            cache_enabled=True,
            cache_file="score_aware_system_feedback_cache.pkl",
            cache_save_frequency=10,          # Save cache after every 10 new entries
            max_retries=3,
            retry_delay=1.0
        )
        
        logger.info(f"🎯 Starting SCORE-AWARE analysis with cache saving every {analyzer_config.cache_save_frequency} entries")
        
        # Initialize and run analyzer with score-aware preprocessing
        analyzer = create_score_aware_analyzer(db_config, analyzer_config)
        results = analyzer.analyze_messages(limit=100000)
        
        # Display results summary
        if not results.empty:
            system_feedback_count = len(results[results['has_system_feedback'] == True])
            total_messages = len(results)
            efficiency = len(results[results['preprocessing_result'] == 'non_system_feedback']) / total_messages
            
            # Score-aware statistics
            score_2_10_feedback = len(results[(results['preprocessing_score'] >= 2.0) & 
                                             (results['preprocessing_score'] <= 10.0) & 
                                             (results['has_system_feedback'] == True)])
            avg_feedback_score = results[results['has_system_feedback'] == True]['preprocessing_score'].mean()
            
            print(f"\n🎯 SCORE-AWARE ANALYSIS COMPLETE WITH PERIODIC CACHING!")
            print(f"📊 Total messages analyzed: {total_messages:,}")
            print(f"💬 Messages with system feedback: {system_feedback_count:,}")
            print(f"📈 System feedback rate: {system_feedback_count/total_messages*100:.2f}%")
            print(f"⚡ Preprocessing efficiency: {efficiency*100:.1f}% filtered")
            print(f"🎯 Feedback in optimal score range (2-10): {score_2_10_feedback:,}")
            print(f"📊 Average feedback score: {avg_feedback_score:.2f}")
            print(f"💾 Cache saved periodically every {analyzer_config.cache_save_frequency} entries")
            
            # Show example system feedback messages with scores
            system_feedback_examples = results[results['has_system_feedback'] == True].head(3)
            if not system_feedback_examples.empty:
                print(f"\n📝 Example system feedback messages:")
                for i, (_, row) in enumerate(system_feedback_examples.iterrows(), 1):
                    print(f"\n{i}. LLM Confidence: {row['confidence_score']:.2f} | Preprocessing Score: {row['preprocessing_score']:.1f} | Words: {row['preprocessing_word_count']}")
                    print(f"   Type: {row['feedback_type']}")
                    print(f"   Text: {row['input_text'][:100]}...")
                    print(f"   LLM Reason: {row['reasoning']}")
                    print(f"   Preprocessing: {row['preprocessing_reasoning']}")
        else:
            print("No results generated. Check your database connection and data.")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Error: {e}")
        print("💡 Make sure to:")
        print("  - Set up your database connection")
        print("  - Configure your OpenAI API key")
        print("  - Install required packages: pip install openai instructor psycopg2-binary pandas")

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    # Uncomment to test the score-aware preprocessing
    # test_score_aware_preprocessing()
    
    # Run the main analysis
    main()