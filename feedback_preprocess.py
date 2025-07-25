import psycopg2
import pandas as pd
import re
import json
import logging
from typing import List, Dict, Tuple, Optional
from src.config import DB_CONFIG, OPENAI_API_KEY
import openai
import instructor
from pydantic import BaseModel
from datetime import datetime
import time
import hashlib
import pickle
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackConfirmation(BaseModel):
    """Model for feedback confirmation results."""
    has_feedback_tone: bool
    confidence_score: float
    reasoning: str

class FeedbackAnalyzerOptimized:
    def __init__(self, api_key=None, model="gpt-4o", batch_size=100, cache_results=True):
        """Initialize the optimized feedback analyzer."""
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.cache_results = cache_results
        self.cache_file = "feedback_confirmation_cache.pkl"
        
        # Load existing cache
        self.cache = self._load_cache()
        
        # Comprehensive feedback keywords for preprocessing
        self.feedback_keywords = [
            # Direct feedback indicators
            'feedback', 'review', 'rating', 'score', 'evaluation', 'assessment',
            'opinion', 'thought', 'view', 'perspective', 'experience',
            'suggestion', 'recommendation', 'advice', 'tip', 'hint',
            'comment', 'remark', 'note', 'observation', 'finding',
            'report', 'analysis', 'study', 'research', 'investigation',
            'survey', 'poll', 'questionnaire', 'response', 'answer',
            'test', 'trial', 'experiment', 'pilot', 'beta',
            'version', 'update', 'upgrade', 'improvement', 'enhancement',
            'change', 'modification', 'adjustment', 'revision', 'edit',
            'compare', 'comparison', 'versus', 'vs', 'difference',
            'prefer', 'preference', 'choice', 'option', 'alternative',
            'expect', 'expectation', 'anticipate', 'predict', 'forecast',
            'hope', 'wish', 'want', 'need', 'require', 'demand',
            'should', 'could', 'would', 'might', 'may', 'can',
            'better', 'worse', 'same', 'different', 'similar',
            'more', 'less', 'most', 'least', 'best', 'worst',
            'always', 'never', 'sometimes', 'often', 'rarely',
            'usually', 'typically', 'generally', 'normally', 'commonly',
            
            # Sentiment indicators that suggest feedback
            'thank you', 'thanks', 'appreciate', 'grateful', 'pleased', 'happy', 'satisfied',
            'excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic', 'brilliant',
            'perfect', 'outstanding', 'superb', 'terrific', 'awesome', 'incredible',
            'love it', 'love this', 'loving', 'enjoy', 'enjoying', 'enjoyed',
            'helpful', 'useful', 'valuable', 'beneficial', 'effective', 'efficient',
            'clear', 'understandable', 'well explained', 'well done', 'good job',
            'impressive', 'impressed', 'surprised', 'exceeded expectations',
            'worked', 'working', 'solved', 'solution', 'resolved', 'fixed',
            'improved', 'better', 'best', 'optimal', 'optimized', 'enhanced',
            'saved time', 'time saver', 'efficient', 'quick', 'fast', 'speedy',
            'accurate', 'precise', 'correct', 'right', 'proper', 'appropriate',
            'professional', 'quality', 'high quality', 'top notch', 'premium',
            'user friendly', 'easy to use', 'intuitive', 'straightforward',
            'comprehensive', 'detailed', 'thorough', 'complete', 'full',
            'innovative', 'creative', 'original', 'unique', 'different',
            'reliable', 'dependable', 'trustworthy', 'consistent', 'stable',
            'cost effective', 'affordable', 'reasonable', 'worth it', 'value',
            'supportive', 'helpful', 'assistance', 'guidance', 'direction',
            'encouraging', 'motivating', 'inspiring', 'empowering', 'enabling',
            'flexible', 'adaptable', 'customizable', 'personalized', 'tailored',
            'modern', 'up to date', 'current', 'relevant', 'timely',
            'secure', 'safe', 'protected', 'confidential', 'private',
            'scalable', 'expandable', 'growable', 'future proof', 'sustainable',
            
            # Negative feedback indicators
            'not working', 'doesn\'t work', 'broken', 'failed', 'failure', 'error',
            'problem', 'issue', 'trouble', 'difficult', 'hard', 'challenging',
            'confusing', 'unclear', 'vague', 'ambiguous', 'unclear', 'unhelpful',
            'useless', 'worthless', 'pointless', 'meaningless', 'irrelevant',
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'dreadful',
            'disappointed', 'disappointing', 'let down', 'frustrated', 'frustrating',
            'annoyed', 'annoying', 'irritated', 'irritating', 'angry', 'mad',
            'upset', 'unhappy', 'sad', 'depressed', 'worried', 'concerned',
            'confused', 'lost', 'stuck', 'blocked', 'hindered', 'prevented',
            'slow', 'slower', 'sluggish', 'laggy', 'delayed', 'late',
            'expensive', 'costly', 'overpriced', 'waste of money', 'not worth it',
            'complicated', 'complex', 'overwhelming', 'too much', 'excessive',
            'missing', 'lacking', 'incomplete', 'partial', 'inadequate',
            'outdated', 'old', 'obsolete', 'deprecated', 'legacy',
            'buggy', 'glitchy', 'unstable', 'crash', 'freeze', 'hang',
            'insecure', 'unsafe', 'vulnerable', 'exposed', 'at risk',
            'incompatible', 'conflict', 'clash', 'mismatch', 'wrong',
            'inaccurate', 'incorrect', 'wrong', 'false', 'misleading',
            'biased', 'unfair', 'discriminatory', 'prejudiced', 'partial',
            'limited', 'restricted', 'constrained', 'narrow', 'small',
            'redundant', 'repetitive', 'duplicate', 'copy', 'same',
            'waste', 'wasted', 'unnecessary', 'needless', 'pointless',
            'boring', 'dull', 'monotonous', 'repetitive', 'tedious',
            'stressful', 'overwhelming', 'exhausting', 'tiring', 'draining',
            'time consuming', 'takes too long', 'slow process', 'delayed',
            'unreliable', 'inconsistent', 'unpredictable', 'variable',
            'difficult to use', 'hard to understand', 'complex interface',
            'poor quality', 'low quality', 'substandard', 'inferior',
            'not user friendly', 'unintuitive', 'confusing interface',
            'lack of features', 'missing functionality', 'limited options',
            'poor performance', 'slow response', 'lag', 'freeze',
            'technical issues', 'system problems', 'glitches', 'bugs',
            'customer service', 'support issues', 'unhelpful staff',
            'billing problems', 'payment issues', 'cost concerns',
            'privacy concerns', 'security issues', 'data protection',
            'accessibility issues', 'usability problems', 'design flaws'
        ]
        
        # Feedback patterns for more accurate preprocessing
        self.feedback_patterns = [
            r'\b(thank you|thanks|appreciate)\b',
            r'\b(feedback|review|rating|opinion)\b',
            r'\b(good|bad|great|terrible|excellent|poor)\b',
            r'\b(like|love|hate|dislike)\b',
            r'\b(helpful|useful|useless|unhelpful)\b',
            r'\b(work|working|broken|failed)\b',
            r'\b(easy|difficult|hard|simple|complex)\b',
            r'\b(fast|slow|quick|delayed)\b',
            r'\b(clear|confusing|unclear|vague)\b',
            r'\b(should|could|would|might|need|want)\b',
            r'\b(better|worse|best|worst|improve|improvement)\b',
            r'\b(expect|hope|wish|want|need)\b',
            r'\b(problem|issue|trouble|error)\b',
            r'\b(satisfied|happy|pleased|disappointed|frustrated)\b',
            r'\b(recommend|suggest|advice|tip)\b'
        ]
        
        logger.info(f"Initialized Optimized Feedback Analyzer with model: {model}, batch_size: {batch_size}")
    
    def _load_cache(self):
        """Load existing cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Could not save cache: {e}")
    
    def _get_cache_key(self, text):
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def connect_to_database(self):
        """Connect to the database and return connection."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            logger.info("Successfully connected to database")
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def fetch_messages_from_db(self, limit=None, offset=0):
        """
        Fetch messages from the database with pagination.
        
        Args:
            limit: Optional limit on number of records to fetch
            offset: Offset for pagination
            
        Returns:
            DataFrame with columns [emp_id, session_id, input, output, chat_type, timestamp]
        """
        conn = self.connect_to_database()
        try:
            query = """
                SELECT emp_id, session_id, input, output, chat_type, timestamp
                FROM chat_messages
                ORDER BY timestamp DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            if offset:
                query += f" OFFSET {offset}"
            
            df = pd.read_sql_query(query, conn)
            logger.info(f"Fetched {len(df)} messages from database (offset: {offset})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching messages: {str(e)}")
            raise
        finally:
            conn.close()
    
    def preprocess_text(self, text):
        """Preprocess text for keyword analysis."""
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        return text
    
    def extract_feedback_keywords(self, text):
        """Extract feedback-related keywords from text."""
        if not text:
            return []
        
        found_keywords = []
        
        # Check for feedback keywords
        for keyword in self.feedback_keywords:
            if keyword in text:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def check_feedback_patterns(self, text):
        """Check for feedback patterns using regex."""
        if not text:
            return 0
        
        pattern_matches = 0
        for pattern in self.feedback_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
        
        return pattern_matches
    
    def preprocess_for_feedback(self, messages_df):
        """
        Preprocess messages to identify potential feedback candidates.
        This is the key optimization - we only send messages to LLM that are likely to be feedback.
        """
        logger.info("Preprocessing messages for potential feedback...")
        
        potential_feedback = []
        non_feedback = []
        
        for idx, row in messages_df.iterrows():
            preprocessed_text = self.preprocess_text(row['input'])
            
            # Extract keywords
            keywords = self.extract_feedback_keywords(preprocessed_text)
            keyword_count = len(keywords)
            
            # Check patterns
            pattern_matches = self.check_feedback_patterns(row['input'])
            
            # Calculate feedback score
            feedback_score = keyword_count + (pattern_matches * 0.5)
            
            # Determine if this is likely feedback
            is_potential_feedback = (
                keyword_count >= 2 or  # At least 2 feedback keywords
                pattern_matches >= 3 or  # At least 3 pattern matches
                feedback_score >= 3 or  # High overall score
                any(keyword in preprocessed_text for keyword in ['feedback', 'review', 'rating', 'opinion', 'suggestion'])
            )
            
            if is_potential_feedback:
                potential_feedback.append({
                    'row_data': row,
                    'keywords': keywords,
                    'keyword_count': keyword_count,
                    'pattern_matches': pattern_matches,
                    'feedback_score': feedback_score
                })
            else:
                non_feedback.append({
                    'row_data': row,
                    'keywords': keywords,
                    'keyword_count': keyword_count,
                    'pattern_matches': pattern_matches,
                    'feedback_score': feedback_score
                })
        
        logger.info(f"Preprocessing complete:")
        logger.info(f"  Potential feedback messages: {len(potential_feedback)}")
        logger.info(f"  Non-feedback messages: {len(non_feedback)}")
        logger.info(f"  Total messages: {len(messages_df)}")
        logger.info(f"  Reduction in LLM calls: {len(non_feedback)} messages ({(len(non_feedback)/len(messages_df)*100):.1f}%)")
        
        return potential_feedback, non_feedback
    
    def confirm_feedback_with_llm(self, batch_messages):
        """
        Use LLM to confirm if preprocessed messages actually contain feedback tone.
        
        Args:
            batch_messages: List of message dictionaries that passed preprocessing
            
        Returns:
            List of confirmation results
        """
        if not batch_messages:
            return []
        
        try:
            # Prepare batch prompt for confirmation
            prompt = self._create_confirmation_prompt(batch_messages)
            
            # Use instructor for structured output
            client_with_instructor = instructor.from_openai(self.client)
            response = client_with_instructor.chat.completions.create(
                model=self.model,
                response_model=FeedbackConfirmation,
                messages=[
                    {"role": "system", "content": "You are confirming whether user inputs contain feedback tone. Focus only on whether the user is providing feedback, not on sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,  # Increased for batch processing
                temperature=0.2
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error confirming feedback with LLM: {str(e)}")
            # Return default confirmation
            return FeedbackConfirmation(
                has_feedback_tone=False,
                confidence_score=0.0,
                reasoning=f"Error during confirmation: {str(e)}"
            )
    
    def _create_confirmation_prompt(self, batch_messages):
        """Create a prompt for batch feedback confirmation."""
        prompt = "Confirm whether these user inputs contain feedback tone (not sentiment analysis).\n\n"
        
        for i, msg in enumerate(batch_messages, 1):
            text = msg['row_data']['input']
            keywords = msg['keywords']
            score = msg['feedback_score']
            
            prompt += f"INPUT {i}: \"{text}\"\n"
            prompt += f"Keywords found: {', '.join(keywords) if keywords else 'None'}\n"
            prompt += f"Feedback score: {score:.1f}\n\n"
        
        prompt += """
CONFIRMATION CRITERIA:
1. Does the user input express feedback, opinion, or evaluation about something?
2. Is the user providing their thoughts, suggestions, or reactions?
3. Is this more than just a factual question or request?

Focus ONLY on whether it's feedback, not on positive/negative sentiment.

Respond with:
- has_feedback_tone: True/False
- confidence_score: 0-1 scale
- reasoning: Brief explanation
"""
        
        return prompt
    
    def process_messages_optimized(self, limit=None):
        """
        Optimized method to process messages from database.
        Uses preprocessing to filter potential feedback, then LLM only for confirmation.
        
        Args:
            limit: Optional limit on number of messages to process
            
        Returns:
            DataFrame with analysis results
        """
        try:
            start_time = time.time()
            
            # Fetch messages from database
            messages_df = self.fetch_messages_from_db(limit)
            
            if messages_df.empty:
                logger.warning("No messages found in database")
                return pd.DataFrame()
            
            logger.info(f"Processing {len(messages_df)} messages with optimization...")
            
            # Step 1: Preprocess to identify potential feedback
            potential_feedback, non_feedback = self.preprocess_for_feedback(messages_df)
            
            # Step 2: Process potential feedback with LLM
            confirmed_feedback = []
            if potential_feedback:
                logger.info(f"Confirming {len(potential_feedback)} potential feedback messages with LLM...")
                
                # Process in batches
                for batch_idx in range(0, len(potential_feedback), self.batch_size):
                    batch = potential_feedback[batch_idx:batch_idx + self.batch_size]
                    batch_num = batch_idx // self.batch_size + 1
                    total_batches = (len(potential_feedback) + self.batch_size - 1) // self.batch_size
                    
                    logger.info(f"Processing feedback batch {batch_num}/{total_batches} ({len(batch)} messages)")
                    
                    # Check cache for each message in batch
                    batch_to_confirm = []
                    cached_results = []
                    
                    for msg in batch:
                        cache_key = self._get_cache_key(msg['row_data']['input'])
                        
                        if cache_key in self.cache and self.cache_results:
                            cached_results.append(self.cache[cache_key])
                        else:
                            batch_to_confirm.append(msg)
                    
                    # Confirm non-cached messages
                    if batch_to_confirm:
                        llm_results = self.confirm_feedback_with_llm(batch_to_confirm)
                        
                        # Cache results
                        for i, msg in enumerate(batch_to_confirm):
                            cache_key = self._get_cache_key(msg['row_data']['input'])
                            self.cache[cache_key] = llm_results[i]
                        
                        # Combine cached and new results
                        confirmed_feedback.extend(cached_results + llm_results)
                    else:
                        confirmed_feedback.extend(cached_results)
                    
                    # Save cache periodically
                    if batch_num % 10 == 0:
                        self._save_cache()
                    
                    # Rate limiting
                    if batch_num < total_batches:
                        time.sleep(0.5)  # 0.5 second pause between batches
            
            # Step 3: Create final results
            final_results = []
            
            # Add confirmed feedback results
            feedback_idx = 0
            for msg in potential_feedback:
                if feedback_idx < len(confirmed_feedback):
                    confirmation = confirmed_feedback[feedback_idx]
                    feedback_idx += 1
                else:
                    confirmation = FeedbackConfirmation(
                        has_feedback_tone=False,
                        confidence_score=0.0,
                        reasoning="Not confirmed"
                    )
                
                # Extract keywords for this message
                preprocessed_text = self.preprocess_text(msg['row_data']['input'])
                keywords = self.extract_feedback_keywords(preprocessed_text)
                
                result_row = {
                    'emp_id': msg['row_data']['emp_id'],
                    'session_id': msg['row_data']['session_id'],
                    'input_text': msg['row_data']['input'],
                    'chat_type': msg['row_data']['chat_type'],
                    'timestamp': msg['row_data']['timestamp'],
                    'has_feedback_tone': confirmation.has_feedback_tone,
                    'confidence_score': confirmation.confidence_score,
                    'reasoning': confirmation.reasoning,
                    'feedback_keywords': ', '.join(keywords) if keywords else '',
                    'keyword_count': msg['keyword_count'],
                    'pattern_matches': msg['pattern_matches'],
                    'feedback_score': msg['feedback_score'],
                    'preprocessing_result': 'potential_feedback'
                }
                
                final_results.append(result_row)
            
            # Add non-feedback results (no LLM analysis needed)
            for msg in non_feedback:
                preprocessed_text = self.preprocess_text(msg['row_data']['input'])
                keywords = self.extract_feedback_keywords(preprocessed_text)
                
                result_row = {
                    'emp_id': msg['row_data']['emp_id'],
                    'session_id': msg['row_data']['session_id'],
                    'input_text': msg['row_data']['input'],
                    'chat_type': msg['row_data']['chat_type'],
                    'timestamp': msg['row_data']['timestamp'],
                    'has_feedback_tone': False,  # Preprocessing determined no feedback
                    'confidence_score': 1.0,  # High confidence in preprocessing result
                    'reasoning': 'Preprocessing determined no feedback tone detected',
                    'feedback_keywords': ', '.join(keywords) if keywords else '',
                    'keyword_count': msg['keyword_count'],
                    'pattern_matches': msg['pattern_matches'],
                    'feedback_score': msg['feedback_score'],
                    'preprocessing_result': 'non_feedback'
                }
                
                final_results.append(result_row)
            
            # Create results DataFrame
            results_df = pd.DataFrame(final_results)
            
            # Save to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_analysis_optimized_{timestamp}.csv"
            results_df.to_csv(filename, index=False)
            
            processing_time = time.time() - start_time
            
            logger.info(f"Optimized analysis complete in {processing_time:.2f} seconds")
            logger.info(f"Results saved to {filename}")
            logger.info(f"Total messages processed: {len(results_df)}")
            logger.info(f"Messages with feedback tone: {len(results_df[results_df['has_feedback_tone'] == True])}")
            logger.info(f"LLM calls saved: {len(non_feedback)} ({(len(non_feedback)/len(messages_df)*100):.1f}% reduction)")
            logger.info(f"Cache hits: {len(self.cache)}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
            raise

def main():
    """Main function to run the optimized feedback analysis."""
    try:
        # Initialize analyzer with optimized settings
        analyzer = FeedbackAnalyzerOptimized(
            batch_size=100,  # Process 100 messages per LLM call
            cache_results=True,  # Enable caching
            model="gpt-4o"  # Use GPT-4 for better accuracy
        )
        
        # Process messages with optimization
        results = analyzer.process_messages_optimized(limit=1000)  # Process first 1000 messages for testing
        
        if not results.empty:
            print(f"\nOptimized Feedback Analysis Summary:")
            print(f"Total messages analyzed: {len(results)}")
            print(f"Messages with feedback tone: {len(results[results['has_feedback_tone'] == True])}")
            print(f"Preprocessing results:")
            print(f"  - Potential feedback: {len(results[results['preprocessing_result'] == 'potential_feedback'])}")
            print(f"  - Non-feedback: {len(results[results['preprocessing_result'] == 'non_feedback'])}")
            print(f"Average confidence score: {results['confidence_score'].mean():.2f}")
            
            # Show some examples
            feedback_examples = results[results['has_feedback_tone'] == True].head(3)
            if not feedback_examples.empty:
                print(f"\nExample feedback messages:")
                for _, row in feedback_examples.iterrows():
                    print(f"- Confidence: {row['confidence_score']:.2f}")
                    print(f"  Text: {row['input_text'][:100]}...")
                    print(f"  Reasoning: {row['reasoning']}")
                    print()
        
    except Exception as e:
        print(f"Error running optimized feedback analysis: {e}")
        print("Make sure your database is set up and OPENAI_API_KEY is configured")

if __name__ == "__main__":
    main() 