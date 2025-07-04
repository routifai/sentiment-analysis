import os
from dotenv import load_dotenv

load_dotenv()

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD')
}

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Table name (matching your existing structure)
FEEDBACK_TABLE = 'feedback'

# Output directory
OUTPUT_DIR = 'outputs'

# Analysis settings
TRENDING_THRESHOLD = 3  # If 3+ recent feedback about same issue = trending
RECENT_DAYS = 7  # Look at last 7 days for trending analysis
MONTHS_BACK = 3  # Only analyze feedback from the last X months (set to None for all time)