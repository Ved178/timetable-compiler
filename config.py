import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "output"
PROMPTS_DIR = BASE_DIR / "prompts"

# Create directories if they don't exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# OCR Settings
OCR_LANGUAGES = ['en']
OCR_GPU = False  # Set to True if GPU available

# Table Detection Settings
TABLE_DETECTION_SCALE = 15  # Lower values detect more lines
JOINT_TOLERANCE = 12  # Clustering tolerance in pixels
CELL_PADDING = 3  # Padding around cells for better OCR
MAX_SKEW_ANGLE = 15  # Maximum angle for deskewing

# LLM Settings
SUPPORTED_LLM_PROVIDERS = {
    'openai': 'OpenAI (GPT-4, GPT-3.5)',
    'anthropic': 'Anthropic (Claude)',
    'google': 'Google (Gemini)'
}

DEFAULT_LLM_PROVIDER = 'openai'
DEFAULT_MODEL = {
    'openai': 'gpt-4-turbo-preview',
    'anthropic': 'claude-3-sonnet-20240229',
    'google': 'gemini-pro'
}

# Temperature for LLM (lower = more deterministic)
LLM_TEMPERATURE = 0.2

# Output Settings
DEFAULT_OUTPUT_FORMAT = 'xlsx'
TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'

# Timetable Settings
DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
TIME_SLOTS = [
    '8:00-9:00', '9:00-10:00', '10:00-11:00', '11:00-12:00',
    '12:00-1:00', '1:00-2:00', '2:00-3:00', '3:00-4:00', '4:00-5:00'
]
