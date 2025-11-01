import os
from datetime import datetime
from pathlib import Path
from config import UPLOADS_DIR, OUTPUT_DIR, TIMESTAMP_FORMAT


def save_uploaded_file(uploaded_file):
    """Save uploaded file to uploads directory"""
    file_path = UPLOADS_DIR / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def generate_output_filename(prefix="timetable", format="xlsx"):
    """Generate timestamped output filename"""
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    filename = f"{prefix}_{timestamp}.{format}"
    return OUTPUT_DIR / filename


def clean_temp_files():
    """Clean up temporary files in uploads directory"""
    for file in UPLOADS_DIR.iterdir():
        if file.is_file():
            file.unlink()


def validate_api_key(provider, api_key):
    """Validate API key format"""
    if not api_key or not api_key.strip():
        return False
    
    if provider == 'openai':
        return api_key.startswith('sk-')
    elif provider == 'anthropic':
        return api_key.startswith('sk-ant-')
    elif provider == 'google':
        return len(api_key) > 20  # Google API keys are long alphanumeric strings
    
    return True


def format_dataframe_for_display(df, max_rows=50):
    """Format DataFrame for better display in Streamlit"""
    if df.empty:
        return df
    
    # Limit rows for display
    if len(df) > max_rows:
        return df.head(max_rows)
    
    return df


def create_readme_content():
    """Generate README.md content"""
    return """# Timetable Compiler

An intelligent timetable extraction and compilation tool that uses OCR and LLM to convert multiple timetable images into a single, accurate DataFrame.

## Features

- 📸 Upload multiple timetable images (JPG, PNG, PDF)
- 🔍 Advanced table detection and cell extraction
- 📝 EasyOCR for text extraction
- 🤖 LLM-powered correction (OpenAI, Anthropic, Google)
- 📊 Merge multiple timetables into one consolidated view
- 💾 Export to Excel or CSV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/timetable-compiler.git
cd timetable-compiler
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

### Steps:
1. Choose your LLM provider (OpenAI, Anthropic, or Google)
2. Enter your API key
3. Upload one or more timetable images
4. Click "Process Timetables"
5. Review and download the merged timetable

## Configuration

Edit `config.py` to customize:
- OCR settings (language, GPU usage)
- Table detection parameters
- LLM settings
- Output formats

Edit `prompts/system_prompt.txt` to:
- Add your specific class names
- Customize correction rules
- Adjust time formats

## Project Structure

```
timetable-compiler/
├── app.py                 # Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── src/
│   ├── image_processor.py    # Image preprocessing
│   ├── table_detector.py     # Table detection
│   ├── ocr_extractor.py      # OCR extraction
│   ├── llm_corrector.py      # LLM correction
│   ├── timetable_merger.py   # Timetable merging
│   └── utils.py              # Helper functions
└── prompts/
    └── system_prompt.txt     # LLM system prompt
```

## License

MIT License

## Contributing

Pull requests are welcome! Please feel free to submit issues or suggestions.
"""


def create_gitkeep_files():
    """Create .gitkeep files in empty directories"""
    dirs = [UPLOADS_DIR, OUTPUT_DIR]
    for dir_path in dirs:
        gitkeep = dir_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
