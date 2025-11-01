"""
Timetable Compiler - Source Package

This package contains modules for:
- Image preprocessing and enhancement
- Table detection and structure extraction
- OCR text extraction
- LLM-powered error correction
- Timetable merging and export
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .image_processor import load_image, deskew_image, enhance_image_for_ocr
from .table_detector import detect_table_structure
from .ocr_extractor import OCRExtractor
from .llm_corrector import LLMCorrector
from .timetable_merger import TimetableMerger

__all__ = [
    'load_image',
    'deskew_image',
    'enhance_image_for_ocr',
    'detect_table_structure',
    'OCRExtractor',
    'LLMCorrector',
    'TimetableMerger',
]
