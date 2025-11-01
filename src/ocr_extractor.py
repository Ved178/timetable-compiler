import easyocr
import cv2
import pandas as pd
from config import OCR_LANGUAGES, OCR_GPU


class OCRExtractor:
    def __init__(self, languages=OCR_LANGUAGES, gpu=OCR_GPU):
        """Initialize EasyOCR reader"""
        self.reader = easyocr.Reader(languages, gpu=gpu)
    
    def extract_text_from_cell(self, cell_img):
        """Extract text from a single cell image"""
        if cell_img is None:
            return ""
        
        try:
            # Convert to grayscale for better OCR
            if len(cell_img.shape) == 3:
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = cell_img
            
            # Run OCR
            results = self.reader.readtext(gray, detail=0)
            text = " ".join(results).strip()
            return text
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def extract_text_from_cells(self, cells):
        """Extract text from all cells in the table"""
        table_text = []
        
        for row_idx, row in enumerate(cells):
            row_text = []
            for col_idx, (bbox, cell_img) in enumerate(row):
                text = self.extract_text_from_cell(cell_img)
                row_text.append(text)
            table_text.append(row_text)
        
        return table_text
    
    def create_dataframe(self, table_text):
        """Convert extracted text to DataFrame"""
        if not table_text:
            return pd.DataFrame()
        
        # Pad rows to equal length
        max_cols = max(len(r) for r in table_text)
        rows_norm = [r + [""] * (max_cols - len(r)) for r in table_text]
        
        df = pd.DataFrame(rows_norm)
        
        # Try to use first row as header if it looks like one
        if len(df) > 1 and (df.iloc[0].astype(bool).sum() >= max_cols // 2):
            df.columns = df.iloc[0]
            df = df.drop(0).reset_index(drop=True)
        
        return df
    
    def extract_full_table(self, cells):
        """Complete pipeline: extract text and create DataFrame"""
        table_text = self.extract_text_from_cells(cells)
        df = self.create_dataframe(table_text)
        return df, table_text
