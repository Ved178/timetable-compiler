import cv2
import numpy as np
import math
from config import MAX_SKEW_ANGLE


def load_image(image_path):
    """Load image from file path or bytes"""
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        # Handle uploaded file bytes
        file_bytes = np.asarray(bytearray(image_path.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_path.seek(0)  # Reset file pointer
    
    if img is None:
        raise ValueError("Failed to load image")
    return img


def deskew_image(img, max_angle=MAX_SKEW_ANGLE):
    """Detect and correct skew in the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 
        threshold=100, 
        minLineLength=img.shape[1]//4, 
        maxLineGap=20
    )
    
    angles = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = math.degrees(math.atan2(y2-y1, x2-x1))
            # Ignore near-vertical lines
            if abs(angle) < 45:
                angles.append(angle)
    
    if len(angles) == 0:
        return img  # No lines found, return original
    
    median_angle = np.median(angles)
    
    # Don't rotate if almost straight
    if abs(median_angle) < 0.5:
        return img
    
    # Avoid extreme rotations
    if abs(median_angle) > max_angle:
        median_angle = 0
    
    # Rotate image
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def preprocess_image(img):
    """Apply preprocessing to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Adaptive thresholding for better text extraction
    thresh = cv2.adaptiveThreshold(
        blurred, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        11, 2
    )
    
    return thresh


def enhance_image_for_ocr(img):
    """Pipeline to enhance image quality for OCR"""
    # Deskew first
    deskewed = deskew_image(img)
    
    # Resize if too small (improves OCR accuracy)
    h, w = deskewed.shape[:2]
    if h < 1000 or w < 1000:
        scale = max(1000/h, 1000/w)
        new_h, new_w = int(h * scale), int(w * scale)
        deskewed = cv2.resize(deskewed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return deskewed
