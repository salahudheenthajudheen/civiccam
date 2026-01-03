"""
CivicCam OCR Engine
License plate text extraction using EasyOCR
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import re


class LicensePlateOCR:
    """OCR engine for reading Indian license plates"""
    
    def __init__(self):
        """Initialize the OCR engine"""
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            print("[OCR] EasyOCR initialized with GPU support")
        except Exception as e:
            print(f"[OCR] Warning: GPU not available, using CPU: {e}")
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # Indian license plate pattern: XX00XX0000 or variations
        self.plate_pattern = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$')
    
    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Preprocess license plate image for better OCR
        
        Args:
            plate_img: Cropped license plate image (BGR)
            
        Returns:
            Preprocessed grayscale image
        """
        # Resize for better OCR
        height, width = plate_img.shape[:2]
        if width < 200:
            scale = 200 / width
            plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale, 
                                  interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def clean_plate_text(self, text: str) -> str:
        """
        Clean and format license plate text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned plate number
        """
        # Remove spaces and special characters
        text = re.sub(r'[^A-Za-z0-9]', '', text.upper())
        
        # Common OCR corrections for Indian plates
        corrections = {
            'O': '0',  # O to 0 in numeric positions
            'I': '1',  # I to 1
            'S': '5',  # S to 5
            'B': '8',  # B to 8 in numeric positions
            'Z': '2',  # Z to 2
        }
        
        # Apply corrections only to numeric positions (after state code)
        if len(text) >= 4:
            state_code = text[:2]
            rest = text[2:]
            
            # Convert likely numeric positions
            cleaned_rest = ""
            for i, char in enumerate(rest):
                if char in corrections and (i < 2 or i >= len(rest) - 4):
                    # Likely numeric position
                    if char in 'OISZB' and (i < 2 or i >= len(rest) - 4):
                        cleaned_rest += corrections.get(char, char)
                    else:
                        cleaned_rest += char
                else:
                    cleaned_rest += char
            
            text = state_code + cleaned_rest
        
        return text
    
    def read_plate(self, plate_img: np.ndarray) -> Tuple[str, float]:
        """
        Read text from a license plate image
        
        Args:
            plate_img: Cropped license plate image (BGR)
            
        Returns:
            Tuple of (plate_text, confidence)
        """
        if plate_img is None or plate_img.size == 0:
            return "", 0.0
        
        # Try multiple preprocessing approaches
        attempts = [
            plate_img,  # Original
            self.preprocess_plate(plate_img),  # Preprocessed
            cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY),  # Just grayscale
        ]
        
        best_text = ""
        best_conf = 0.0
        
        for img in attempts:
            try:
                results = self.reader.readtext(img)
                
                for (bbox, text, conf) in results:
                    cleaned = self.clean_plate_text(text)
                    
                    # Prefer longer valid-looking texts
                    if len(cleaned) >= 6 and conf > best_conf:
                        best_text = cleaned
                        best_conf = conf
                        
            except Exception as e:
                continue
        
        return best_text, best_conf
    
    def extract_plate_from_frame(self, frame: np.ndarray, 
                                  bbox: List[int]) -> Tuple[str, float, np.ndarray]:
        """
        Extract and read license plate from a detected bounding box
        
        Args:
            frame: Full frame (BGR)
            bbox: [x1, y1, x2, y2] bounding box
            
        Returns:
            Tuple of (plate_text, confidence, cropped_plate_image)
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding around the plate
        pad = 5
        h, w = frame.shape[:2]
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        # Crop the plate
        plate_img = frame[y1:y2, x1:x2]
        
        if plate_img.size == 0:
            return "", 0.0, None
        
        # Read the plate
        text, conf = self.read_plate(plate_img)
        
        return text, conf, plate_img
    
    def is_valid_indian_plate(self, text: str) -> bool:
        """
        Check if text matches Indian license plate format
        
        Args:
            text: Plate text to validate
            
        Returns:
            True if valid format
        """
        # Indian plates: AA00AA0000 or AA00A0000 or AA000000
        if len(text) < 9 or len(text) > 11:
            return False
        
        # First two characters must be state code (letters)
        if not text[:2].isalpha():
            return False
        
        # Last 4 characters must be numbers
        if not text[-4:].isdigit():
            return False
        
        return True


# Install easyocr if not present
def ensure_easyocr():
    try:
        import easyocr
        return True
    except ImportError:
        print("[OCR] EasyOCR not found. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'easyocr'])
        return True


if __name__ == "__main__":
    ensure_easyocr()
    
    # Test OCR
    ocr = LicensePlateOCR()
    
    # Test with sample plate images
    from pathlib import Path
    test_images = list(Path("datasets/license_plate/valid/images").glob("*.jpg"))[:3]
    
    for img_path in test_images:
        print(f"\nTesting: {img_path.name}")
        img = cv2.imread(str(img_path))
        text, conf = ocr.read_plate(img)
        print(f"  Plate: {text}, Confidence: {conf:.2f}")
