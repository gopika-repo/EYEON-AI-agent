from thefuzz import fuzz
import pytesseract
import pandas as pd
import numpy as np

class ValidatorEngine:
    
    def calculate_ocr_confidence(self, image):
        """
        1. Asks Tesseract for detailed data (not just string).
        2. Calculates the average confidence of all words.
        """
        try:
            # Get detailed data (includes coordinate, confidence, text)
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Filter out empty text/low confidence noise
            conf_scores = [int(conf) for conf in data['conf'] if int(conf) != -1]
            
            if not conf_scores:
                return 0
            
            # Average score (0-100)
            avg_conf = np.mean(conf_scores)
            return round(avg_conf, 2)
        except Exception as e:
            print(f"⚠️ OCR Scoring Error: {e}")
            return 0

    def validate_extraction(self, extracted_json: dict, raw_text: str):
        """
        Checks if extracted values actually exist in the raw text.
        Returns a 'Trust Score' for each field.
        """
        validation_report = {}
        raw_text_lower = raw_text.lower()
        
        for key, value in extracted_json.items():
            # Skip lists or empty values for now
            if isinstance(value, (list, dict)) or not value:
                continue
                
            val_str = str(value).lower()
            
            # 1. Exact Match Check
            if val_str in raw_text_lower:
                score = 100
                status = "✅ Verified"
            else:
                # 2. Fuzzy Match Check (Handle typos/formatting diffs)
                # 'partial_ratio' checks if the value is a substring of the text
                score = fuzz.partial_ratio(val_str, raw_text_lower)
                
                if score > 85:
                    status = "✅ Verified (Fuzzy)"
                elif score > 60:
                    status = "⚠️ Low Confidence"
                else:
                    status = "❌ Potential Hallucination"
            
            validation_report[key] = {
                "value": value,
                "confidence_score": score,
                "status": status
            }
            
        return validation_report