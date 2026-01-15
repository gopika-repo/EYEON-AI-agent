import os
import cv2
import pytesseract
import numpy as np
import ssl
# This tells Python to ignore SSL certificate errors

from easyocr import Reader
ssl._create_default_https_context = ssl._create_unverified_context

# Force Windows to ignore the threading issues that cause SegFaults
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1" 

class TripleFusionOCR:
    def __init__(self):
        print("--- INITIALIZING STABLE OCR ENGINES ---")
        
        # 1. Initialize EasyOCR (Extremely stable on Windows)
        self.easy = Reader(['en'], gpu=False)
        
        # 2. Try to initialize Paddle, but catch the crash before it kills the program
        self.paddle = None
        try:
            from paddleocr import PaddleOCR
            # We use a very specific config that avoids the crashing libraries
            self.paddle = PaddleOCR(lang='en', use_angle_cls=False, show_log=False)
            print("✅ PaddleOCR Loaded")
        except Exception as e:
            print(f"⚠️ PaddleOCR skipped for stability: {e}")

    def extract_with_fusion(self, image_np):
        results = []

        # Engine 1: EasyOCR (Our Reliable Anchor)
        try:
            e_res = self.easy.readtext(image_np)
            if e_res:
                results.append(" ".join([res[1] for res in e_res]))
        except:
            pass

        # Engine 2: Tesseract (The Backup)
        try:
            t_txt = pytesseract.image_to_string(image_np).strip()
            if t_txt:
                results.append(t_txt)
        except:
            pass

        # Engine 3: Paddle (Only if it didn't crash during init)
        if self.paddle:
            try:
                p_res = self.paddle.ocr(image_np, cls=False)
                if p_res and p_res[0]:
                    results.append(" ".join([line[1][0] for line in p_res[0]]))
            except:
                pass

        if not results:
            return "No text detected"
            
        # Return the most complete result
        return max(results, key=len)