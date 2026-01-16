# Add this at the top of orchestrator.py
import pytesseract

# Set Tesseract path for Windows
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    # Try alternative paths
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
    except:
        print("⚠️ Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")