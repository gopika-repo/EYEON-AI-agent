from PIL import Image, ImageDraw
import pytesseract
import re

class RedactorEngine:
    def redact_sensitive_data(self, image: Image.Image, sensitive_values: list = None):
        """
        Redacts provided values AND automatically detects PII (Emails/Phones) using Regex.
        """
        if sensitive_values is None: sensitive_values = []
        
        print(f"🛡️ Redactor: Scanning for PII...")
        
        # 1. Get OCR Data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        redacted_image = image.copy()
        draw = ImageDraw.Draw(redacted_image)
        
        # 2. Prepare Match List (Clean up input values)
        clean_sensitive = [re.sub(r'\W+', '', str(v)).lower() for v in sensitive_values if v]

        # 3. Define Universal Regex Patterns (The "Safety Net")
        # Matches "name@domain.com"
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        # Matches "+91 999..." or "999-999..." (Broad phone matcher)
        phone_pattern = re.compile(r'(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}')

        n_boxes = len(data['level'])
        
        # 4. Scan Every Word
        for i in range(n_boxes):
            word = data['text'][i]
            if not word.strip(): continue
            
            is_match = False
            clean_word = re.sub(r'\W+', '', word).lower()
            
            # CHECK A: Is it in the AI's list?
            if clean_word:
                for val in clean_sensitive:
                    if val and (clean_word in val or val in clean_word):
                        is_match = True
                        break
            
            # CHECK B: Is it a Phone/Email pattern? (Regex)
            if not is_match:
                if email_pattern.search(word) or phone_pattern.search(word):
                    is_match = True
                # Special: Catch isolated phone parts (like "+91")
                if word.startswith("+91") or (word.isdigit() and len(word) >= 10):
                    is_match = True
                # Special: Catch emails split by OCR
                if "@" in word and "." in word:
                    is_match = True

            # 5. Draw Black Box
            if is_match:
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                draw.rectangle([x, y, x + w, y + h], fill="black", outline="black")
                
        return redacted_image