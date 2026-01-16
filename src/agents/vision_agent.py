import os
from PIL import Image
import base64
from io import BytesIO
from groq import Groq
import json
from typing import Dict, Any
import time

class EnhancedVisionAgent:
    def __init__(self):
        """Initialize enhanced vision agent"""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        
        self.client = Groq(api_key=api_key, timeout=30.0)
        
        # Vision models available
        self.vision_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview"
        ]
        
        self.model = self.vision_models[0]  # Default
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if too large (for performance)
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = tuple(int(dim * ratio) for dim in img.size)
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
                
        except Exception as e:
            raise Exception(f"Image encoding failed: {str(e)}")
    
    def analyze_layout(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive document layout analysis with structured output
        """
        try:
            print(f"🔍 Analyzing document layout: {image_path}")
            
            # Encode image
            img_base64 = self.encode_image_to_base64(image_path)
            
            # Structured prompt for better analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Analyze this document image and provide a structured analysis. Focus on:

1. **Document Type**: What type of document is this? (Invoice, Resume, ID, Report, Letter, etc.)
2. **Layout Structure**: Describe the overall layout, sections, and organization
3. **Visual Elements**: 
   - Tables: Count and describe content/structure
   - Figures/Charts: Count and describe
   - Headers/Footers: Identify and describe
   - Text Blocks: Estimate number and arrangement
4. **Content Overview**: Summarize what the document contains
5. **Key Information**: Highlight important data points visible

Format your response as a structured analysis with clear sections."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]
            
            # Call Groq API
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9
            )
            
            processing_time = time.time() - start_time
            
            analysis_result = response.choices[0].message.content
            
            # Parse the response to extract structured information
            elements_found = self._extract_elements(analysis_result)
            confidence = self._calculate_confidence(analysis_result, len(elements_found))
            
            return {
                "description": analysis_result,
                "confidence": confidence,
                "elements_found": elements_found,
                "processing_time": f"{processing_time:.2f}s",
                "model_used": self.model,
                "image_dimensions": self._get_image_dimensions(image_path),
                "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            print(f"❌ Vision analysis error: {e}")
            return {
                "description": f"Vision analysis failed: {str(e)}",
                "confidence": 0.0,
                "elements_found": ["Error"],
                "error": str(e)
            }
    
    def _extract_elements(self, analysis_text: str) -> list:
        """Extract specific elements from analysis text"""
        elements = []
        text_lower = analysis_text.lower()
        
        # Check for elements
        if any(word in text_lower for word in ['table', 'grid', 'tabular']):
            elements.append("Table")
        
        if any(word in text_lower for word in ['figure', 'chart', 'graph', 'diagram', 'image', 'photo']):
            elements.append("Figure/Chart")
        
        if any(word in text_lower for word in ['header', 'title', 'heading']):
            elements.append("Header")
        
        if any(word in text_lower for word in ['footer', 'page number', 'footer text']):
            elements.append("Footer")
        
        if any(word in text_lower for word in ['signature', 'sign', 'stamp', 'seal']):
            elements.append("Signature/Stamp")
        
        if any(word in text_lower for word in ['logo', 'brand', 'company logo']):
            elements.append("Logo")
        
        if any(word in text_lower for word in ['barcode', 'qr code', 'code']):
            elements.append("Barcode/QR")
        
        # Add generic text blocks if no specific elements found
        if not elements and 'text' in text_lower:
            elements.append("Text Content")
        
        return elements if elements else ["General Document"]
    
    def _calculate_confidence(self, analysis_text: str, element_count: int) -> float:
        """Calculate confidence score based on analysis quality"""
        confidence = 0.5  # Base confidence
        
        # Increase based on analysis length
        if len(analysis_text) > 200:
            confidence += 0.2
        
        # Increase based on specific elements found
        if element_count > 0:
            confidence += min(element_count * 0.1, 0.3)
        
        # Check for structured response indicators
        if any(indicator in analysis_text for indicator in ['**', '1.', '2.', '- ', '* ']):
            confidence += 0.1
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _get_image_dimensions(self, image_path: str) -> dict:
        """Get image dimensions"""
        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format
                }
        except:
            return {"width": 0, "height": 0}
    
    def extract_text_from_image(self, image_path: str) -> Dict[str, Any]:
        """Enhanced OCR with structure"""
        try:
            img_base64 = self.encode_image_to_base64(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract all text from this image with the following guidelines:

1. Preserve the original formatting and layout as much as possible
2. Group related text together (paragraphs, sections)
3. Identify headings and their levels if possible
4. Extract tables as structured data if present
5. Preserve numerical formatting and units
6. Return the text in a clean, readable format

If the image contains multiple columns or complex layouts, try to maintain the reading order."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2048,
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            
            return {
                "text": extracted_text,
                "char_count": len(extracted_text),
                "word_count": len(extracted_text.split()),
                "extraction_method": "Vision AI",
                "confidence": 0.85
            }
            
        except Exception as e:
            print(f"Text extraction error: {e}")
            return {
                "text": "",
                "error": str(e),
                "confidence": 0.0
            }

# For backward compatibility
VisionAgent = EnhancedVisionAgent