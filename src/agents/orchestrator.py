import os
import json
from typing import Dict, Any
import time
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from groq import Groq
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

class Orchestrator:
    def __init__(self):
        """Initialize orchestrator with error handling"""
        self.client = None
        self.vision_agent = None
        self.rag_engine = None
        
        # Initialize components with fallbacks
        try:
            from src.agents.rag_engine import RAGEngine
            self.rag_engine = RAGEngine()
            print("✅ RAG Engine initialized")
        except Exception as e:
            print(f"⚠️ RAG Engine not available: {e}")
        
        try:
            from src.agents.vision_agent import VisionAgent
            self.vision_agent = VisionAgent()
            print("✅ Vision Agent initialized")
        except Exception as e:
            print(f"⚠️ Vision Agent not available: {e}")
        
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            print("⚠️ GROQ_API_KEY not set")
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2 (no poppler required)"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    page_text = page.extract_text() or ""
                    text += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                
            return text
            
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""
    
    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using Tesseract"""
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""
    
    def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from file based on type"""
        try:
            if file_type in ['pdf']:
                return self._extract_text_from_pdf(file_path)
            elif file_type in ['png', 'jpg', 'jpeg']:
                return self._extract_text_from_image(file_path)
            elif file_type in ['txt']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            else:
                return ""
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""
    
    def _detect_language(self, text: str) -> tuple:
        """Detect language of text"""
        if not text or len(text) < 50:
            return "Unknown", text
        
        try:
            if self.client:
                prompt = f"""Detect the language of this text. Return only the language name in English.
                
                Text: {text[:500]}
                
                Language:"""
                
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    max_tokens=10
                )
                
                language = response.choices[0].message.content.strip()
                
                # Translate if not English
                if language.lower() not in ['english', 'en']:
                    translate_prompt = f"Translate this {language} text to English:\n\n{text[:3000]}"
                    translate_response = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": translate_prompt}],
                        model="llama-3.1-8b-instant",
                        temperature=0.1,
                        max_tokens=1024
                    )
                    translated = translate_response.choices[0].message.content
                    return language, translated
                
                return language, text
                
        except Exception as e:
            print(f"Language detection error: {e}")
        
        return "English", text
    
    def _classify_document(self, text: str, vision_analysis: Dict = None) -> str:
        """Classify document type"""
        categories = ["Invoice", "Resume", "ID Card", "Contract", "Report", "Letter", "General"]
        
        if not text:
            return "General"
        
        try:
            if self.client:
                context = f"Text: {text[:2000]}"
                if vision_analysis and vision_analysis.get('description'):
                    context += f"\nVision Analysis: {vision_analysis['description'][:500]}"
                
                prompt = f"""Classify this document into ONE of these categories: {', '.join(categories)}.
                
                {context}
                
                Return only the category name."""
                
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=50
                )
                
                category = response.choices[0].message.content.strip()
                return category if category in categories else "General"
                
        except Exception as e:
            print(f"Classification error: {e}")
        
        return "General"
    
    def _extract_entities(self, text: str, doc_type: str) -> Dict:
        """Extract entities based on document type"""
        templates = {
            "Invoice": ["Vendor", "Invoice Number", "Date", "Total Amount", "Items", "Due Date"],
            "Resume": ["Name", "Email", "Phone", "Skills", "Experience", "Education"],
            "ID Card": ["Name", "ID Number", "Date of Birth", "Expiry Date", "Address"],
            "Contract": ["Parties", "Effective Date", "Term", "Payment Terms", "Signatures"],
            "Report": ["Title", "Author", "Date", "Findings", "Recommendations"],
            "Letter": ["Sender", "Recipient", "Date", "Subject", "Content"],
            "General": ["Key People", "Organizations", "Dates", "Important Facts"]
        }
        
        template = templates.get(doc_type, templates["General"])
        
        try:
            if self.client:
                prompt = f"""Extract the following information from this document:
                
                Document Type: {doc_type}
                Fields to extract: {', '.join(template)}
                
                Document Text: {text[:3000]}
                
                Format as JSON with field names as keys. Use null if information not found.
                Example: {{"Name": "John Doe", "Email": "john@example.com"}}"""
                
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.1,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content
                
                # Try to parse JSON
                try:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != 0:
                        return json.loads(content[start:end])
                except:
                    pass
                
                # Return raw if JSON parsing fails
                return {"raw_response": content[:500]}
                
        except Exception as e:
            print(f"Entity extraction error: {e}")
        
        return {"error": "Extraction failed"}
    
    def _generate_summary(self, text: str, entities: Dict = None) -> str:
        """Generate document summary"""
        if not text:
            return "No content available for summary."
        
        try:
            if self.client:
                context = f"Document Content: {text[:4000]}"
                if entities:
                    context += f"\n\nExtracted Entities: {json.dumps(entities, indent=2)}"
                
                prompt = f"""Provide a comprehensive 3-4 sentence summary of this document.
                
                {context}
                
                Focus on:
                1. Main purpose and content
                2. Key information and entities
                3. Important findings or conclusions"""
                
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    max_tokens=300
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            print(f"Summary generation error: {e}")
        
        return "Summary generation failed."
    
    def process_document(self, file_path: str, metadata: Dict = None) -> Dict[str, Any]:
        """Main processing pipeline"""
        start_time = time.time()
        
        if not metadata:
            metadata = {}
        
        file_type = metadata.get('file_type', 'pdf')
        
        print(f"\n🚀 Starting document processing: {file_path}")
        print(f"📄 File type: {file_type}")
        
        # Step 1: Extract text
        print("1️⃣ Extracting text...")
        raw_text = self._extract_text(file_path, file_type)
        
        if not raw_text or len(raw_text.strip()) < 10:
            return {
                "status": "error",
                "message": "Could not extract meaningful text from document",
                "processing_time": f"{time.time() - start_time:.2f}s"
            }
        
        print(f"   ✓ Extracted {len(raw_text)} characters")
        
        # Step 2: Vision Analysis (if applicable and agent available)
        vision_analysis = {}
        if file_type in ['png', 'jpg', 'jpeg'] and self.vision_agent:
            print("2️⃣ Running vision analysis...")
            try:
                vision_analysis = self.vision_agent.analyze_layout(file_path)
                print(f"   ✓ Vision found: {len(vision_analysis.get('elements_found', []))} elements")
            except Exception as e:
                print(f"   ⚠️ Vision analysis failed: {e}")
                vision_analysis = {"error": str(e)}
        elif file_type == 'pdf':
            print("2️⃣ Skipping vision analysis for PDF (requires poppler)")
            vision_analysis = {"info": "Vision analysis requires poppler installation for PDFs"}
        else:
            print("2️⃣ No vision analysis for text files")
        
        # Step 3: Language detection and translation
        print("3️⃣ Detecting language...")
        language, translated_text = self._detect_language(raw_text)
        print(f"   ✓ Language: {language}")
        
        # Step 4: Document classification
        print("4️⃣ Classifying document...")
        doc_category = self._classify_document(translated_text, vision_analysis)
        print(f"   ✓ Category: {doc_category}")
        
        # Step 5: Entity extraction
        print("5️⃣ Extracting entities...")
        entities = self._extract_entities(translated_text, doc_category)
        if isinstance(entities, dict):
            entity_count = sum(1 for v in entities.values() if v and v != 'null')
            print(f"   ✓ Extracted {entity_count} entities")
        
        # Step 6: Generate summary
        print("6️⃣ Generating summary...")
        summary = self._generate_summary(translated_text, entities)
        print(f"   ✓ Summary generated ({len(summary)} chars)")
        
        # Step 7: Compile results
        processing_time = time.time() - start_time
        
        result = {
            "status": "success",
            "processing_time": f"{processing_time:.2f}s",
            "file_info": {
                "path": file_path,
                "type": file_type,
                "name": metadata.get('file_name', 'Unknown')
            },
            "content_analysis": {
                "raw_text_length": len(raw_text),
                "translated_text_length": len(translated_text),
                "detected_language": language,
                "word_count": len(translated_text.split()),
                "doc_category": doc_category
            },
            "vision_analysis": vision_analysis,
            "entities": entities,
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": metadata
        }
        
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        
        return result

# Global instance for Streamlit compatibility
_orchestrator = None

def get_orchestrator():
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator

def process_document(file_path: str, metadata: Dict = None):
    """Main function for Streamlit app"""
    orchestrator = get_orchestrator()
    return orchestrator.process_document(file_path, metadata)

if __name__ == "__main__":
    # Test the orchestrator
    test_file = "sample.pdf"
    if os.path.exists(test_file):
        result = process_document(test_file, {"file_type": "pdf"})
        print("\n" + "="*50)
        print("Processing Results:")
        print(json.dumps(result, indent=2))
    else:
        print(f"Test file {test_file} not found")
        print("Creating a test workflow with minimal data...")
        
        # Create a test file
        test_content = """Invoice
        
        Vendor: ABC Corporation
        Invoice Number: INV-2024-001
        Date: January 15, 2024
        Total Amount: $1,250.00
        
        Items:
        1. Software License - $1,000.00
        2. Support Services - $250.00
        
        Due Date: February 15, 2024
        """
        
        with open("test_invoice.txt", "w") as f:
            f.write(test_content)
        
        result = process_document("test_invoice.txt", {"file_type": "txt", "file_name": "test_invoice.txt"})
        print("\nTest Results:")
        print(f"Category: {result.get('doc_category')}")
        print(f"Entities: {result.get('entities')}")