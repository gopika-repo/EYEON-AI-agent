import os
import json
from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
from PIL import Image
from pdf2image import convert_from_path
from dotenv import load_dotenv
import pytesseract
from groq import Groq

# --- 1. INITIALIZE ENGINES ---
try:
    from src.agents.rag_engine import RAGEngine
    rag_engine = RAGEngine()
except ImportError: rag_engine = None

try:
    from src.agents.validator import ValidatorEngine
    validator = ValidatorEngine()
except ImportError: validator = None

try:
    from src.agents.layout_engine import LayoutEngine
    layout_engine = LayoutEngine()
except ImportError: layout_engine = None

try:
    from src.agents.redactor import RedactorEngine
    redactor = RedactorEngine()
except ImportError: redactor = None

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 2. SCHEMAS ---
SENSITIVE_KEYS = ["Phone", "Mobile", "Contact", "Email", "E-mail", "ID", "ID Number", "SSN", "DOB", "Date of Birth", "Total", "Total Amount"]

SCHEMA_REGISTRY = {
    "Resume": {
        "fields": ["Full Name", "Email", "Phone", "Technical Skills", "Education", "Latest Job Role"],
        "rag_query": "Who is the candidate? Contact details?"
    },
    "Invoice": {
        "fields": ["Vendor Name", "Invoice Number", "Date", "Total Amount"],
        "rag_query": "Vendor and total?"
    },
    "ID Card": {
        "fields": ["Full Name", "ID Number", "Date of Birth", "Expiration Date"],
        "rag_query": "ID details?"
    },
    "General": { "fields": ["Main Topic", "Summary"], "rag_query": "Summary?" }
}

# --- STATE ---
class AgentState(TypedDict):
    file_path: str
    poppler_path: str
    pages: List[Any]
    layout_image: Any      
    layout_data: dict      
    raw_text: str          
    translated_text: str    # NEW: Stores the English version
    detected_language: str  # NEW: Stores origin language
    doc_category: str
    entities: dict
    validation_report: dict
    redacted_image: Any     
    summary: str
    metadata: dict

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: raise ValueError("❌ GROQ_API_KEY missing!")
    return Groq(api_key=api_key)

# --- HELPER: TRANSLATION ---
def translate_to_english(text):
    """
    Uses LLM to detect language and translate to English if needed.
    """
    if not text or len(text) < 10: return "English", text
    
    client = get_groq_client()
    # Efficient prompt to detect and translate in one go
    prompt = f"""
    Task: Translate the following text to English.
    
    Rules:
    1. If it is already in English, return it exactly as is.
    2. If it is in another language, translate it to English.
    3. Output format: First line: [Language Name], Second line onwards: [Translated Text]
    
    Text:
    {text[:3000]} 
    """
    # We truncate to 3000 chars for speed in this demo
    
    try:
        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant"
        )
        content = resp.choices[0].message.content.strip()
        
        # Parse output
        lines = content.split('\n', 1)
        lang = lines[0].strip().replace("[", "").replace("]", "")
        translated = lines[1].strip() if len(lines) > 1 else text
        
        return lang, translated
    except:
        return "Unknown", text

# --- NODE 1: PERCEPTION (Updated with Translation) ---
def load_and_ocr_node(state: AgentState):
    print("\n--- 👁️ NODE: Perception ---")
    file_path = state['file_path']
    file_type = state['metadata'].get('file_type', 'pdf')
    
    extracted_text = ""
    pages = []
    layout_img = None
    layout_info = {}
    ocr_scores = []
    cropped_images = [] 
    crop_metadata = []

    try:
        if file_type == 'image':
            img = Image.open(file_path).convert('RGB')
            pages.append(img)
        else:
            pages = convert_from_path(file_path, poppler_path=state['poppler_path'])
        
        # Layout
        if layout_engine and pages:
            layout_info, layout_plot_array = layout_engine.analyze_layout(pages[0])
            layout_img = Image.fromarray(layout_plot_array[..., ::-1])
            for block in layout_info.get('text_blocks', []):
                box = block['box']
                if box[2] > box[0] and box[3] > box[1]:
                    crop = pages[0].crop((box[0], box[1], box[2], box[3]))
                    cropped_images.append(crop)
                    crop_metadata.append(block['label'])

        # OCR
        full_text_list = []
        for i, page in enumerate(pages):
            text = pytesseract.image_to_string(page)
            full_text_list.append(text)
            if validator:
                ocr_scores.append(validator.calculate_ocr_confidence(page))
        
        extracted_text = "\n\n".join(full_text_list)
        avg_score = sum(ocr_scores)/len(ocr_scores) if ocr_scores else 0
        
        # --- NEW: TRANSLATION LAYER ---
        print("   🌍 Checking Language...")
        lang, translated_text = translate_to_english(extracted_text)
        print(f"   🗣️ Detected: {lang}")
        
        # Index the TRANSLATED text (So RAG works in English)
        if rag_engine:
            if translated_text.strip(): 
                rag_engine.index_document(translated_text, {"source": file_path, "original_lang": lang})
            if cropped_images: 
                rag_engine.index_images(cropped_images, crop_metadata)
            
        return {
            "raw_text": extracted_text,
            "translated_text": translated_text, # Passed to other agents
            "detected_language": lang,
            "pages": pages,
            "layout_image": layout_img, 
            "layout_data": layout_info,
            "metadata": {**state['metadata'], "ocr_score": avg_score}
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"raw_text": "", "metadata": {"error": str(e)}}

# --- NODE 2: CLASSIFIER (Uses Translated Text) ---
def classification_agent(state: AgentState):
    # Use translated text for better understanding
    text = state.get('translated_text', state['raw_text'])[:2000]
    
    if not text.strip(): return {"doc_category": "Unknown"}
    client = get_groq_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Classify ONE: [Resume, Invoice, ID Card, General]. Text: {text}"}],
        model="llama-3.1-8b-instant",
    )
    return {"doc_category": response.choices[0].message.content.strip()}

# --- NODE 3: EXTRACTOR (Uses Translated Text) ---
def entity_extraction_agent(state: AgentState):
    category = state.get('doc_category', 'General')
    schema = SCHEMA_REGISTRY.get(category, SCHEMA_REGISTRY["General"])
    
    # RAG Context (Already Indexed in English)
    context = ""
    if rag_engine: context = rag_engine.query(schema["rag_query"])
    else: context = state.get('translated_text', "")[:4000]

    client = get_groq_client()
    prompt = f"Extract JSON. Context: {context}. Fields: {json.dumps(schema['fields'])}. Return JSON only."
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}, {"role": "system", "content": "Return JSON only."}],
        model="llama-3.1-8b-instant",
    )
    
    entities = {}
    validation_report = {}
    redacted_img = None

    try:
        content = response.choices[0].message.content
        start = content.find('{')
        end = content.rfind('}') + 1
        entities = json.loads(content[start:end])
        
        # Validate against ORIGINAL text (to ensure OCR match) or Translated? 
        # Validator matches text on page, so we use raw_text for validation to be safe.
        if validator:
            validation_report = validator.validate_extraction(entities, state['raw_text'])
            
        if redactor and state['pages']:
            values_to_hide = []
            if isinstance(entities, dict):
                for key, val in entities.items():
                    if key in SENSITIVE_KEYS and val:
                        values_to_hide.append(str(val))
            
            # Redactor uses Image + Regex, so it works on the original visual doc
            redacted_img = redactor.redact_sensitive_data(state['pages'][0], values_to_hide)

    except Exception as e:
        print(f"❌ Extraction Error: {e}")
        
    return {
        "entities": entities, 
        "validation_report": validation_report,
        "redacted_image": redacted_img 
    }

# --- NODE 4: SUMMARIZER ---
def summarization_agent(state: AgentState):
    text = state.get('translated_text', "")[:4000]
    client = get_groq_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Summarize in 3 sentences. Text: {text}"}],
        model="llama-3.1-8b-instant",
    )
    return {"summary": response.choices[0].message.content}

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("ocr", load_and_ocr_node)
workflow.add_node("classifier", classification_agent)
workflow.add_node("extractor", entity_extraction_agent)
workflow.add_node("summarizer", summarization_agent)
workflow.set_entry_point("ocr")
workflow.add_edge("ocr", "classifier")
workflow.add_edge("classifier", "extractor")
workflow.add_edge("classifier", "summarizer")
workflow.add_edge("extractor", END)
workflow.add_edge("summarizer", END)
app = workflow.compile()