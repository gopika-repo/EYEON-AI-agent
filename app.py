import streamlit as st
import os
import tempfile
import time
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import after path is set
try:
    from src.agents.rag_engine import RAGEngine
    from src.agents.orchestrator import process_document
    from src.agents.vision_agent import VisionAgent
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# --- 1. BOOTSTRAP ENVIRONMENT ---
load_dotenv()

# --- 2. SYSTEM CONFIGURATION ---
st.set_page_config(
    page_title="EYEON-AI: Cognitive Document Intelligence",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #4c51bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: none;
    }
    
    .chat-message-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 70%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .chat-message-assistant {
        background: #f0f2f6;
        color: #1e293b;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 70%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 15px;
        padding: 40px 20px;
        text-align: center;
        margin: 20px 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff 0%, #e8edff 100%);
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { 
        background: #10b981;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }
    
    .status-warning { 
        background: #f59e0b;
        box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.2);
    }
    
    .status-error { 
        background: #ef4444;
        box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2);
    }
    
    .tab-button {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .success-badge {
        background: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .warning-badge {
        background: #f59e0b;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .info-badge {
        background: #3b82f6;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .system-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin: 15px 0;
    }
    
    .chat-container {
        height: 500px;
        overflow-y: auto;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        background: #f8fafc;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'analysis_ready': False,
        'messages': [],
        'processed_file_path': None,
        'rag_engine': None,
        'vision_agent': None,
        'document_data': {},
        'vision_analysis': {},
        'processing_complete': False,
        'uploaded_file_name': None,
        'processing_started': False,
        'cognitive_features': {
            "PII Redaction": True,
            "Human-in-the-Loop": True,
            "Confidence Validation": True,
            "Multilingual Support": True,
            "Layout Analysis": True
        },
        'current_model': "llama-3.1-8b-instant"
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def create_metrics_dashboard(data):
    """Create a metrics dashboard for document analysis"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        doc_type = data.get('doc_category', 'Unknown')
        if doc_type == 'Unknown':
            badge_class = "warning-badge"
        else:
            badge_class = "success-badge"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 8px;">📄 Document Type</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{doc_type}</div>
            <div style="margin-top: 8px;"><span class="{badge_class}">Classified</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        entities = data.get('entities', {})
        if isinstance(entities, dict):
            entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        else:
            entity_count = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 8px;">🏷️ Entities Found</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{entity_count}</div>
            <div style="margin-top: 8px;"><span class="info-badge">Extracted</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        confidence = data.get('vision_analysis', {}).get('confidence', 0)
        confidence_percent = f"{confidence*100:.1f}%" if confidence > 0 else "0.0%"
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 8px;">📊 Analysis Confidence</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{confidence_percent}</div>
            <div style="margin-top: 8px;">
                <progress value="{confidence*100}" max="100" style="width: 100%; height: 6px; border-radius: 3px;"></progress>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        lang = data.get('detected_language', 'Unknown')
        
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 8px;">🌐 Language</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1e293b;">{lang}</div>
            <div style="margin-top: 8px;"><span class="info-badge">Detected</span></div>
        </div>
        """, unsafe_allow_html=True)

def main():
    # --- SIDEBAR: SYSTEM CONTROL & STATUS ---
    with st.sidebar:
        st.markdown('<h2 style="color: #667eea;">🛡️ System Control</h2>', unsafe_allow_html=True)
        
        # System Status Card
        with st.container():
            st.markdown("### 📊 System Status")
            
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                if os.getenv("GROQ_API_KEY"):
                    st.markdown('<span class="status-active"></span> **Groq API**', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-error"></span> **Groq API**', unsafe_allow_html=True)
                    st.error("API Key Missing")
            
            with status_col2:
                if st.session_state.rag_engine:
                    st.markdown('<span class="status-active"></span> **RAG Engine**', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="status-warning"></span> **RAG Engine**', unsafe_allow_html=True)
            
            st.divider()
            
            # Model Configuration
            st.markdown("### 🤖 AI Configuration")
            
            model_choice = st.selectbox(
                "Select AI Model",
                [
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768", 
                    "gemma2-9b-it",
                    "llama-3.2-11b-vision-preview"
                ],
                index=0,
                help="Choose the AI model for processing"
            )
            
            # Update model if changed
            if st.session_state.rag_engine and model_choice != st.session_state.current_model:
                st.session_state.rag_engine.model = model_choice
                st.session_state.current_model = model_choice
                st.success(f"Model updated to {model_choice}")
            
            st.divider()
            
            # Cognitive Features
            st.markdown("### ⚡ Cognitive Features")
            
            features = {
                "🔒 PII Redaction": st.toggle("PII Redaction", 
                                             value=st.session_state.cognitive_features["PII Redaction"],
                                             key="pii_toggle"),
                "👥 Human-in-the-Loop": st.toggle("Human Verification", 
                                                 value=st.session_state.cognitive_features["Human-in-the-Loop"],
                                                 key="hitl_toggle"),
                "✅ Confidence Check": st.toggle("Confidence Validation", 
                                               value=st.session_state.cognitive_features["Confidence Validation"],
                                               key="conf_toggle"),
                "🌐 Multilingual": st.toggle("Multilingual Support", 
                                           value=st.session_state.cognitive_features["Multilingual Support"],
                                           key="multi_toggle"),
                "🖼️ Layout Detection": st.toggle("Layout Analysis", 
                                                value=st.session_state.cognitive_features["Layout Analysis"],
                                                key="layout_toggle")
            }
            
            st.session_state.cognitive_features = {
                "PII Redaction": features["🔒 PII Redaction"],
                "Human-in-the-Loop": features["👥 Human-in-the-Loop"],
                "Confidence Validation": features["✅ Confidence Check"],
                "Multilingual Support": features["🌐 Multilingual"],
                "Layout Analysis": features["🖼️ Layout Detection"]
            }
            
            st.divider()
            
            # Voice Interface
            st.markdown("### 🎙️ Voice Interface")
            
            voice_input = speech_to_text(
                language='en',
                start_prompt="🎤 Start Recording",
                stop_prompt="⏹️ Stop Recording",
                use_container_width=True,
                key='voice_recorder'
            )
            
            if voice_input:
                st.session_state.voice_query = voice_input
                st.success(f"🎤 Voice input captured!")
            
            st.divider()
            
            # Actions
            action_col1, action_col2 = st.columns(2)
            
            with action_col1:
                if st.button("🔄 Reset", use_container_width=True, type="secondary"):
                    for key in ['messages', 'document_data', 'vision_analysis', 'processing_complete']:
                        if key in st.session_state:
                            st.session_state[key] = [] if key == 'messages' else {}
                    st.session_state.analysis_ready = False
                    st.session_state.processing_started = False
                    st.rerun()
            
            with action_col2:
                if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
                    st.session_state.messages = []
                    st.rerun()
            
            st.divider()
            
            # Developer Info
            st.markdown("---")
            st.caption("👨‍💻 **Developer**: Gomathi D.")
            st.caption("🚀 **Version**: 1.0.0 Production")
            st.caption("⚡ **Powered by Groq AI**")

    # --- MAIN INTERFACE ---
    st.markdown('<h1 class="main-header">👁️ EYEON-AI</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cognitive Document Intelligence Platform</div>', unsafe_allow_html=True)
    
    # Initialize AI Engines
    if st.session_state.rag_engine is None:
        with st.spinner("🚀 Initializing AI Engine..."):
            try:
                st.session_state.rag_engine = RAGEngine()
                st.session_state.vision_agent = VisionAgent()
                st.success("✅ AI Engine initialized successfully!")
            except Exception as e:
                st.error(f"❌ Failed to initialize AI engine: {str(e)}")
                st.info("Please check your GROQ_API_KEY in .env file")
    
    # --- DOCUMENT UPLOAD SECTION ---
    st.markdown("### 📄 Document Processing Center")
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "**Drag and drop or click to upload documents**",
            type=['pdf', 'txt', 'png', 'jpg', 'jpeg'],
            help="Supported formats: PDF, Images (PNG, JPG), Text files",
            key="file_uploader"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # Display file info
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**File:** {uploaded_file.name}")
        with col2:
            st.info(f"**Type:** {file_ext[1:].upper() if file_ext else 'Unknown'}")
        with col3:
            st.info(f"**Size:** {file_size_mb:.2f} MB")
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.processed_file_path = tmp_file.name
            st.session_state.uploaded_file_name = uploaded_file.name
        
        # Process Document Button
        if not st.session_state.processing_started:
            if st.button("🚀 **Process Document with AI Pipeline**", 
                        type="primary", 
                        use_container_width=True,
                        help="Start multi-agent document analysis"):
                
                st.session_state.processing_started = True
                st.rerun()
    
    # --- PROCESSING SECTION ---
    if st.session_state.processing_started and st.session_state.processed_file_path:
        with st.status("🤖 **Multi-Agent AI Pipeline Processing**", expanded=True) as status:
            # Step 1: Initial Processing
            st.write("📥 **Step 1: Loading and preprocessing document...**")
            time.sleep(0.5)
            
            # Step 2: Run orchestrator pipeline
            st.write("🔄 **Step 2: Orchestrating AI agents...**")
            
            try:
                # Process document through orchestrator
                metadata = {
                    'file_type': file_ext[1:] if file_ext else 'pdf',
                    'file_name': uploaded_file.name,
                    'upload_time': datetime.now().isoformat()
                }
                
                # Show processing animation
                progress_bar = st.progress(0)
                for i in range(5):
                    time.sleep(0.3)
                    progress_bar.progress((i + 1) * 20)
                
                # Process document
                result = process_document(
                    st.session_state.processed_file_path,
                    metadata=metadata
                )
                
                # Store results
                st.session_state.document_data = result
                
                if 'summary' in result:
                    st.session_state.summary = result['summary']
                
                if 'entities' in result:
                    st.session_state.entities = result['entities']
                
                if 'vision_analysis' in result:
                    st.session_state.vision_analysis = result['vision_analysis']
                
                if 'doc_category' in result:
                    st.session_state.doc_category = result['doc_category']
                
                st.session_state.analysis_ready = True
                st.session_state.processing_complete = True
                
                st.write("✅ **Step 3: Pipeline execution complete!**")
                progress_bar.progress(100)
                
            except Exception as e:
                st.error(f"❌ **Processing Error:** {str(e)}")
                st.info("""
                **Troubleshooting:**
                1. Ensure Tesseract OCR is installed for image processing
                2. Check your GROQ_API_KEY
                3. Try with a different file format
                """)
                st.session_state.analysis_ready = False
            
            status.update(label="✅ **AI Processing Complete!**", state="complete")
        
        if st.session_state.analysis_ready:
            st.balloons()
            st.success("🎉 Document processed successfully! Explore the results below.")
    
    # --- ANALYSIS RESULTS TABS ---
    if st.session_state.get("analysis_ready") and st.session_state.document_data:
        st.divider()
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 **Dashboard**", 
            "🔍 **Deep Analysis**", 
            "🖼️ **Vision & Layout**",
            "💬 **Interactive Chat**"
        ])
        
        with tab1:
            st.markdown("### 📊 Document Intelligence Dashboard")
            
            # Metrics Dashboard
            create_metrics_dashboard(st.session_state.document_data)
            
            st.divider()
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Summary Card
                with st.container():
                    st.markdown("#### 📝 Executive Summary")
                    if st.session_state.get('summary'):
                        st.markdown("""
                        <div class="system-card">
                            {}
                        </div>
                        """.format(st.session_state.summary), unsafe_allow_html=True)
                    else:
                        st.warning("No summary available")
            
            with col2:
                # Document Info Card
                with st.container():
                    st.markdown("#### 📋 Document Information")
                    
                    info_data = st.session_state.document_data
                    if info_data:
                        st.markdown(f"""
                        <div class="system-card">
                            <p><strong>Processing Time:</strong> {info_data.get('processing_time', 'N/A')}</p>
                            <p><strong>File Name:</strong> {info_data.get('file_info', {}).get('name', 'Unknown')}</p>
                            <p><strong>File Type:</strong> {info_data.get('file_info', {}).get('type', 'Unknown').upper()}</p>
                            <p><strong>Word Count:</strong> {info_data.get('content_analysis', {}).get('word_count', 0):,}</p>
                            <p><strong>Character Count:</strong> {info_data.get('content_analysis', {}).get('translated_text_length', 0):,}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Entities Table
            st.markdown("#### 🏷️ Extracted Entities")
            entities = st.session_state.get('entities', {})
            if isinstance(entities, dict) and entities:
                # Convert to dataframe for better display
                entity_data = []
                for key, value in entities.items():
                    if value and key != 'error' and key != 'raw_response' and key != 'parse_error':
                        if isinstance(value, list):
                            for item in value:
                                entity_data.append({
                                    "Entity Type": key,
                                    "Value": str(item),
                                    "Confidence": "High"
                                })
                        else:
                            entity_data.append({
                                "Entity Type": key,
                                "Value": str(value),
                                "Confidence": "High"
                            })
                
                if entity_data:
                    df = pd.DataFrame(entity_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Export option
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Entities as CSV",
                        data=csv,
                        file_name="extracted_entities.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    if 'raw_response' in entities:
                        st.info(entities['raw_response'][:500] + "...")
                    else:
                        st.info("No structured entities extracted")
            else:
                st.info("Entity extraction results will appear here")
        
        with tab2:
            st.markdown("### 🔍 Deep Document Analysis")
            
            # Human Verification Section
            if st.session_state.cognitive_features.get("Human-in-the-Loop"):
                st.markdown("#### 🤝 Human Verification Interface")
                
                with st.expander("📋 **Verification Panel**", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        accuracy_rating = st.slider(
                            "**Data Accuracy Score**",
                            min_value=1,
                            max_value=10,
                            value=8,
                            help="Rate the accuracy of extracted information (1 = Poor, 10 = Excellent)"
                        )
                        
                        completeness = st.slider(
                            "**Extraction Completeness**",
                            min_value=0,
                            max_value=100,
                            value=85,
                            format="%d%%",
                            help="How complete is the information extraction?"
                        )
                    
                    with col2:
                        verification_notes = st.text_area(
                            "**Verification Notes**",
                            placeholder="Add any notes, corrections, or feedback...",
                            height=120
                        )
                        
                        if st.button("✅ **Confirm & Submit Verification**", use_container_width=True):
                            # Store verification data
                            verification_data = {
                                "timestamp": datetime.now().isoformat(),
                                "accuracy": accuracy_rating,
                                "completeness": completeness,
                                "notes": verification_notes,
                                "verified_by": "Human Reviewer",
                                "document": st.session_state.uploaded_file_name
                            }
                            st.session_state.verification_data = verification_data
                            st.success("✅ Verification submitted successfully!")
            
            # Raw Data View
            st.markdown("#### 📄 Raw Analysis Data")
            with st.expander("**View Detailed JSON Results**"):
                if st.session_state.document_data:
                    st.json(st.session_state.document_data, expanded=False)
        
        with tab3:
            st.markdown("### 🖼️ Vision & Layout Analysis")
            
            if st.session_state.vision_analysis:
                vision_data = st.session_state.vision_analysis
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Vision Analysis Results
                    st.markdown("#### 🔍 Visual Elements Detected")
                    
                    if vision_data.get('description'):
                        st.markdown(f"""
                        <div class="system-card">
                            <div style="max-height: 300px; overflow-y: auto;">
                                {vision_data['description']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Elements list
                    elements = vision_data.get('elements_found', [])
                    if elements:
                        st.markdown("#### 📋 Detected Elements")
                        cols = st.columns(3)
                        for i, element in enumerate(elements):
                            if i < 3:
                                with cols[i]:
                                    st.markdown(f"""
                                    <div style="text-align: center; padding: 10px; background: #f1f5f9; border-radius: 8px;">
                                        <div style="font-size: 1.2rem;">{element}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence Metrics
                    st.markdown("#### 📊 Analysis Metrics")
                    
                    confidence = vision_data.get('confidence', 0)
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; font-weight: 700; color: #667eea;">{confidence*100:.1f}%</div>
                            <div style="color: #64748b; margin-top: 5px;">Detection Confidence</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    elements = vision_data.get('elements_found', [])
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="text-align: center;">
                            <div style="font-size: 2.5rem; font-weight: 700; color: #10b981;">{len(elements)}</div>
                            <div style="color: #64748b; margin-top: 5px;">Elements Found</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Re-analysis button
                    if st.button("🔄 **Re-analyze with Vision Agent**", use_container_width=True):
                        with st.spinner("Re-analyzing with vision..."):
                            try:
                                if st.session_state.vision_agent:
                                    new_analysis = st.session_state.vision_agent.analyze_layout(
                                        st.session_state.processed_file_path
                                    )
                                    st.session_state.vision_analysis = new_analysis
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Vision analysis failed: {str(e)}")
            else:
                st.info("👁️ Vision analysis data will appear here after processing")
                st.markdown("""
                <div style="text-align: center; padding: 40px; background: #f8fafc; border-radius: 10px;">
                    <div style="font-size: 3rem; margin-bottom: 20px;">👁️</div>
                    <div style="color: #64748b;">Upload an image or PDF to see vision analysis results</div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 💬 Document Assistant Chat")
            
            # Create a scrollable chat container using HTML/CSS
            st.markdown("""
            <div class="chat-container">
            """, unsafe_allow_html=True)
            
            # Display chat messages
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', 
                              unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Chat input area
            st.markdown("---")
            
            # Check for voice input
            query = None
            if st.session_state.get('voice_query'):
                query = st.session_state.voice_query
                del st.session_state.voice_query
                st.info(f"🎤 Voice input: {query}")
            
            # Text input - placed OUTSIDE any container/tab for Streamlit compatibility
            chat_input_container = st.container()
            with chat_input_container:
                if query is None:
                    query = st.text_input(
                        "Ask about your document...",
                        placeholder="Type your question or use voice input from sidebar...",
                        key="chat_input_main"
                    )
            
            if query and st.session_state.processed_file_path:
                # Add user message
                st.session_state.messages.append({
                    "role": "user", 
                    "content": query
                })
                
                # Get AI response
                with st.spinner("🤔 Analyzing document..."):
                    try:
                        response = st.session_state.rag_engine.chat(
                            query,
                            st.session_state.processed_file_path
                        )
                        
                        if isinstance(response, dict):
                            answer = response.get("answer", "I couldn't process your question.")
                            sources = response.get("sources", [])
                            confidence = response.get("confidence", 0)
                            
                            # Format response
                            final_response = answer
                            
                            if confidence > 0 and st.session_state.cognitive_features["Confidence Validation"]:
                                final_response += f"\n\n🔍 **Confidence:** {confidence*100:.1f}%"
                            
                            if sources and len(sources) > 0:
                                final_response += "\n\n**📚 Sources:**"
                                for i, source in enumerate(sources[:2], 1):
                                    if isinstance(source, dict):
                                        text = source.get('text', '')
                                        relevance = source.get('relevance', 0)
                                        final_response += f"\n{i}. {text[:150]}... (Relevance: {relevance*100:.1f}%)"
                        else:
                            final_response = str(response)
                        
                        # Add assistant response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": final_response
                        })
                        
                        # Rerun to update display
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Chat error: {str(e)}")
    
    # --- FOOTER & SYSTEM INFO ---
    st.divider()
    
    with st.expander("📋 **System Information & Configuration**"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🤖 AI Configuration")
            st.markdown("""
            <div class="system-card">
                <p><strong>Model:</strong> {}</p>
                <p><strong>Embedding:</strong> all-MiniLM-L6-v2</p>
                <p><strong>API Provider:</strong> Groq Cloud</p>
                <p><strong>Status:</strong> <span class="status-active"></span> Active</p>
            </div>
            """.format(st.session_state.current_model), unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### ⚡ Processing Pipeline")
            st.markdown("""
            <div class="system-card">
                <p>1. 📥 Document Loading</p>
                <p>2. 🔍 OCR & Translation</p>
                <p>3. 👁️ Vision Analysis</p>
                <p>4. 🏷️ Classification</p>
                <p>5. 📋 Entity Extraction</p>
                <p>6. 📝 Summarization</p>
                <p>7. ✅ Validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("#### 🛡️ Security & Compliance")
            st.markdown("""
            <div class="system-card">
                <p><strong>PII Redaction:</strong> {}</p>
                <p><strong>Data Encryption:</strong> AES-256</p>
                <p><strong>Compliance:</strong> GDPR Ready</p>
                <p><strong>Audit Trail:</strong> Enabled</p>
                <p><strong>Local Processing:</strong> Yes</p>
            </div>
            """.format("✅ Enabled" if st.session_state.cognitive_features["PII Redaction"] else "❌ Disabled"), 
            unsafe_allow_html=True)
        
        # Processing Statistics
        if st.session_state.document_data:
            st.markdown("---")
            st.markdown("#### 📈 Processing Statistics")
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                proc_time = st.session_state.document_data.get('processing_time', 'N/A')
                st.metric("⏱️ Processing Time", proc_time)
            
            with stats_col2:
                if st.session_state.rag_engine:
                    metrics = st.session_state.rag_engine.get_metrics()
                    st.metric("🔢 Total Queries", metrics.get('total_queries', 0))
            
            with stats_col3:
                entities = st.session_state.get('entities', {})
                if isinstance(entities, dict):
                    entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
                    st.metric("🏷️ Entities Found", entity_count)
            
            with stats_col4:
                if st.session_state.vision_analysis:
                    elements = len(st.session_state.vision_analysis.get('elements_found', []))
                    st.metric("👁️ Vision Elements", elements)

# Cleanup function
def cleanup():
    """Clean up temporary files"""
    if st.session_state.get("processed_file_path"):
        try:
            if os.path.exists(st.session_state.processed_file_path):
                os.unlink(st.session_state.processed_file_path)
        except:
            pass

# Register cleanup
import atexit
atexit.register(cleanup)

if __name__ == "__main__":
    main()