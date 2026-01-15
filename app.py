# --- 1. CLOUD DATABASE FIX (MUST BE AT THE VERY TOP) ---
# This specific block fixes the "Offline RAG" / "Red Light" on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    # This just means you are running locally on Windows, which is fine!
    pass

# --- 2. STANDARD IMPORTS ---
import streamlit as st
import os
import tempfile
import time

# --- 3. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EYEON-AI Agent",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 4. IMPORT YOUR LOCAL MODULES ---
# ⚠️ MAKE SURE THESE MATCH YOUR FILE NAMES IN THE 'SRC' FOLDER
try:
    from src.utils.pdf_helper import pdf_to_images  # The file we fixed earlier!
    # If your RAG logic is in a file named 'rag_engine.py', keep this.
    # If it is named something else, change 'src.rag_engine' to that name.
    from src.rag_engine import RAGEngine 
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.info("Make sure your 'src' folder has an '__init__.py' file and the names match.")
    st.stop()

# --- 5. SETUP SECRETS & KEYS ---
def setup_keys():
    # Try to load from Streamlit Secrets (Cloud)
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    
    # Check if they exist now
    if not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ Google API Key is missing. Check your Secrets.")
        return False
    return True

# --- 6. CACHED RESOURCE LOADING (Fixes Blank Screen) ---
@st.cache_resource
def load_rag_brain():
    """
    Load the heavy AI models only once.
    This prevents the app from going 'Blank' on every reload.
    """
    try:
        return RAGEngine()
    except Exception as e:
        st.error(f"Failed to load AI Brain: {e}")
        return None

# --- 7. MAIN APP UI ---
def main():
    st.title("👁️ EYEON-AI: Cognitive Document Agent")
    st.markdown("""
    Features: **Cloud Storage**, **Multilingual**, **Privacy Redaction**, **Voice Chat**.
    """)

    # Initialize Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize Keys
    keys_loaded = setup_keys()

    # Sidebar
    with st.sidebar:
        st.header("📂 Document Control")
        uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
        
        # System Status Indicators
        st.divider()
        st.subheader("System Status")
        st.success("🟢 Orchestrator: Online")
        
        # DYNAMIC RAG STATUS
        rag_engine = load_rag_brain()
        if rag_engine and keys_loaded:
             st.success("🟢 RAG Memory: Active")
        else:
             st.error("🔴 RAG Memory: Offline")

    # Processing Logic
    if uploaded_file and keys_loaded:
        with st.spinner("👀 Analyzing Document..."):
            # Create a temporary file to handle the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # --- VISUALIZATION TAB ---
            tab1, tab2 = st.tabs(["📄 Visual Analysis", "💬 AI Chat"])
            
            with tab1:
                # Use our smart PDF helper (works on Linux & Windows)
                images = pdf_to_images(tmp_path)
                st.image(images[0], caption="First Page Preview", use_container_width=True)
                
            with tab2:
                # Chat Interface
                prompt = st.chat_input("Ask about this document...")
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    # Generate Response
                    with st.chat_message("assistant"):
                        if rag_engine:
                            # Assuming your RAG engine has a method like 'chat' or 'query'
                            # Adjust this line to match your actual function name!
                            response = rag_engine.chat(prompt, tmp_path) 
                            st.write(response)
                        else:
                            st.write("⚠️ Memory is offline. Check logs.")

            # Cleanup
            os.remove(tmp_path)

if __name__ == "__main__":
    main()