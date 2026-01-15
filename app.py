import streamlit as st
from streamlit_mic_recorder import speech_to_text
import time

# --- Restored 2026 Layout Configuration ---
st.set_page_config(page_title="EYEON-AI", layout="wide")

def main():
    st.title("👁️ EYEON-AI: Cognitive Document Agent")

    # --- Session State for Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Sidebar: Voice Interface & Document Control ---
    with st.sidebar:
        st.header("📂 Document Control")
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        st.divider()
        st.subheader("🎙️ Voice Command")
        
        # Use speech_to_text for high-accuracy Google API transcription
        voice_text = speech_to_text(
            language='en', 
            start_prompt="⏺️ Push to Talk", 
            stop_prompt="⏹️ Stop", 
            just_once=True, 
            key='STT'
        )
        
        if voice_text:
            st.info(f"Captured: {voice_text}")
            # Inject voice transcript into chat session
            st.session_state.messages.append({"role": "user", "content": voice_text})
            # Trigger RAG Brain (Assume rag_engine is initialized elsewhere)
            # response = rag_engine.chat(voice_text, tmp_path)
            # st.session_state.messages.append({"role": "assistant", "content": response})

    # --- Main Chat Display ---
    chat_container = st.container()
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Standard Text Input ---
    if prompt := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Placeholder for AI Response
        with st.chat_message("assistant"):
            st.write("Processing...")

if __name__ == "__main__":
    main()