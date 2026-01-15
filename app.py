import streamlit as st
import os
import json
import sys
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

# --- PATH SETUP ---
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import Agents & Utils
from src.agents.orchestrator import app as langgraph_agent, rag_engine, layout_engine
try:
    from src.utils.cloud_handler import CloudManager
    cloud_manager = CloudManager()
except ImportError:
    cloud_manager = None

# --- PAGE CONFIG ---
st.set_page_config(page_title="EYEON-AI", page_icon="👁️", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #008080; color: white; font-weight: bold;} /* Teal for 'Eye' vibe */
    .chat-message { padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex; }
    .chat-message.user { background-color: #f0f2f6; }
    .chat-message.bot { background-color: #e0f7fa; } /* Light cyan for bot */
    </style>
    """, unsafe_allow_html=True)

# --- UPDATED TITLE ---
st.title("👁️ EYEON-AI: Cognitive Document Agent")
st.markdown("Features: **Cloud Storage**, **Multilingual**, **Human-in-the-Loop**, **Privacy Redaction**, **Voice Chat**, & **Batch Processing**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Status")
    st.success("🟢 Orchestrator: Online")
    if rag_engine: st.success("🟢 RAG Memory: Active")
    else: st.error("🔴 RAG Memory: Offline")
    if layout_engine: st.success("🟢 Layout Engine: Active")
    st.divider()
    st.write("Developer: Gomathi D.")

# --- SESSION STATE ---
if "processed_data" not in st.session_state: st.session_state.processed_data = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# --- FEEDBACK SAVER FUNCTION ---
def save_feedback(original_file, doc_type, corrected_data):
    """Saves user corrections to a JSONL file for future training."""
    feedback_entry = {
        "timestamp": time.time(),
        "source_file": original_file,
        "doc_type": doc_type,
        "corrected_entities": corrected_data
    }
    
    os.makedirs("data/feedback", exist_ok=True)
    with open("data/feedback/training_log.jsonl", "a") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    return True

# --- MAIN INTERFACE (BATCH VS SINGLE) ---
uploaded_files = st.file_uploader("Upload Documents (Resume, Invoice, ID)", type=["pdf", "png", "jpg"], accept_multiple_files=True)

if uploaded_files:
    # ---------------------------------------------------------
    # 📊 MODE 1: BATCH PROCESSING (If >1 file)
    # ---------------------------------------------------------
    if len(uploaded_files) > 1:
        st.info(f"📦 Batch Mode Detected: Processing {len(uploaded_files)} documents...")
        
        if st.button("🚀 Process Batch"):
            progress_bar = st.progress(0)
            batch_results = []
            
            for i, file in enumerate(uploaded_files):
                # Save temp
                temp_path = f"data/batch/{file.name}"
                os.makedirs("data/batch", exist_ok=True)
                with open(temp_path, "wb") as f: f.write(file.getbuffer())
                
                # Run Agent
                state = {
                    "file_path": temp_path,
                    "poppler_path": r"C:\poppler\Library\bin",
                    "metadata": {"file_type": "image" if file.type.startswith("image") else "pdf"}
                }
                res = langgraph_agent.invoke(state)
                
                row = {"File": file.name, "Type": res.get('doc_category', 'Unknown')}
                entities = res.get('entities', {})
                if isinstance(entities, dict): row.update(entities)
                
                batch_results.append(row)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.success("✅ Batch Processing Complete!")
            
            # Show Table
            df = pd.DataFrame(batch_results).astype(str)
            st.dataframe(df, use_container_width=True)
            
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.download_button("📥 Download Summary (CSV)", df.to_csv(index=False), "batch_summary.csv")
            
            # --- CLOUD UPLOAD FOR BATCH ---
            with col_b2:
                if st.button("☁️ Upload to Cloud (S3/Drive)"):
                    if cloud_manager:
                        success, msg = cloud_manager.upload_json_data(batch_results, f"batch_export_{int(time.time())}.json")
                        if success: st.toast(msg, icon="✅")
                        else: st.error(msg)

    # ---------------------------------------------------------
    # 🕵️ MODE 2: SINGLE DOCUMENT (Deep Dive)
    # ---------------------------------------------------------
    else:
        file = uploaded_files[0]
        temp_path = f"data/sample_docs/{file.name}"
        os.makedirs("data/sample_docs", exist_ok=True)
        with open(temp_path, "wb") as f: f.write(file.getbuffer())

        if st.button("🚀 Analyze & Memorize"):
            with st.spinner("🤖 EYEON-AI is Reading, Redacting, and Memorizing..."):
                initial_state = {
                    "file_path": temp_path,
                    "poppler_path": r"C:\poppler\Library\bin", 
                    "metadata": {"file_type": "image" if file.type.startswith("image") else "pdf"}
                }
                result = langgraph_agent.invoke(initial_state)
                st.session_state.processed_data = result
                st.success("✅ Analysis Complete!")

        # --- RESULTS DISPLAY ---
        if st.session_state.processed_data:
            res = st.session_state.processed_data
            
            # Header Controls
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.subheader(f"📄 Analysis: {res.get('doc_category')}")
            with col_h2:
                # --- SINGLE FILE CLOUD UPLOAD ---
                if st.button("☁️ Push to Cloud"):
                    if cloud_manager:
                        success, msg = cloud_manager.upload_json_data(res, f"analysis_{file.name}_{int(time.time())}.json")
                        if success: st.toast(msg, icon="✅")
                        else: st.error(msg)
            
            tab1, tab2, tab3 = st.tabs(["📝 Data & Feedback", "📐 Layout (YOLO)", "💬 Multilingual Chat"])

            # --- TAB 1: HUMAN-IN-THE-LOOP EDITOR ---
            with tab1:
                col1, col2 = st.columns([1, 1])
                with col1:
                    detected_lang = res.get('detected_language', 'English')
                    st.caption(f"🌍 **Language:** {detected_lang}")
                    st.markdown(f"**📝 Summary:** {res.get('summary')}")
                
                with col2:
                    secure_mode = st.toggle("🛡️ Privacy Mode (Redact PII)", value=True)
                    if secure_mode and res.get('redacted_image'):
                        st.image(res['redacted_image'], caption="🔒 Secure View", width=350)
                    elif res.get('pages'):
                        st.image(res['pages'][0], caption="📄 Original View", width=350)

                st.divider()
                st.subheader("✍️ Human Verification (Edit to Fix Errors)")
                
                # --- ROBUST DATA PREPARATION ---
                entities = res.get('entities', {})
                report = res.get('validation_report', {})
                editor_data = []

                if isinstance(entities, dict) and entities:
                    for key, val in entities.items():
                        conf = 0
                        if isinstance(report, dict):
                            field_info = report.get(key)
                            if isinstance(field_info, dict):
                                conf = field_info.get("confidence_score", 0)

                        editor_data.append({
                            "Field": key,
                            "Extracted Value": str(val) if val is not None else "",
                            "Confidence": f"{conf}%",
                            "Status": "✅ Verified" if conf > 85 else "⚠️ Check"
                        })
                
                # --- DISPLAY EDITOR ---
                if editor_data:
                    df_editor = pd.DataFrame(editor_data)
                    
                    edited_df = st.data_editor(
                        df_editor,
                        column_config={
                            "Status": st.column_config.TextColumn(disabled=True),
                            "Confidence": st.column_config.TextColumn(disabled=True),
                            "Field": st.column_config.TextColumn(disabled=True),
                            "Extracted Value": st.column_config.TextColumn(
                                help="Double-click to edit this value"
                            )
                        },
                        use_container_width=True,
                        num_rows="dynamic",
                        key="editor_key_main"
                    )
                    
                    # Save Button
                    col_s1, col_s2 = st.columns([1, 4])
                    with col_s1:
                        if st.button("💾 Confirm & Train"):
                            if not edited_df.empty:
                                corrected_dict = dict(zip(edited_df["Field"], edited_df["Extracted Value"]))
                                save_feedback(file.name, res.get('doc_category'), corrected_dict)
                                
                                if cloud_manager:
                                    cloud_manager.upload_json_data(corrected_dict, f"CORRECTED_{file.name}.json")
                                    
                                st.toast("✅ Corrections synced to Cloud & Learning DB!", icon="🎉")
                                st.balloons()
                else:
                    st.info("ℹ️ No specific fields were extracted to edit.")

            # --- TAB 2: LAYOUT ---
            with tab2:
                if res.get('layout_image'):
                    st.image(res['layout_image'], caption="YOLOv11 Structure Detection", use_container_width=True)
                else:
                    st.warning("No layout detected.")

            # --- TAB 3: MULTILINGUAL CHAT ---
            with tab3:
                st.subheader("💬 Chat with Document")
                
                detected_lang = res.get('detected_language', 'English')
                if detected_lang != 'English':
                    st.info(f"💡 Document detected in **{detected_lang}**. Auto-translated to English. You can ask in {detected_lang}!")
                
                chat_container = st.container(height=400)
                
                for role, content in st.session_state.chat_history:
                    with chat_container.chat_message(role):
                        if isinstance(content, tuple): 
                            st.write(content[0])
                            if content[1]: st.image(content[1], width=300)
                        else:
                            st.write(content)

                col_v1, col_v2 = st.columns([1, 5])
                with col_v1:
                    try:
                        from streamlit_mic_recorder import speech_to_text
                        voice_text = speech_to_text(language='en', start_prompt="🎙️", stop_prompt="🛑", just_once=True, key='STT')
                    except ImportError:
                        st.error("Install `streamlit-mic-recorder`")
                        voice_text = None
                
                with col_v2:
                    txt_input = st.chat_input("Ask in any language...")

                user_query = voice_text if voice_text else txt_input

                if user_query:
                    st.session_state.chat_history.append(("user", user_query))
                    with chat_container.chat_message("user"):
                        st.write(user_query)
                    
                    with chat_container.chat_message("assistant"):
                        with st.spinner("Thinking & Translating..."):
                            context = rag_engine.query(user_query) if rag_engine else ""
                            visual_match = rag_engine.query_visuals(user_query) if rag_engine else None
                            
                            from groq import Groq
                            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                            prompt = f"""
                            Context from Document (English): {context}
                            User Question: {user_query}
                            Instructions:
                            1. Answer based ONLY on context.
                            2. Reply in the SAME language as the User Question.
                            """
                            resp = client.chat.completions.create(messages=[{"role":"user","content":prompt}], model="llama-3.1-8b-instant")
                            answer = resp.choices[0].message.content
                            
                            st.write(answer)
                            if visual_match: st.image(visual_match, width=300, caption="Visual Match")
                            st.session_state.chat_history.append(("assistant", (answer, visual_match)))