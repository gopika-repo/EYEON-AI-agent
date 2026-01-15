import os
from src.agents.orchestrator import app

POPPLER_PATH = r"C:\poppler\Library\bin"

def run_pipeline():
    pdf_path = os.path.join("data", "sample_docs", "test_document.pdf")
    
    if not os.path.exists(pdf_path) or not os.path.exists(POPPLER_PATH):
        print("❌ Error: Missing PDF or Poppler path.")
        return

    print(f"🚀 Initializing Omni-Scribe Pipeline for: {os.path.basename(pdf_path)}")
    
    initial_state = {
        "file_path": pdf_path,
        "poppler_path": POPPLER_PATH,
        "pages": [],
        "raw_text": [],
        "metadata": {}
    }

    try:
        # Stream the graph execution
        for output in app.stream(initial_state):
            for key, value in output.items():
                print(f"--- Finished Step: {key} ---")
                if key == "extract_text" and "raw_text" in value:
                    print(f"📄 Detected {len(value['raw_text'])} text blocks.")
        
        print("\n🏆 PROJECT COMPLETE! Check 'extraction_results.json' for the data.")

    except Exception as e:
        print(f"💥 Pipeline Crash: {str(e)}")

if __name__ == "__main__":
    run_pipeline()