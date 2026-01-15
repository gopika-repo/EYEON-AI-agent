import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_models():
    # 1. Define and create the models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("⏳ Starting download of YOLOv11 Document Layout weights...")
    
    try:
        # 2. Download the .pt weights file
        # Repository: Armaggheddon/yolo11-document-layout
        # This model is specifically trained to detect: 
        # Text, Title, Table, Picture, Caption, etc.
        model_path = hf_hub_download(
            repo_id="Armaggheddon/yolo11-document-layout",
            filename="yolo11n_doc_layout.pt",
            local_dir="models",
            local_dir_use_symlinks=False  # Recommended for Windows/Git Bash to avoid permission issues
        )
        
        print(f"✅ Success! Weights saved to: {model_path}")
        
    except Exception as e:
        print(f"❌ Error downloading weights: {e}")
        print("Tip: Check your internet connection or verify the repo_id.")

if __name__ == "__main__":
    download_models()