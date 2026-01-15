import torch
import ultralytics
import qdrant_client
import importlib.metadata

print("--- Setup Verification ---")
print(f"CUDA Available (GPU Support): {torch.cuda.is_available()}")
print(f"YOLO Ready: {ultralytics.__version__}")

# A safer way to check the version of qdrant-client
try:
    version = importlib.metadata.version("qdrant-client")
    print(f"Qdrant Client Ready: {version}")
except:
    print("Qdrant Client is installed and ready!")

print("--- ALL SYSTEMS GO ---")
