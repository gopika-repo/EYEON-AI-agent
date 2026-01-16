# test_complete.py
import os
from dotenv import load_dotenv

load_dotenv()

print("🧪 Final Comprehensive Test")
print("="*50)

# Test all imports
imports = [
    ("streamlit", None),
    ("groq", "Groq"),
    ("sentence_transformers", "SentenceTransformer"),
    ("PyPDF2", None),
    ("numpy", None),
]

print("\n📦 Testing imports...")
for module, attr in imports:
    try:
        if attr:
            exec(f"from {module} import {attr}")
            print(f"  ✅ {module}.{attr}")
        else:
            __import__(module)
            print(f"  ✅ {module}")
    except Exception as e:
        print(f"  ❌ {module}: {type(e).__name__}")

# Test API key
print(f"\n🔑 GROQ_API_KEY: {'✅ Set' if os.environ.get('GROQ_API_KEY') else '❌ Missing'}")

# Test creating SentenceTransformer model
print("\n🤖 Testing SentenceTransformer...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✅ Model loaded successfully")
    
    # Quick embedding test
    embeddings = model.encode("Hello world")
    print(f"  ✅ Embeddings generated: {embeddings.shape}")
except Exception as e:
    print(f"  ❌ Error: {type(e).__name__}: {str(e)[:100]}")

# Test RAGEngine
print("\n⚙️ Testing RAGEngine...")
try:
    from src.agents.rag_engine import RAGEngine
    print("  ✅ RAGEngine import successful")
    
    if os.environ.get("GROQ_API_KEY"):
        rag = RAGEngine()
        print("  ✅ RAGEngine initialized")
    else:
        print("  ⚠️ RAGEngine requires GROQ_API_KEY")
except ImportError:
    print("  ❌ RAGEngine not found in src/agents/")
    print("  ℹ️ Make sure your rag_engine.py is in the correct location")
except Exception as e:
    print(f"  ❌ RAGEngine error: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*50)
print("🎯 READY TO RUN!")
print("Command: streamlit run app.py")