import os
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# --- CONFIGURATION ---
TEXT_MODEL_NAME = "all-MiniLM-L6-v2"
IMAGE_MODEL_NAME = "clip-ViT-B-32" # The bridge between Text and Images

class RAGEngine:
    def __init__(self):
        print("🔧 Initializing RAG Engine (Text + Vision)...")
        
        # 1. Text Brain (LangChain)
        self.text_embeddings = HuggingFaceEmbeddings(model_name=TEXT_MODEL_NAME)
        self.text_store = None
        
        # 2. Vision Brain (SentenceTransformers / CLIP)
        print("   👁️ Loading CLIP Model (This might download ~300MB once)...")
        self.vision_model = SentenceTransformer(IMAGE_MODEL_NAME)
        self.image_store = [] # Simple list to store [{img, embedding, id}]

    def index_document(self, raw_text: str, metadata: dict = None):
        """ Stores text in ChromaDB (RAM). """
        print("📚 Indexing Text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = [Document(page_content=raw_text, metadata=metadata or {})]
        splits = text_splitter.split_documents(docs)
        
        self.text_store = Chroma.from_documents(
            documents=splits, 
            embedding=self.text_embeddings
        )
        return True

    def index_images(self, images: list, metadata_list: list):
        """ 
        Encodes cropped images (tables/figures) into vectors.
        """
        print(f"🖼️ Indexing {len(images)} visual elements...")
        if not images:
            return
            
        # Convert images to vectors
        embeddings = self.vision_model.encode(images, convert_to_tensor=True)
        
        # Store in memory
        self.image_store = []
        for img, emb, meta in zip(images, embeddings, metadata_list):
            self.image_store.append({
                "image": img,
                "embedding": emb,
                "metadata": meta
            })
        print("✅ Visuals memorized.")

    def query(self, user_question: str):
        """ Text Search """
        if not self.text_store: return "⚠️ No text indexed."
        results = self.text_store.similarity_search(user_question, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    def query_visuals(self, user_question: str, threshold=0.25):
        """ 
        Text-to-Image Search: "Show me the table" -> [Image Object]
        """
        if not self.image_store: return None
        
        # 1. Convert user text to vector
        query_emb = self.vision_model.encode(user_question, convert_to_tensor=True)
        
        # 2. Compare with all image vectors
        best_match = None
        highest_score = -1
        
        for item in self.image_store:
            # Calculate Cosine Similarity
            score = util.cos_sim(query_emb, item['embedding']).item()
            
            if score > highest_score:
                highest_score = score
                best_match = item['image']
        
        print(f"   🔍 Visual Search Score: {highest_score:.2f}")
        
        # Only return if it's a decent match
        if highest_score > threshold:
            return best_match
        return None