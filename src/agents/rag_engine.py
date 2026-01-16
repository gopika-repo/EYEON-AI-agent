import os
import PyPDF2
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Tuple
import hashlib
from dataclasses import dataclass
import json
import concurrent.futures
import time

@dataclass
class DocumentChunk:
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    page: int
    start_pos: int
    end_pos: int

class EnhancedRAGEngine:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        # Initialize Groq client with better error handling
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY is not set in environment variables")
        
        try:
            self.client = Groq(api_key=api_key, timeout=30.0)
            self.model = model_name
            
            # Load embedding model with better configuration
            print("🚀 Loading embedding model...")
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu'  # Force CPU for compatibility
            )
            
            # Document cache for better performance
            self.document_cache = {}
            self.chunk_cache = {}
            
            # Performance metrics
            self.metrics = {
                "total_queries": 0,
                "avg_response_time": 0,
                "cache_hits": 0
            }
            
            print("✅ RAG Engine initialized successfully!")
            
        except Exception as e:
            raise Exception(f"Failed to initialize RAG Engine: {str(e)}")
    
    def _extract_text_enhanced(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced text extraction with multiple fallbacks"""
        text = ""
        metadata = {
            "pages": 0,
            "extraction_method": "",
            "has_tables": False,
            "has_images": False
        }
        
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(pdf_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    
                    # Check for tables
                    tables = page.extract_tables()
                    if tables:
                        metadata["has_tables"] = True
                        for table in tables:
                            table_text = " | ".join([" | ".join(filter(None, row)) for row in table if any(row)])
                            page_text += f"\n\n[Table {len(tables)}]: {table_text}"
                    
                    text += page_text + "\n\n"
                
                metadata["extraction_method"] = "pdfplumber"
                
        except Exception as e1:
            print(f"pdfplumber failed: {e1}, trying PyPDF2...")
            try:
                # Fallback to PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata["pages"] = len(reader.pages)
                    
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n\n"
                    
                    metadata["extraction_method"] = "PyPDF2"
                    
            except Exception as e2:
                print(f"PyPDF2 failed: {e2}")
                raise Exception(f"Failed to extract text from PDF: {str(e2)}")
        
        return text.strip(), metadata
    
    def _chunk_text_semantic(self, text: str, chunk_size: int = 800, overlap: int = 150) -> List[DocumentChunk]:
        """Semantic-aware text chunking"""
        if not text:
            return []
        
        # First split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If paragraph itself is too large, split it
            if para_length > chunk_size:
                sentences = para.replace('. ', '.\n').split('\n')
                for sent in sentences:
                    sent_length = len(sent)
                    if current_length + sent_length > chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                        
                        if chunk_hash not in self.chunk_cache:
                            embedding = self.embedding_model.encode(chunk_text)
                            self.chunk_cache[chunk_hash] = embedding
                        
                        chunks.append(DocumentChunk(
                            text=chunk_text,
                            embedding=self.chunk_cache[chunk_hash],
                            metadata={"type": "paragraph", "hash": chunk_hash},
                            page=0,
                            start_pos=0,
                            end_pos=len(chunk_text)
                        ))
                        
                        # Start new chunk with overlap
                        overlap_words = ' '.join(current_chunk[-20:]) if len(current_chunk) > 20 else ' '.join(current_chunk)
                        current_chunk = [overlap_words] if overlap_words else []
                        current_length = len(overlap_words)
                    
                    current_chunk.append(sent)
                    current_length += sent_length
            
            # Normal paragraph handling
            elif current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
                
                if chunk_hash not in self.chunk_cache:
                    embedding = self.embedding_model.encode(chunk_text)
                    self.chunk_cache[chunk_hash] = embedding
                
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    embedding=self.chunk_cache[chunk_hash],
                    metadata={"type": "paragraph", "hash": chunk_hash},
                    page=0,
                    start_pos=0,
                    end_pos=len(chunk_text)
                ))
                
                # Start new chunk with overlap
                overlap_words = ' '.join(current_chunk[-10:]) if len(current_chunk) > 10 else ' '.join(current_chunk)
                current_chunk = [overlap_words, para] if overlap_words else [para]
                current_length = len(overlap_words) + para_length if overlap_words else para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            
            if chunk_hash not in self.chunk_cache:
                embedding = self.embedding_model.encode(chunk_text)
                self.chunk_cache[chunk_hash] = embedding
            
            chunks.append(DocumentChunk(
                text=chunk_text,
                embedding=self.chunk_cache[chunk_hash],
                metadata={"type": "paragraph", "hash": chunk_hash},
                page=0,
                start_pos=0,
                end_pos=len(chunk_text)
            ))
        
        return chunks
    
    def _get_relevant_chunks_enhanced(self, query: str, text: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Enhanced semantic search with metadata"""
        start_time = time.time()
        self.metrics["total_queries"] += 1
        
        # Check cache first
        query_hash = hashlib.md5(query.encode()).hexdigest()
        if query_hash in self.document_cache:
            self.metrics["cache_hits"] += 1
            return self.document_cache[query_hash]
        
        if not text or not query:
            return "", []
        
        # Chunk text
        chunks = self._chunk_text_semantic(text)
        if not chunks:
            return "", []
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities
        similarities = []
        for chunk in chunks:
            similarity = np.dot(query_embedding, chunk.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
            )
            similarities.append((similarity, chunk))
        
        # Sort by similarity and get top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_chunks = similarities[:top_k]
        
        # Build context and sources
        context_parts = []
        sources = []
        
        for i, (similarity, chunk) in enumerate(top_chunks):
            context_parts.append(f"[Context {i+1}, Relevance: {similarity:.2%}]:\n{chunk.text}")
            sources.append({
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "relevance": float(similarity),
                "metadata": chunk.metadata
            })
        
        context = "\n\n".join(context_parts)
        
        # Update cache
        self.document_cache[query_hash] = (context, sources)
        
        # Update metrics
        process_time = time.time() - start_time
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["total_queries"] - 1) + process_time) 
            / self.metrics["total_queries"]
        )
        
        return context, sources
    
    def _call_groq_enhanced(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """Enhanced Groq API call with better error handling"""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document analysis assistant. Provide accurate, concise, and well-structured responses based only on the given context."
                    },
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            if "rate limit" in str(e).lower():
                return "⚠️ Rate limit exceeded. Please try again in a moment."
            elif "timeout" in str(e).lower():
                return "⏰ Request timeout. The server is taking too long to respond."
            else:
                return f"❌ API Error: {str(e)[:100]}"
    
    def get_summary(self, pdf_path: str) -> Dict[str, Any]:
        """Generate comprehensive summary with structure"""
        try:
            text, metadata = self._extract_text_enhanced(pdf_path)
            
            if len(text) < 50:
                return {
                    "summary": "Document is too short or could not be read.",
                    "metadata": metadata,
                    "status": "error"
                }
            
            prompt = f"""Analyze this document and provide a structured summary:

DOCUMENT CONTENT (truncated for brevity):
{text[:6000]}

Please provide a structured summary with the following sections:

1. **Document Overview**: What type of document is this? What is its main purpose?
2. **Key Information**: What are the most important facts, figures, or data points?
3. **Main Topics/Chapters**: What are the primary sections or topics covered?
4. **Critical Findings**: Any significant discoveries, conclusions, or recommendations?
5. **Action Items**: What needs to be done based on this document?

Format your response with clear section headers and bullet points where appropriate."""

            summary = self._call_groq_enhanced(prompt, max_tokens=1024)
            
            return {
                "summary": summary,
                "metadata": metadata,
                "status": "success",
                "word_count": len(text.split()),
                "char_count": len(text)
            }
            
        except Exception as e:
            return {
                "summary": f"Summary Error: {str(e)}",
                "metadata": {},
                "status": "error"
            }
    
    def get_entities(self, pdf_path: str) -> Dict[str, Any]:
        """Extract structured entities with categories"""
        try:
            text, metadata = self._extract_text_enhanced(pdf_path)
            
            if len(text) < 50:
                return {
                    "entities": {},
                    "metadata": metadata,
                    "status": "error",
                    "message": "Document is too short"
                }
            
            prompt = f"""Extract and categorize all important entities from this document:

TEXT:
{text[:4000]}

Extract entities in these categories:
1. **People**: Names, titles, roles
2. **Organizations**: Companies, institutions, departments
3. **Dates**: Important dates, deadlines, timelines
4. **Financial**: Amounts, prices, costs, budgets
5. **Locations**: Addresses, places, regions
6. **Technical Terms**: Specific terminology, acronyms
7. **Contact Info**: Emails, phones, URLs

Format as a JSON object with categories as keys and arrays of entities as values.
Example:
{{
    "People": ["John Doe", "Jane Smith"],
    "Organizations": ["ABC Corp", "XYZ University"],
    "Dates": ["2024-01-15", "Q2 2024"]
}}

Return ONLY the JSON object, no additional text."""
            
            response = self._call_groq_enhanced(prompt, max_tokens=1024, temperature=0.0)
            
            # Try to parse JSON
            try:
                # Find JSON in response
                start = response.find('{')
                end = response.rfind('}') + 1
                if start != -1 and end != 0:
                    entities = json.loads(response[start:end])
                else:
                    entities = {"raw_response": response, "parse_error": "No JSON found"}
            except json.JSONDecodeError as je:
                entities = {"raw_response": response, "parse_error": str(je)}
            
            return {
                "entities": entities,
                "metadata": metadata,
                "status": "success",
                "entity_count": sum(len(v) for v in entities.values() if isinstance(v, list))
            }
            
        except Exception as e:
            return {
                "entities": {"error": str(e)},
                "metadata": {},
                "status": "error"
            }
    
    def chat(self, query: str, pdf_path: str) -> Dict[str, Any]:
        """Enhanced chat with document context"""
        try:
            # Extract text
            text, metadata = self._extract_text_enhanced(pdf_path)
            
            if len(text) < 50:
                return {
                    "answer": "Document is too short or could not be read.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": metadata
                }
            
            # Get relevant context
            context, sources = self._get_relevant_chunks_enhanced(query, text)
            
            if not context:
                return {
                    "answer": "I couldn't find relevant information in the document to answer your question.",
                    "sources": [],
                    "confidence": 0.0,
                    "metadata": metadata
                }
            
            # Create enhanced prompt
            prompt = f"""You are a document analysis assistant. Answer the question based ONLY on the provided context.

DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the answer is not in the context, say "Based on the document, I cannot find information about this."
3. Be precise and concise
4. Include relevant details from the context
5. If discussing numerical data, mention specific numbers
6. Format your answer clearly with paragraphs and bullet points if needed

ANSWER:"""
            
            # Get answer
            answer = self._call_groq_enhanced(prompt, max_tokens=1024)
            
            # Calculate confidence based on source relevance
            confidence = 0.0
            if sources:
                avg_relevance = sum(s.get('relevance', 0) for s in sources) / len(sources)
                confidence = min(avg_relevance * 1.5, 1.0)  # Scale but cap at 1.0
            
            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "metadata": metadata,
                "context_length": len(context),
                "response_time": self.metrics["avg_response_time"]
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0,
                "metadata": {}
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        return {
            **self.metrics,
            "cache_size": len(self.document_cache),
            "chunk_cache_size": len(self.chunk_cache),
            "model": self.model,
            "embedding_model": "all-MiniLM-L6-v2"
        }

# For backward compatibility
RAGEngine = EnhancedRAGEngine