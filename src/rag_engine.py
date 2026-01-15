import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ✅ FIXED: RetrievalQA moved to its own internal path in v0.3
from langchain_community.chains import RetrievalQA

# ✅ FIXED: Text splitters now have their own dedicated package
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

class RAGEngine:
    def __init__(self):
        # Initialize Google Models
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def chat(self, query, pdf_path):
        try:
            # 1. Load PDF
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # 2. Split Text into chunks
            splits = self.text_splitter.split_documents(docs)

            # 3. Create Vector Store (Memory)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="temp_pdf_chat" 
            )

            # 4. Create the Q&A Chain
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )

            # 5. Get Response
            response = qa_chain.invoke({"query": query})
            
            # Cleanup
            vectorstore.delete_collection()
            
            return response['result']

        except Exception as e:
            return f"Error processing document: {str(e)}"