import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# ✅ FIXED: This is the ONLY way to use RetrievalQA in LangChain v1.0+
from langchain_classic.chains import RetrievalQA

# ✅ FIXED: Text splitters are now in their own separate package
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

class RAGEngine:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def chat(self, query, pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            splits = self.text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                collection_name="temp_pdf_chat" 
            )

            retriever = vectorstore.as_retriever()
            
            # Using the classic chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False
            )

            # ✅ IMPORTANT: LangChain v1.0 REQUIRES .invoke() instead of .run()
            response = qa_chain.invoke({"query": query})
            
            vectorstore.delete_collection()
            return response['result']

        except Exception as e:
            return f"Error: {str(e)}"