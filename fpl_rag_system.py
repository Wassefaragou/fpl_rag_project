import os
import json
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class FPLRagSystem:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo"):
        """Initialize the FPL RAG system"""
        self.openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        self.vector_store = None
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
    
    def load_documents(self, json_file_path):
        """Load documents from a JSON file"""
        with open(json_file_path, 'r') as f:
            docs_data = json.load(f)
        
        documents = []
        for doc in docs_data:
            documents.append(
                Document(
                    page_content=doc['content'],
                    metadata=doc['metadata']
                )
            )
        
        # Split documents into chunks if they're large
        split_docs = self.text_splitter.split_documents(documents)
        print(f"Loaded and split {len(documents)} documents into {len(split_docs)} chunks")
        return split_docs
    
    def create_vector_store(self, documents):
        """Create a vector store from documents"""
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        print("Vector store created successfully")
        
    def save_vector_store(self, directory_path="fpl_vectorstore"):
        """Save the vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(directory_path)
            print(f"Vector store saved to {directory_path}")
        else:
            print("No vector store to save")
    
    def load_vector_store(self, directory_path="fpl_vectorstore"):
        """Load a vector store from disk"""
        if os.path.exists(directory_path):
            self.vector_store = FAISS.load_local(directory_path, self.embeddings)
            print(f"Vector store loaded from {directory_path}")
        else:
            print(f"No vector store found at {directory_path}")
    
    def setup_retrieval_chain(self):
        """Set up the retrieval and generation chain"""
        # Create the retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Create the prompt template
        
        template = """
        You are a Fantasy Premier League expert assistant. Use the following information about FPL players to answer the question.
        
        Context information from the FPL database:
        {context}
        
        Question: {question}
        
        When answering:
        1. If the context doesn't contain relevant information, say what you know about the topic but clarify that you don't have current FPL data on this specific point.
        2. Cite specific stats when they're available in the context.
        3. Be concise and to the point.
        4. If appropriate, suggest actions the user might take based on the information (e.g., transfers, captain picks).
        
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain
        self.chain = (
            {"context": retriever, "question": lambda x: x}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        print("Retrieval chain set up successfully")
    
    def query(self, question):
        """Query the RAG system with a question"""
        if not self.vector_store:
            print("Vector store not initialized. Please load or create one first.")
            return None
        
        if not hasattr(self, 'chain'):
            self.setup_retrieval_chain()
        
        return self.chain.invoke(question)