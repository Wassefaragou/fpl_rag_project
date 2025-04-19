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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
You are a Fantasy Premier League (FPL) expert assistant.

Use the following context from the FPL database to help answer the user's question:
{context}

User's Question:
{question}

Instructions for your response:
- While generating the answer look in your ganeral knowledge and see if what in database is correct or not.
- If relevant information is available from the context, cite specific player stats and facts.
- If no direct context is available, answer based on your general FPL knowledge and clearly mention that recent FPL data for this point is unavailable.
- Be concise, accurate, and friendly.
- Whenever appropriate, suggest actionable advice (e.g., transfer tips, captaincy picks, wildcard usage).
- Structure your answers clearly for easy reading.
- Always keep the tone professional, knowledgeable, and approachable.

Begin your answer:
"""
        prompt = ChatPromptTemplate.from_template(template)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True,k=2)

    # Build Conversational Retrieval Chain
        self.chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
        },
        return_source_documents=False,   
        verbose=False)
    
        print("Retrieval chain set up successfully")
    
    def query(self, question):
        """Query the RAG system with a question"""
        if not self.vector_store:
            print("Vector store not initialized. Please load or create one first.")
            return None
        
        if not hasattr(self, 'chain'):
            self.setup_retrieval_chain()
        result = self.chain.invoke({"question": question})
        return result["answer"]
