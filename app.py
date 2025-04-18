import gradio as gr
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

def setup_api_key():
    """Set up the OpenAI API key securely"""
    # Check if the API key is in environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If not found, prompt the user
    if not api_key:
        print("OpenAI API key not found in environment variables or .env file.")
        api_key = input("Please enter your OpenAI API key: ")
        
        # Save to environment for current session
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Offer to save to .env file
        save_to_env = input("Save API key to .env file for future use? (y/n): ").lower()
        if save_to_env == 'y':
            with open('.env', 'a') as f:
                f.write(f"\nOPENAI_API_KEY={api_key}")
            print("API key saved to .env file.")
    
    return api_key

# Get API key securely
api_key = setup_api_key()

# Import the FPL RAG system
from fpl_rag_system import FPLRagSystem

# Initialize the FPL RAG system with secure API key
fpl_rag = FPLRagSystem(api_key)

# Load the vector store
try:
    fpl_rag.load_vector_store()
except:
    print("No existing vector store found. Please create one first.")
    print("Running data collection and vector store creation...")
    # Import the data collection module
    from fpl_data_collection import fetch_fpl_data, prepare_documents
    
    # Fetch and prepare the data
    players_df, player_details = fetch_fpl_data()
    documents = prepare_documents(players_df, player_details)
    
    # Create the vector store
    split_docs = fpl_rag.load_documents(documents)
    fpl_rag.create_vector_store(split_docs)
    fpl_rag.save_vector_store()

# Set up the chain
fpl_rag.setup_retrieval_chain()

def query_fpl_assistant(question):
    """Process a question through the FPL RAG system"""
    if not question:
        return "Please ask a question about Fantasy Premier League."
    
    try:
        response = fpl_rag.query(question)
        return response
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Create the Gradio interface
demo = gr.Interface(
    fn=query_fpl_assistant,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Ask me anything about Fantasy Premier League...",
        label="Your Question"
    ),
    outputs=gr.Textbox(
        lines=10,
        label="FPL Assistant Response"
    ),
    title="Fantasy Premier League Assistant",
    description="Ask questions about FPL players, teams, strategies, and more!",
    examples=[
        ["Who is the highest scoring midfielder this season?"],
        ["Which defenders under £5.0M have the best points per game?"],
        ["Should I captain Haaland or Salah this week?"],
        ["Who are the best budget forwards this season?"]
    ],
    theme="default"
)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False)  # Set share=False in production