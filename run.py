import os
import json
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

def setup_fpl_rag_system():
    """Set up the complete FPL RAG system"""
    # Import our modules
    from fpl_data_collection import fetch_fpl_data, prepare_documents
    from fpl_rag_system import FPLRagSystem
    
    # Get the API key securely
    api_key = setup_api_key()
    
    print("Setting up FPL RAG system...")
    
    # Initialize the RAG system
    fpl_rag = FPLRagSystem(api_key)
    
    # Check if we have existing data and vector store
    vector_store_path = Path("fpl_vectorstore")
    documents_path = Path("fpl_documents.json")
    
    if vector_store_path.exists() and documents_path.exists():
        print("Found existing vector store and documents.")
        update = input("Do you want to update the FPL data? (y/n): ").lower()
        
        if update == 'y':
            print("Fetching fresh FPL data...")
            run_data_collection()
    else:
        print("No existing data found. Running data collection...")
        run_data_collection()
    
    # Load documents and create/update vector store
    documents = fpl_rag.load_documents("fpl_documents.json")
    fpl_rag.create_vector_store(documents)
    fpl_rag.save_vector_store()
    
    # Set up the chain
    fpl_rag.setup_retrieval_chain()
    
    return fpl_rag

def run_data_collection():
    """Run the FPL data collection process"""
    from fpl_data_collection import fetch_fpl_data, prepare_documents
    
    print("Fetching FPL data...")
    players_df, player_details = fetch_fpl_data()
    
    print("Preparing documents...")
    documents = prepare_documents(players_df, player_details)
    
    # Save the data
    players_df.to_csv('fpl_players.csv', index=False)
    
    with open('player_details.json', 'w') as f:
        json.dump(player_details, f)
    
    with open('fpl_documents.json', 'w') as f:
        json.dump(documents, f)
    
    print(f"Collected data for {len(players_df)} players")
    print(f"Created {len(documents)} documents for the RAG system")

def run_cli_interface():
    """Run a simple CLI interface for the FPL RAG system"""
    fpl_rag = setup_fpl_rag_system()
    
    print("\n===== FPL Assistant CLI =====")
    print("Ask questions about Fantasy Premier League players, teams, and strategies.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\nYour question: ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query.strip():
            print("Please ask a question.")
            continue
        
        try:
            response = fpl_rag.query(query)
            print("\nFPL Assistant says:")
            print(response)
        except Exception as e:
            print(f"Error: {str(e)}")

def run_web_interface():
    """Run the web interface for the FPL RAG system"""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not found. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "gradio"])
        import gradio as gr
    
    fpl_rag = setup_fpl_rag_system()
    
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
    demo.launch(share=False)  # Set share=False in production

if __name__ == "__main__":
    print("Welcome to the FPL RAG System Setup")
    interface = input("Choose interface (cli/web): ").lower()
    
    if interface == "web":
        run_web_interface()
    else:
        run_cli_interface()