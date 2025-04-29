# ⚽ FPL RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions about Fantasy Premier League (FPL) using live player data, OpenAI's GPT models, and Gradio for a web UI.

---

## 🚧 Work in Progress

This project is still under active development. Expect frequent changes and improvements.

---

## 📥 Clone the Repository

```bash
git clone https://github.com/Wassefaragou/fpl-rag-project.git
cd fpl-rag-assistant
```
---

## 🧠 What it Does

- **Scrapes live FPL data** from the official FPL API
- **Prepares natural language documents** for each player
- **Generates embeddings** and builds a FAISS vector store
- **Answers your FPL questions** using a custom prompt and GPT-based retrieval chain

---

## 🚀 Quickstart

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Set your OpenAI API key**

Create a `.env` file or set the environment variable:

```
OPENAI_API_KEY=your-key-here
```

3. **Run the app**

You can choose between:

- **CLI interface**:
  ```bash
  python run.py
  ```
- **Web UI (Gradio)**:
  ```bash
  python run.py
  # then type `web` when prompted
  ```

---

## 🛠️ Project Structure

- `run.py`: Entry point for CLI or Gradio UI.
- `app.py`: Alternative Gradio-only interface.
- `fpl_data_collection.py`: Scrapes and structures FPL data.
- `fpl_rag_system.py`: RAG pipeline setup (vector store, retriever, LLM chain).
- `fpl_documents.json`: Sample generated documents.
- `requirements.txt`: Dependencies.

---

## 📦 Features

- Uses **LangChain**, **FAISS**, **OpenAI GPT**, **Sentence-Transformers**
- Fast data retrieval using embeddings
- Modular, easy to customize
- Helpful for **FPL strategy, transfers, captain picks, budget analysis**

---

## 🧪 Example Queries

- "Who are the top-scoring defenders under £5.0M?"
- "Should I captain Haaland or Salah this week?"
- "Which midfielders have the best form over the last 5 matches?"

---

## 📋 Notes

- Be mindful of OpenAI API usage costs.
- `fpl_documents.json` and `fpl_vectorstore/` will be created on first run.
- The system pulls **live data** from `https://fantasy.premierleague.com/api/`.

---

## 📄 License

Use freely, but attribute if shared.
