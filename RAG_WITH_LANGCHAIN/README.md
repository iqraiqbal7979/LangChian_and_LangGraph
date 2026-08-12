# RAG Application with LangChain & Pinecone

This project is a **Retrieval-Augmented Generation (RAG)** system built using `LangChain`, `Pinecone Vector Database`, and the `Groq API` (Llama 3.3). It processes local text files to provide context-aware, accurate answers to your questions.

## 🚀 Features

* **Efficient Data Ingestion:** Automatically chunks and vectorizes your data for storage in Pinecone.
* **Professional RAG Pipeline:** Built with `LCEL` (LangChain Expression Language) for a clean, scalable, and modular architecture.
* **Smart Retrieval:** Uses semantic search via vector databases to find the most relevant context.
* **High Performance:** Leverages the Groq-hosted Llama 3.3 model for fast and intelligent responses.

## ⚙️ Setup & Configuration

### 1. Environment Variables

Create a `.env` file in the root directory and add your API keys:

```env
PINECONE_API_KEY=your_pinecone_api_key_here
INDEX_NAME=your_index_name_here
GROQ_API_KEY=your_groq_api_key_here

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

## 💻 How to Run

1. **Ingest Data:** (To load and upload your document to Pinecone)
```
python ingest.py

```


2. **Run Query:** (To ask questions and get AI-generated answers)
```
python main.py

```



## 🏗️ Project Structure

* `ingest.py`: Handles data loading, text splitting, and Pinecone ingestion.
* `main.py`: Contains the RAG pipeline and LLM response generation logic.
* `.env`: Stores sensitive API keys (ensure this is added to `.gitignore`).
