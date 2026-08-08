"""
Ingestion Script:
This script handles the ETL (Extract, Transform, Load) process for our RAG system.
It loads text data, splits it into manageable chunks, clears the existing 
Pinecone index, and uploads the new document vectors.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Load environment variables (API keys)
load_dotenv()

# Initialize Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Step 1: Data Extraction & Transformation ---
print("Loading and splitting data...")
loader = TextLoader(r"D:\LangChain_Course\RAG_WITH_LANGCHAIN\mediumblog.txt")
document = loader.load()

# Splitting text into chunks to optimize retrieval quality
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(document)

# --- Step 2: Pinecone Vector Database Management ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["INDEX_NAME"])

# Clear index to avoid data duplication
print("Deleting existing data from index...")
index.delete(delete_all=True)

# --- Step 3: Loading into Vector Store ---
print("Ingesting new data...")
PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
print("Data successfully uploaded to Pinecone!")