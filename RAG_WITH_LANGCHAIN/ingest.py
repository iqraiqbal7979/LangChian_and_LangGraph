import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables (API keys)
load_dotenv()

# Verify essential environment variables are present
required_env_vars = ["PINECONE_API_KEY", "INDEX_NAME"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing environment variable: {var}")

def run_ingestion():
    """
    Handles the ETL process for the RAG system: 
    Loads data, splits into chunks, clears Pinecone index, and ingests vectors.
    """
    try:
        # Initialize Embedding Model
        print("🤖 Initializing HuggingFace Embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # --- Step 1: Data Extraction & Transformation ---
        file_path = r"D:\LangChain_Course\RAG_WITH_LANGCHAIN\mediumblog.txt"
        print(f"📂 Loading document from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find the document at {file_path}")
            
        loader = TextLoader(file_path, encoding="utf-8")
        document = loader.load()

        # Splitting text into chunks to optimize retrieval quality
        print("✂️ Splitting text into chunks...")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        print(f"    Total chunks created: {len(texts)}")

        # --- Step 2: Pinecone Vector Database Management ---
        print("🔌 Connecting to Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("INDEX_NAME")
        index = pc.Index(index_name)

        # Clear index to avoid data duplication
        print("🗑️ Clearing existing data from Pinecone index...")
        index.delete(delete_all=True)

        # --- Step 3: Loading into Vector Store ---
        print("⬆️ Ingesting new vectors into Pinecone...")
        PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
        print("✅ Data successfully uploaded to Pinecone!")

    except Exception as e:
        print(f"❌ An error occurred during ingestion: {str(e)}")

if __name__ == "__main__":
    run_ingestion()