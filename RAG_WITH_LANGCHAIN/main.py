import os
import warnings
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables
load_dotenv()

# Verify essential environment variables
required_env_vars = ["GROQ_API_KEY", "PINECONE_API_KEY", "INDEX_NAME"]
for var in required_env_vars:
    if not os.getenv(var):
        raise ValueError(f"Missing environment variable: {var}")

def run_rag_query(query: str) -> str | None:
    """
    Executes the Retrieval-Augmented Generation (RAG) pipeline 
    using Pinecone vector store and Groq LLM.
    """
    try:
        print("🤖 Initializing Models and Vector Store Retriever...")
        # Initialize Models
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

        # Setup Vector Store Retriever
        vectorstore = PineconeVectorStore(
            index_name=os.getenv("INDEX_NAME"), 
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        def format_docs(docs) -> str:
            """Helper function to concatenate retrieved document contents."""
            return "\n\n".join(doc.page_content for doc in docs)

        # Define Prompt Template
        prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            Question: {question}
            Provide a detailed answer:"""
        )

        # --- Define LCEL Retrieval Chain ---
        rag_chain = (
            {
                "context": itemgetter("question") | retriever | format_docs, 
                "question": itemgetter("question")
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        print(f"🔍 Asking: {query}\n")
        answer = rag_chain.invoke({"question": query})
        return answer

    except Exception as e:
        print(f"❌ An error occurred during RAG execution: {str(e)}")
        return None

if __name__ == "__main__":
    query = "What is LANGCHAIN?"
    result = run_rag_query(query)
    
    if result:
        print(f"--- Answer ---\n{result}")