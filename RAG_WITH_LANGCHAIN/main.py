"""
Main RAG Application:
This script implements a Retrieval-Augmented Generation (RAG) pipeline using LCEL.
It connects to a Pinecone vector store and uses a Groq-hosted Llama model 
to provide context-aware answers.
"""

import os
from dotenv import load_dotenv
from operator import itemgetter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Initialize Models
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

# Setup Vector Store Retriever
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
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
# We use itemgetter to extract the 'question' from the input dictionary.
# This ensures a clean data flow through the pipe (|).
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question")
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    query = "What is LANGCHAIN?"
    print(f"Asking: {query}\n")
    
    # Execute the chain
    answer = rag_chain.invoke({"question": query})
    print(f"--- Answer ---\n{answer}")