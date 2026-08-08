import warnings
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent 
from tavily import TavilyClient

# --- Configuration & Setup ---
# Ignore specific warnings to keep terminal output clean
warnings.filterwarnings("ignore", category=UserWarning, module="langgraph")

# Load environment variables
load_dotenv()

# Verify essential API keys are present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from environment variables or .env file.")
if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing from environment variables or .env file.")

# --- Tool Definition ---
@tool
def search(query: str) -> str:
    """Useful for searching the internet for job postings, current information, or technical details."""
    try:
        tavily_client = TavilyClient()
        results = tavily_client.search(query=query)
        return str(results)
    except Exception as e:
        return f"An error occurred during the search: {str(e)}"

def initialize_agent():
    """Initializes and returns the LangGraph ReAct Agent with Groq LLM."""
    tools = [search]
    # Using a high-performance open model via Groq
    llm = ChatGroq(temperature=0.0, model="llama-3.3-70b-versatile")
    return create_react_agent(llm, tools)

# --- Main Execution ---
def main() -> None:
    print("🤖 Initializing AI Agent...")
    agent = initialize_agent()
    
    # Define the user query
    user_query = "Search for 3 job postings for an AI engineer using LangChain in the Bay Area on LinkedIn."
    print(f"\nUser Query: {user_query}\n")
    print("⏳ Searching and processing, please wait...\n")
    
    try:
        # Invoke the agent
        response = agent.invoke({
            "messages": [HumanMessage(content=user_query)]
        })
        
        # Extract the final answer from the agent's message history
        final_answer = response["messages"][-1].content
        
        print("=" * 50)
        print("--- Agent Response ---")
        print("=" * 50)
        print(final_answer)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ An error occurred while running the agent: {str(e)}")

if __name__ == "__main__":
    main()