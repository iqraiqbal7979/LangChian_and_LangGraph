import warnings
# --- Configuration & Setup ---
# Ignore LangGraph deprecation warnings to keep the terminal output clean
warnings.filterwarnings("ignore", category=UserWarning, module="langgraph")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent 
from tavily import TavilyClient

# Load environment variables (API Keys: GROQ_API_KEY, TAVILY_API_KEY)
load_dotenv()

# Initialize Clients
tavily = TavilyClient()

# --- Tool Definition ---
@tool
def search(query: str) -> str:
    """Useful for searching the internet for job postings or information."""
    results = tavily.search(query=query)
    return str(results)

# Define tools list for the Agent
tools = [search]
llm = ChatGroq(temperature=0, model="llama-3.3-70b-versatile")

# Initialize the ReAct Agent
agent = create_react_agent(llm, tools)

# --- Main Execution ---
def main():
    print("Hello from langchain-course!")
    # Define the user query and Invoke the agent
    response = agent.invoke({
        "messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin")]
    })
    
    # Extract and print the final answer from the agent's message history
    final_answer = response["messages"][-1].content
    print("\n--- Agent Response ---")
    print(final_answer)
    
if __name__ == "__main__":
    main()