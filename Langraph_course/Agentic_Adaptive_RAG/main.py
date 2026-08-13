from dotenv import load_dotenv
from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("--- Starting Advanced Adaptive RAG Agent ---")
    
    # Invoke the LangGraph application
    response = app.invoke(input={"question": "what is agent memory?"})
    print(response)