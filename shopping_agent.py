import os
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables (API keys) from .env file
load_dotenv()

# Verify essential API keys are present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from environment variables or .env file.")

MAX_ITERATIONS = 10
MODEL = "llama-3.3-70b-versatile"

# --- Tools Definition ---

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog. 
    Args:
        product (str): Name of the product (e.g., laptop, mobile, keyboard, headphone).
    """
    print(f" >> Executing get_product_price(product='{product}')")
    
    # Product catalog database
    prices = {
        'laptop': 1299.99,
        'mobile': 2001.98, 
        'keyboard': 1000.99, 
        'headphone': 499.99 
    }
    
    # Returns price or 0 if product not found
    price = prices.get(product.lower(), 0.0)
    return price
    
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    Args:
        price (float): The original price of the product.
        discount_tier (str): The discount category (bronze, silver, or gold).
    """
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower(), 0)
    
    return round(price * (1 - discount / 100), 2)


# --- Agent Core Logic ---

@traceable(name="LangChain Agent Loop")
def run_agent(question: str) -> str | None:
    """
    Main agent function that implements a ReAct (Reasoning + Acting) loop.
    It iteratively decides which tool to call until a final answer is reached.
    """
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    
    # Initialize LLM with tool-binding capability
    llm = ChatGroq(temperature=0, model=MODEL)
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"Question: {question}")
    print("=" * 60)
    
    # Setting up the conversation context
    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES:\n"
                "1. NEVER guess prices. Call get_product_price first.\n"
                "2. Call apply_discount ONLY after getting a price.\n"
                "3. Use the apply_discount tool for calculations.\n"
                "4. If discount tier is missing, ask the user."
            )
        ),
        HumanMessage(content=question),
    ]
    
    # Main ReAct loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        try:
            # LLM decides whether to call a tool or provide a final response
            ai_message = llm_with_tools.invoke(messages)
            tool_calls = ai_message.tool_calls

            # If no tools are called, the agent is finished
            if not tool_calls:
                print(f"\nFinal Answer: {ai_message.content}")
                return ai_message.content

            # Handle tool execution
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

            # Execute the tool and capture the observation
            tool_to_use = tools_dict.get(tool_name)
            if tool_to_use is None:
                raise ValueError(f"Tool '{tool_name}' not found")

            observation = tool_to_use.invoke(tool_args)
            print(f"  [Tool Result] {observation}")

            # Update history with the AI's call and the tool's result
            messages.append(ai_message)
            messages.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )
            
        except Exception as e:
            print(f"❌ An error occurred during iteration {iteration}: {str(e)}")
            break

    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("🤖 Initializing LangChain Agent (.bind_tools)...")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")