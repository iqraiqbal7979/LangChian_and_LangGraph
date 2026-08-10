import os
import re
import warnings
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable

# --- Configuration & Setup ---
warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables (GROQ_API_KEY, LANGSMITH_API_KEY)
load_dotenv()

# Verify essential API keys are present
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from environment variables or .env file.")

MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(temperature=0, model_name=MODEL)
MAX_ITERATIONS = 10

# --- Tools Definition ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product. Input must be just the product name."""
    print(f"    >> [Tool] get_product_price executed for: {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower().strip(), 0.0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount. Input format: price, tier (e.g., 1299.99, gold)"""
    print(f"    >> [Tool] apply_discount executed: {price}, {discount_tier}")
    try:
        price_val = float(price)
    except (ValueError, TypeError):
        price_val = 0.0
        
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(str(discount_tier).lower().strip(), 0)
    return round(price_val * (1 - discount / 100), 2)

# Registry for easy tool access by name
tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

# --- Prompt Engineering ---
react_prompt = """
You are a helpful AI Agent. Use the following tools to answer the question.
Available Tools:
- get_product_price(product: str)
- apply_discount(price: float, discount_tier: str)

STRICT RULES:
1. Use tools for data lookup and calculations. Do not guess.
2. After an Action, you will receive an Observation. Use it to proceed.
3. Action Input must NOT contain function names, just arguments.

Format:
Thought: think about what to do
Action: [tool_name]
Action Input: [comma separated args]
Observation: [tool result]
Final Answer: [your answer]

Question: {question}
Thought:"""

# --- Agentic Loop (The Reasoning Engine) ---
@traceable(name="Groq Custom Agent Loop")
def run_agent(question: str) -> str | None:
    """
    Executes a custom prompt-driven ReAct loop using regular expressions for parsing.
    """
    prompt = react_prompt.format(question=question)
    scratchpad = ""  # Stores the history of thoughts and observations

    print(f"Question: {question}")
    print("=" * 60)

    for i in range(1, MAX_ITERATIONS + 1):
        try:
            # Combine prompt and memory
            full_prompt = prompt + scratchpad
            
            # Invoke LLM
            response = llm.invoke(full_prompt)
            output = response.content
            print(f"\n--- Iteration {i} ---\n{output}")

            # Check for Final Answer
            if "Final Answer:" in output:
                final_result = output.split("Final Answer:")[-1].strip()
                print(f"\nResult: {final_result}")
                print("=" * 60)
                return final_result

            # Regex parsing to extract Action and Input from plain text
            action_match = re.search(r"Action:\s*(\w+)", output)
            action_input_match = re.search(r"Action Input:\s*(.+)", output)

            if action_match and action_input_match:
                tool_name = action_match.group(1).strip()
                tool_input_raw = action_input_match.group(1).strip()
                
                # Prepare arguments dynamically
                args = [x.strip().strip("'\"") for x in tool_input_raw.split(",")]
                
                # Execute the tool safely and capture result
                if tool_name in tools:
                    try:
                        observation = str(tools[tool_name](*args))
                    except Exception as tool_err:
                        observation = f"Error executing tool {tool_name}: {str(tool_err)}"
                else:
                    observation = f"Error: Tool '{tool_name}' not found."
                
                print(f"    [Observation] {observation}")
                
                # Update scratchpad with new info
                scratchpad += f"{output}\nObservation: {observation}\nThought:"
            else:
                scratchpad += f"{output}\nObservation: Error in format. Please follow Thought/Action/Action Input format.\nThought:"
                
        except Exception as e:
            print(f"❌ An error occurred during iteration {i}: {str(e)}")
            break

    print("ERROR: Max iterations reached without a final answer")
    return None

# --- Execution ---
if __name__ == "__main__":
    print("🤖 Initializing Custom Prompt-Based Agent...")
    print()
    run_agent("What is the price of a laptop after applying a gold discount?")