import re
import inspect
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import traceable

# Load environment variables (GROQ_API_KEY, LANGCHAIN_API_KEY)
load_dotenv()

# --- Configuration ---
MODEL = "llama-3.3-70b-versatile"
llm = ChatGroq(temperature=0, model_name=MODEL)
MAX_ITERATIONS = 10

# --- Tools Definition ---
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product. Input must be just the product name."""
    print(f"    >> [Tool] get_product_price executed for: {product}")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product.lower().strip(), 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount. Input format: price, tier (e.g., 1299.99, gold)"""
    print(f"    >> [Tool] apply_discount executed: {price}, {discount_tier}")
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier.lower().strip(), 0)
    return round(price * (1 - discount / 100), 2)

# Registry for easy tool access by name
tools = {"get_product_price": get_product_price, "apply_discount": apply_discount}

# --- Prompt Engineering ---
# This defines the "Brain" of the Agent.
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
@traceable(name="Groq Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question)
    scratchpad = ""  # This stores the "history" of the Agent's thoughts and results

    for i in range(1, MAX_ITERATIONS + 1):
        # Combine prompt and memory
        full_prompt = prompt + scratchpad
        
        # Invoke LLM
        response = llm.invoke(full_prompt)
        output = response.content
        print(f"\n--- Iteration {i} ---\n{output}")

        # Check for Final Answer
        if "Final Answer:" in output:
            print(f"\nResult: {output.split('Final Answer:')[-1].strip()}")
            return

        # Regex parsing to extract Action and Input from plain text
        action_match = re.search(r"Action:\s*(\w+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if action_match and action_input_match:
            tool_name = action_match.group(1).strip()
            tool_input_raw = action_input_match.group(1).strip()
            
            # Prepare arguments for the Python function
            args = [x.strip().strip("'\"") for x in tool_input_raw.split(",")]
            
            # Execute the tool and capture result
            if tool_name in tools:
                observation = str(tools[tool_name](*args))
            else:
                observation = "Error: Tool not found."
            
            # Update scratchpad with new info
            scratchpad += f"{output}\nObservation: {observation}\nThought:"
        else:
            scratchpad += f"{output}\nObservation: Error in format.\nThought:"

# --- Execution ---
if __name__ == "__main__":
    run_agent("What is the price of a laptop after applying a gold discount?")