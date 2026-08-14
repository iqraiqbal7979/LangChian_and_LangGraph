from typing import Any, Dict
from graph.state import GraphState
from graph.chains.generation import generation_chain

def generate(state: GraphState):
    
    question = state["question"]
    documents = state["documents"]
    
    
    generation = generation_chain.invoke({
        "context": documents,
        "question": question
    })
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation  
    }