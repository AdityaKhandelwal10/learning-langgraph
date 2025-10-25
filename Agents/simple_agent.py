from typing import *
from langgraph.graph import StateGraph, START, END
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage




class AgentState(TypedDict):
    message : List[HumanMessage]


llm = OllamaLLM(model="llama3.1:latest")

def process_node(state : AgentState) -> AgentState:
    """
    Simple node that invokes an LLM
    """
    response = llm.invoke(state['messages'])
    print(response)

    return state


graph = StateGraph(AgentState)

graph.add_node("process", process_node)

graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


user_input = input("Enter your message: ")

while user_input != "exit":
    
    agent.invoke({"message" : [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")