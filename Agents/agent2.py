from typing import *
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


class AgentState(TypedDict):
    messages : List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3.1:latest")


def process_node(state : AgentState) -> AgentState:
    """
    Node to invoke an LLM with conversation history
    """
    response = llm.invoke(state['messages'])

    state['messages'].append(AIMessage(content=response.content))
    print(response.content)
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_node)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

conversation_history = []

user_input = input("Enter your message: ")

while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages" : conversation_history})
    conversation_history = result['messages']
    user_input = input("Enter: ")

