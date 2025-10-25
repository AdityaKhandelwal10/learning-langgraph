from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

tools = [add]
model = ChatOllama(model="llama3.1:latest").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:

    system_message = SystemMessage(content="You are my AI assistant. You are helpful and friendly. Answer to the best of your ability.")
    response = model.invoke([system_message] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    """
    This function is used to determine if the agent should continue or not.

    """
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "end": END,
        "continue": "tools"
    }
)

graph.add_edge("tools", "our_agent")
graph.add_edge(START, "our_agent")

app = graph.compile()

agent_state = {"messages": [HumanMessage(content="What is 10 + 20?")]}
result = app.invoke(agent_state)
print(result["messages"][-1].content)
