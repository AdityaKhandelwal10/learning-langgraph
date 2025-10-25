from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# Inject a state - check this later

document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# @tool
# def update(content: str) -> str:
#     """
#     Updates the document with provided content
#     """
#     global document_content
#     document_content += content
#     return "Document updated successfully"

@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


# @tool
# def save(filename: str) -> str:
#     """
#     Saves the document to a file anf finishes the process
#     """
#     try:
#         if not filename.endswith(".txt"):
#             filename += ".txt"
#         global document_content
#         with open(filename, "w") as f:
#                 f.write(document_content)
#     except Exception as e:
#         return f"Error saving document: {e}"
    
#     return "Document saved successfully"
@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"


    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"
    


tools = [update, save]

model = ChatOllama(model="llama3.1:latest").bind_tools(tools)

# def our_agent(state: AgentState) -> AgentState:
#     """
#     This agent is used to draft a document
#     """
#     system_message = SystemMessage(content=f"""
#         You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
#     - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
#     - If the user wants to save and finish, you need to use the 'save' tool.
#     - Make sure to always show the current document state after modifications.
    
#     The current document content is:{document_content}
#     """)

#     if not state["messages"]:
#         user_input = "I am here to help you write a document. What would you like to do?"
#         user_message = HumanMessage(content=user_input)
#         print(user_message)

#     else:
#         user_input = input("what would you like to do with this document?")
#         user_message = HumanMessage(content=user_input)
#         print(user_message)

#     all_messages = [system_message] + list(state["messages"]) + [user_message]
#     response = model.invoke(all_messages)
#     return {"messages" : list(state["messages"]) + [user_message, response]}

def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState):
    """
    This function is used to determine if the agent should continue or not.
    """

    messages = state["messages"]

    if not messages:
        return "continue"
    
    # This looks for the most recent tool message call - We want to exit when the user has saved the document
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and "saved" in message.content.lower() and "document" in message.content.lower():
            return "end"
    return "continue"

def print_messages(messages):
    """Function I made to print the messages in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


graph = StateGraph(AgentState)

graph.add_node("our_agent", our_agent)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("our_agent")

graph.add_edge("our_agent", "tools")

graph.add_conditional_edges(
    "tools", 
    should_continue, 
    {
        "end": END,
        "continue": "our_agent"
    }
)

app = graph.compile()

def run_document_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

if __name__ == "__main__":
    run_document_agent()