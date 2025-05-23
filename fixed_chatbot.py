import asyncio
import getpass
import os
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain.chat_models import init_chat_model

# Set up your OpenAI API key
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize the model
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define the state schema
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

# Create the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the call_model function
async def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

# Create the workflow
workflow = StateGraph(state_schema=State)

# Add the node and edges
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

# Compile the app
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

async def main():
    # Test the chatbot
    query = "Hi! I'm David"
    language = "Akan"
    
    input_messages = [HumanMessage(query)]
    config = {"configurable": {"thread_id": "test123"}}
    
    print("Testing the chatbot...")
    print(f"Query: {query}")
    print(f"Language: {language}")
    print("-" * 50)
    
    # First interaction
    output = await app.ainvoke(
        {"messages": input_messages, "language": language},
        config,
    )
    
    print("Response:")
    print(output["messages"][-1].content)
    print("-" * 50)
    
    # Follow-up question to test memory
    follow_up_query = "What is my name?"
    follow_up_messages = [HumanMessage(follow_up_query)]
    
    print(f"Follow-up query: {follow_up_query}")
    
    output2 = await app.ainvoke(
        {"messages": follow_up_messages, "language": language},
        config,  # Same config to maintain conversation thread
    )
    
    print("Follow-up response:")
    print(output2["messages"][-1].content)
    print("-" * 50)
    
    # Test with different language
    spanish_query = "How are you today?"
    spanish_messages = [HumanMessage(spanish_query)]
    spanish_config = {"configurable": {"thread_id": "spanish123"}}
    
    print(f"Spanish test query: {spanish_query}")
    
    spanish_output = await app.ainvoke(
        {"messages": spanish_messages, "language": "Spanish"},
        spanish_config,
    )
    
    print("Spanish response:")
    print(spanish_output["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main()) 