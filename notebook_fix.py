# COPY THIS INTO A NEW JUPYTER NOTEBOOK CELL
# This is the corrected version that will work

# Cell 1: Fresh setup - create a new workflow from scratch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence

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

# Create a fresh workflow
new_workflow = StateGraph(state_schema=State)

# Add the node and edges
new_workflow.add_node("model", call_model)
new_workflow.add_edge(START, "model")
new_workflow.add_edge("model", END)

# Compile the app
memory = MemorySaver()
new_app = new_workflow.compile(checkpointer=memory)

print("Fresh workflow setup complete!")

# Cell 2: Test the working chatbot (in Jupyter, you can use await directly)
# Note: The following code should be run in separate cells in Jupyter notebook

async def test_chatbot():
    query = "Hi! I'm David"
    language = "Akan"
    
    input_messages = [HumanMessage(query)]
    config = {"configurable": {"thread_id": "test123"}}
    
    output = await new_app.ainvoke(
        {"messages": input_messages, "language": language},
        config,
    )
    
    print("Response:")
    output["messages"][-1].pretty_print()
    return output, config

# For Jupyter notebook, you can call this directly:
# await test_chatbot()

async def test_follow_up():
    # Get the config from the previous test
    _, config = await test_chatbot()
    
    follow_up_query = "What is my name?"
    follow_up_messages = [HumanMessage(follow_up_query)]
    
    output2 = await new_app.ainvoke(
        {"messages": follow_up_messages, "language": "Akan"},
        config,  # Same config to maintain conversation thread
    )
    
    print("Follow-up response:")
    output2["messages"][-1].pretty_print()

# For Jupyter notebook, you can call this directly:
# await test_follow_up()

async def test_different_language():
    spanish_query = "How are you today?"
    spanish_messages = [HumanMessage(spanish_query)]
    spanish_config = {"configurable": {"thread_id": "spanish123"}}
    
    spanish_output = await new_app.ainvoke(
        {"messages": spanish_messages, "language": "Spanish"},
        spanish_config,
    )
    
    print("Spanish response:")
    spanish_output["messages"][-1].pretty_print()

# For Jupyter notebook, you can call this directly:
# await test_different_language() 