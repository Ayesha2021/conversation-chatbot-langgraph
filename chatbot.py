from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated
from langchain_groq import ChatGroq
import os
# in google colab use user data from google.colab import userdata
from dotenv import load_dotenv

# load environment

load_dotenv()

# Loading API key from .env
# groq key for llm
groq_api_key = os.getenv("GROQ_API_KEY")

# Langsmith used for Auditing
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# setting Environment variable

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "TrainingLanggraph"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

# Initializing llm

llm = ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-120b")

# Building Chatbot using Langgraph


class State(TypedDict):
    # add_message functin in the annoation defines how this state key should be updated
    # in this case ,it appends messages to list ,rather than overriting them .
    # for every add message is changing and messages contains appended message response from llm
    messages: Annotated[list, add_messages]
# build graphbuilder


graph_builder = StateGraph(State)

# START---------->Chatbot-------->END   ,chatbot interacting with LLM


def chatbot(state: State):
    # messages inside state['messages'] is user query send to llm
    return {"messages": llm.invoke(state['messages'])}
    # whenever return "messages" call happens it trigger add_message to messages list of type typedDict


# add chatbot node to graph builder
# first param is node name and second name of function .
graph_builder.add_node("chatbot", chatbot)

# connect chatbot to START and END node
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# compile graph build to get graph
graph = graph_builder.compile()

# display graph

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass

# Chat loop which exit for four key words quit q exit byte
while True:
    user_input = input("User: ")
    if user_input in ["quit", "q", "exit", "bye"]:
        print("Good Bye")
        break
    for event in graph.stream({'messages': ("user", user_input)}):
        # print(event.values())
        for value in event.values():
            messages = value.get('messages', [])
            if not isinstance(messages, list):
                messages = [messages]
            for message in messages:
                if hasattr(message, 'content') and message.content:
                    print("Assistant:", message.content)
