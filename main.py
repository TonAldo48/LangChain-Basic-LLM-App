import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
  os.getenv["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# messages = [
#   SystemMessage("Translate the following text from English to French"),
#   HumanMessage("Hi!")
# ]

# print(model.invoke(messages))

# print(model.invoke("Hello"))

for token in model.stream("Hello"):
   print(token.content, end="|")