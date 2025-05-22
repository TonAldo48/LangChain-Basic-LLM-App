import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
  os.getenv["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

model = init_chat_model("gpt-4o-mini", model_provider="openai")

# messages = [
#   SystemMessage("Translate the following text from English to French"),
#   HumanMessage("Hi!")
# ]

# print(model.invoke(messages))

# print(model.invoke("Hello"))

# for token in model.stream("Hello"):
#    print(token.content, end="|")

system_template = "Translate the following text from English to {language}"
prompt_template = ChatPromptTemplate.from_messages([
  ("system", system_template),
  ("user", "{text}")
])

prompt = prompt_template.invoke({"language": "French", "text": "I love programming in Python"})

# print(prompt)

print(model.invoke(prompt).content)