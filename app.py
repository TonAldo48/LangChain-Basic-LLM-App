import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print("OpenAI API key loaded successfully!")
else:
    print("Failed to load OpenAI API key. Make sure you've added your key to the .env file.") 