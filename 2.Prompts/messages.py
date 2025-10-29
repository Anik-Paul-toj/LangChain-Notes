from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token_here")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,    
    max_new_tokens=100,
    huggingfacehub_api_token=api_key
)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful research assistant."),
    HumanMessage(content="Tell me about Langchain"),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)