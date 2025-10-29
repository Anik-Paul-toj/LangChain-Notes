from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token_here")

# Create LLM instance
llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7,
                max_new_tokens=100,
                huggingfacehub_api_token=api_key
            )
            
# Wrap for chat functionality
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]

while(True):
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting chat. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("Bot:", result.content)
print(chat_history)


