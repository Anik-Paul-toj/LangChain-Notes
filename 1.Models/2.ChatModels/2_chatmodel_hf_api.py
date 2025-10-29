from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_KEY")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=api_token,
    max_new_tokens=512,
    temperature=0.7,
    top_k=50,
)

prompt = "Explain the theory of relativity in simple terms."
response = llm.invoke(prompt)
print("\nResponse:\n", response)