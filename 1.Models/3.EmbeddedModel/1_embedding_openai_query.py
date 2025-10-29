from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002",dimensions=32)

result = embedding.embed_query("Hello world")
print("\nEmbedding Result:\n", str(result))