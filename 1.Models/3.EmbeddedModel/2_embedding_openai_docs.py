from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-ada-002",dimensions=32)


document =[
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build robust and scalable AI applications with ease.",    

]

result = embedding.embed_query(document)
print("\nEmbedding Result:\n", str(result))