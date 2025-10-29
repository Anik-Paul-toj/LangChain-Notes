from langchain_huggingface import HuggingFaceEmbeddings
import os

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Example 1: Embedding a single query text
single_text = "What is artificial intelligence?"
single_result = embedding.embed_query(single_text)
print(f"Single text: '{single_text}'")
print(f"Embedding dimension: {len(single_result)}")
print(f"First 10 values: {single_result[:10]}")

print("\n" + "="*60 + "\n")

# Example 2: Embedding multiple documents
documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables developers to build robust and scalable AI applications with ease.",
]

# For multiple documents, use embed_documents
result = embedding.embed_documents(documents)
print(f"\nNumber of documents: {len(documents)}")
print(f"Number of embeddings: {len(result)}")
print(f"Embedding dimension: {len(result[0])}")
print("\nFirst document embedding (first 10 values):")
print(result[0][:10])
print("\nSecond document embedding (first 10 values):")
print(result[1][:10])