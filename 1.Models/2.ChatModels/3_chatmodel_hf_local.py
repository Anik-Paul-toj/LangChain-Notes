from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os


os.environ["HF_HOME"] = r"D:\code\GenAI\Models\2.ChatModels" # Set this to your local HF models directory

llm = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.7,  
    }
)

model = ChatHuggingFace(llm=llm)

prompt = "Explain the theory of relativity in simple terms."
response = model.invoke(prompt)
print("\nResponse:\n", response.content)