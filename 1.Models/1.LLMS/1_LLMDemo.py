from  langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.7) # model making use of the OPENAI_API_KEY from .env

# take string as an input and return response in string format
result = llm.invoke("Explain the theory of relativity in simple terms.") # Example prompt to the LLM

print(result)