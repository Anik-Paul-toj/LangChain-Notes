from langchain import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

chat_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)  # model making use of the OPENAI_API_KEY from .env


# take string as an input and return response as chat message
result = chat_model.invoke("Explain the theory of relativity in simple terms.")  # Example prompt to the Chat Model

print(result.content)