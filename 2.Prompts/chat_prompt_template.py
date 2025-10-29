from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
    
    # this work its wired but works    
    ('system', "You are a helpful {domain} expert."),
    ('human', "Explain in simple terms, what is {topic}?"),
    
    
    # this dont work in dynamic \
    # SystemMessage(content="You are a helpful {domain} expert."),
    # HumanMessage(content = "Explain in simple terms, what is {topic}?"),
])

prompt = chat_template.invoke({
    "domain": "artificial intelligence",
    "topic": "machine learning"
}) 

print(prompt)