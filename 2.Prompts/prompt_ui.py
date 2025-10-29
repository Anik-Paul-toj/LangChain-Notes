from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import load_prompt
import os

load_dotenv()

# Get API key
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN", "your_huggingface_token_here")

# Streamlit UI
st.header("🧠 Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

if st.button("Generate Summary"):
    with st.spinner("Generating summary..."):
        try:
            # Create LLM instance
            llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.7,
                max_new_tokens=512,
                huggingfacehub_api_token=api_key
            )
            
            # Wrap for chat functionality
            chat_llm = ChatHuggingFace(llm=llm)
            
            # Load template and format prompt            
            template = load_prompt("template.json")
            
            # Create chain and invoke            
            chain = template | chat_llm
            result = chain.invoke({
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input
            })
            
            
            # prompt = template.format(
            #     paper_input=paper_input,
            #     style_input=style_input,
            #     length_input=length_input
            # )
            
            # # Generate response
            # result = chat_llm.invoke(prompt)
            
            st.subheader("📝 Generated Summary:")
            st.write(result.content)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")