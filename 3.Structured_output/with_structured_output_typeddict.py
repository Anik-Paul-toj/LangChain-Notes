# ================================================
# Gemini Structured Output Example
# Author: Anik (using Gemini 2.5 Pro)
# ================================================

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
import os
import google.generativeai as genai

# ================================================
# Load Environment Variables
# ================================================
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in your .env file!")

# Configure Google Generative AI SDK (for model listing)
genai.configure(api_key=api_key)


# Uncomment if you want to see all models
# list_models()

# ================================================
# Create Gemini Chat Model
# ================================================

model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro",  # ✅ Latest working model
        temperature=0.7,
        google_api_key=api_key
)


# ================================================
# Define TypedDict Schema for Structured Output
# ================================================
##### Define schema for structured output ##### 
# ## Simple TypedDict without annotations 
# # class Review(TypedDict): 
# # summary: str 
# # sentiment: str 
# ## TypedDict with field annotations
class Review(TypedDict):
    key_themes: Annotated[list[str], "List all key themes discussed in the review."]
    summary: Annotated[str, "A concise summary of the review."]
    sentiment: Annotated[Literal["pos", "neg"], "Overall sentiment of the review."]
    pros: Annotated[Optional[list[str]], "List of pros mentioned."]
    cons: Annotated[Optional[list[str]], "List of cons mentioned."]

# Wrap model for structured output
structured_model = model.with_structured_output(Review)

# ================================================
# Input Text for Analysis
# ================================================
review_text = """
The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. 
Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. 
The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. 
Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful

Cons:
- Bulky and heavy—not great for one-handed use
- Bloatware still exists in One UI
- Expensive compared to competitors
"""

# ================================================
# Generate Structured Output
# ================================================
result = structured_model.invoke(review_text)

# ================================================
# Display the Result
# ================================================
print("\n✅ Structured Output:\n")
for k, v in result.items():
    print(f"{k}: {v}")
