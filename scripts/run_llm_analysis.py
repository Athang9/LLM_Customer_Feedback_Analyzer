import os

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")

# Load data
df = pd.read_csv("../data/cleaned_reviews.csv")
reviews = df["Text"].tolist()

# Combine into one big chunk (you can use batches for big data)
review_text = "\n\n".join(reviews[:100])  # First 100 reviews

# Define prompt
template = """
You are a business analyst. Analyze the following customer reviews and:
1. List top 3 complaints
2. List top 3 things customers praise
3. Recommend 2 product or service improvements

Reviews:
{reviews}

Your response:
"""

prompt = PromptTemplate(input_variables=["reviews"], template=template)

llm = ChatOpenAI(
    temperature=0.3,
    model_name="mistralai/mistral-7b-instruct",  # or "gpt-3.5-turbo" if available
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run
response = chain.run(reviews=review_text)

# Save output
with open("analysis_summary.txt", "w", encoding="utf-8") as f:
    f.write(response)

print("Analysis saved to analysis_summary.txt")
