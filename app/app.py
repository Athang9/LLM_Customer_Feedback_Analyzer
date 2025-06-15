import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from topic_modeling import get_topics

load_dotenv()

# Setup LLM
llm = ChatOpenAI(
    temperature=0.3,
    model_name="mistralai/mistral-7b-instruct",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
)

prompt = PromptTemplate(
    input_variables=["reviews"],
    template="""
You are an AI assistant. Analyze the customer reviews below and extract insights into three categories:

1. ✅ **What Customers Loved**  
2. ⚠️ **What Customers Complained About**  
3. 💡 **What Customers Suggested or Recommended**

---  
Customer Reviews:
{reviews}

---

Provide your response in this structure:
**What Customers Loved:**  
- Point 1  
- Point 2  
...

**What Customers Complained About:**  
- Point 1  
...

**What Customers Suggested or Recommended:**  
- Point 1  
...
""",
)

# Initialize session variable
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# Streamlit UI
st.set_page_config(page_title="LLM Feedback Analyzer", layout="centered")
st.title("🛍️ Customer Feedback Analyzer (LLM + Clustering)")
st.markdown("Upload a CSV file containing product reviews.")

uploaded_file = st.file_uploader("📤 Upload your CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # ✅ Detect review column
    text_col = None
    for col in df.columns:
        if col.strip().lower() in [
            "text",
            "review",
            "reviewtext",
            "feedback",
            "comments",
        ]:
            text_col = col
            break

    if not text_col:
        st.error(
            "❌ Couldn't find a column with review text. Please include one named 'Text', 'Review', etc."
        )
    else:
        df = df[[text_col]].dropna()
        df.columns = ["Text"]

        st.success("✅ Reviews loaded successfully!")

        # Slider to pick how many reviews to analyze
        num_reviews = st.slider(
            "📊 How many reviews to analyze?", 10, min(300, len(df)), 50
        )

        # 🔍 Summarize with AI
        if st.button("🔍 Analyze with AI"):
            with st.spinner("Summarizing reviews..."):
                reviews = df["Text"].dropna().unique().tolist()
                text_blob = "\n\n".join(reviews[:num_reviews])

                chain = LLMChain(llm=llm, prompt=prompt)
                summary = chain.run(reviews=text_blob)

                st.subheader("🧾 AI Summary:")
                st.markdown(summary)

                st.session_state["analysis_done"] = True

        # 📊 Topic Clustering
        if st.session_state.get("analysis_done") and st.checkbox(
            "📊 Show Thematic Clusters"
        ):
            with st.spinner("Clustering topics..."):
                reviews_subset = [
                    x[:200] for x in df["Text"].dropna().tolist()[:300]
                ]
                topics = get_topics(
                    reviews_subset, num_clusters=5, top_n_words=5
                )

                st.subheader("🧠 Key Topics Identified:")
                for cluster, keywords in topics.items():
                    st.markdown(f"**{cluster}**: {', '.join(keywords)}")
