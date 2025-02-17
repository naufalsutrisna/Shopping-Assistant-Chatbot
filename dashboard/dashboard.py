import streamlit as st
import pandas as pd
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai

# Load the dataset
df = pd.read_csv(os.path.join(os.getcwd(),"data", "Amazon-Products-Cleaned.csv"))

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the FAISS index
index = faiss.read_index(os.path.join(os.getcwd(), "model", "faiss_index.index"))

# OpenAI client
client = openai.OpenAI(api_key="",
                       base_url="https://openrouter.ai/api/v1")

def retrieve_products(query, budget=None, top_k=5):
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, top_k * 2)  # Retrieve more for filtering
    retrieved_df = df.iloc[indices[0]]

    if budget:
        retrieved_df = retrieved_df[retrieved_df["discount_price"] <= budget]

    if re.search(r'\b(cheapest|lowest price|budget-friendly|affordable)\b', query, re.IGNORECASE):
        retrieved_df = retrieved_df.sort_values(by=["discount_price", "ratings", "no_of_ratings"], ascending=[True, False, False])
    else:
        retrieved_df = retrieved_df.sort_values(by=["ratings", "no_of_ratings"], ascending=[False, False])

    return retrieved_df.head(top_k)

def generate_response(query, retrieved_products):
    product_details = "\n".join([
        f"{row['name']} ({row['main_category']} - {row['sub_category']}): "
        f"Ratings: {row['ratings']} ({row['no_of_ratings']} ratings), "
        f"Discount Price: {row['discount_price']}, Actual Price: {row['actual_price']}"
        for _, row in retrieved_products.iterrows()
    ])

    prompt = f"""You are a shopping assistant. Recommend products based on the user's query.
Query: {query}
Products:
{product_details}
Response:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

def extract_budget(query):
    match = re.search(r'\$?(\d{1,6})', query)
    return int(match.group(1)) if match else None

def shopping_assistant(query):
    budget = extract_budget(query)
    retrieved_products = retrieve_products(query, budget)
    response = generate_response(query, retrieved_products)
    return retrieved_products, response

st.title("AI Shopping Assistant")
query = st.text_input("Enter your shopping query:")

if st.button("Find Products"):
    if query:
        retrieved_products, response = shopping_assistant(query)
        st.subheader("Recommended Products")
        st.dataframe(retrieved_products)
        st.subheader("Shopping Assistant's Response")
        st.write(response)
