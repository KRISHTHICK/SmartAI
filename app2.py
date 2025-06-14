# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO

# Load dummy fashion DB (normally you'd load from a CSV or DB)
@st.cache_data
def load_fashion_data():
    df = pd.read_json("fashion_items.json")  # contains image_url, name, link, embedding
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    return df

# Load CLIP model
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Extract embedding
def get_image_embedding(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings[0].cpu().numpy()

# Find top-k similar fashion items
def get_top_k_similar_items(query_embedding, df, k=5):
    item_embeddings = np.vstack(df["embedding"].values)
    scores = cosine_similarity([query_embedding], item_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return df.iloc[top_indices].copy(), scores[top_indices]

# Streamlit UI
st.set_page_config(page_title="SmartAI - Celebrity Fashion Recommender")
st.title("👗 Fashion Recommendation from Celebrity Outfits")
st.write("Upload a celebrity outfit photo. We’ll find similar outfits you can buy!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Celebrity Look", use_column_width=True)

    with st.spinner("Analyzing fashion features..."):
        df_fashion = load_fashion_data()
        model, processor = load_clip_model()
        query_embed = get_image_embedding(image, model, processor)
        results_df, scores = get_top_k_similar_items(query_embed, df_fashion, k=5)

    st.subheader("🛍️ Similar Outfits You Can Buy")
    for idx, row in results_df.iterrows():
        st.markdown(f"### [{row['name']}]({row['buy_link']})")
        try:
            response = requests.get(row["image_url"])
            product_image = Image.open(BytesIO(response.content))
            st.image(product_image, width=250)
        except:
            st.image("https://via.placeholder.com/250", caption="Image not found", width=250)
        st.markdown(f"[🛒 Buy Now]({row['buy_link']})")
        st.markdown("---")
