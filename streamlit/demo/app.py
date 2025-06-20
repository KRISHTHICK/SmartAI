# app.py (SmartAI Full Version)
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import BytesIO
import time
import os

# Langchain + Ollama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# App Config
st.set_page_config(page_title="SmartAI - Fashion & Personalization", layout="wide")

# Session State Init
if "log" not in st.session_state:
    st.session_state.log = []

def log_status(msg):
    st.session_state.log.append(f"✅ {msg}")

def show_status():
    st.subheader("📝 System Status Log")
    for msg in st.session_state.log:
        st.markdown(msg)

# Tabs
page = st.sidebar.radio("📌 Navigate", ["🔐 Sign In", "⚙️ Settings", "👗 Virtual Try-On", "🌟 Celebrity Match", "✍️ Content Agent", "📈 Trend Analyzer", "📋 Status"])

# Model/API Options
if "model_mode" not in st.session_state:
    st.session_state.model_mode = "Local"

if page == "⚙️ Settings":
    st.header("⚙️ Configuration")
    st.session_state.model_mode = st.selectbox("Choose Mode", ["Local", "API"], index=0)
    st.success(f"You selected: {st.session_state.model_mode}")

# Sign In
if page == "🔐 Sign In":
    st.header("Sign In")
    user_email = st.text_input("📧 Email")
    phone = st.text_input("📱 Phone Number")
    if st.button("Continue"):
        st.success("Signed in successfully!")
        log_status("User signed in")

# Load BLIP model (local)
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# Load CLIP model
@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Load fashion data
@st.cache_data
def load_fashion_data():
    df = pd.read_json("fashion_items.json")
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    return df

# Embedding function
def get_image_embedding(image, model, processor):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings[0].cpu().numpy()

def get_top_k_similar_items(query_embedding, df, k=5):
    item_embeddings = np.vstack(df["embedding"].values)
    if query_embedding.shape[0] != item_embeddings.shape[1]:
        st.error("Embedding size mismatch. Upload matching format image.")
        return df.iloc[:k].copy(), [0]*k
    scores = cosine_similarity([query_embedding], item_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return df.iloc[top_indices].copy(), scores[top_indices]

# Virtual Try-On Mock
if page == "👗 Virtual Try-On":
    st.header("👗 Virtual Try-On")
    user_image = st.file_uploader("Upload Your Photo", type=["jpg", "jpeg", "png"], key="user")
    cloth_image = st.file_uploader("Upload Outfit Image", type=["jpg", "jpeg", "png"], key="cloth")

    if user_image and cloth_image:
        st.image(user_image, caption="User Image", width=250)
        st.image(cloth_image, caption="Clothing Image", width=250)
        st.success("🧪 This is a mock! Real merging requires GAN/image model integration.")
        log_status("Virtual try-on image preview loaded")

# Celebrity Match
if page == "🌟 Celebrity Match":
    st.header("🌟 Celebrity Fashion Recommender")
    uploaded_file = st.file_uploader("Upload Celebrity Outfit Photo", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Look", use_container_width=True)

        df = load_fashion_data()
        clip_model, clip_processor = load_clip()
        embed = get_image_embedding(image, clip_model, clip_processor)
        results_df, scores = get_top_k_similar_items(embed, df, k=5)

        for idx, row in results_df.iterrows():
            st.markdown(f"### [{row['name']}]({row['buy_link']})")
            try:
                response = requests.get(row["image_url"])
                st.image(Image.open(BytesIO(response.content)), width=250)
            except:
                st.image("https://via.placeholder.com/250", caption="Image not found", width=250)
            st.markdown(f"[🛒 Buy Now]({row['buy_link']})")
            st.markdown("---")
        log_status("Celebrity match completed")

# Content Creator
if page == "✍️ Content Agent":
    st.header("🧠 AI Content Generator")
    image_file = st.file_uploader("📸 Upload Image", type=["jpg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Uploaded", use_container_width=True)
        blip_processor, blip_model = load_blip()

        inputs = blip_processor(image, return_tensors="pt")
        out = blip_model.generate(**inputs)
        desc = blip_processor.decode(out[0], skip_special_tokens=True)
        st.success(f"📝 Caption: {desc}")

        prompt_template = PromptTemplate(
            input_variables=["desc"],
            template="""
            You are a fashion influencer. Based on: "{desc}", write:
            1. 100-word blog post
            2. Short caption
            3. 5 fashion hashtags
            4. 3 related tags
            """
        )
        llm = Ollama(model="tinyllama")
        chain = LLMChain(prompt=prompt_template, llm=llm)
        response = chain.run(desc=desc)

        st.text_area("🧵 Generated Post", response, height=250)
        platform = st.selectbox("📤 Simulate Post To", ["Instagram", "Twitter"])
        if st.button(f"Post to {platform}"):
            time.sleep(2)
            st.success(f"✅ Post simulated on {platform}")
        log_status("Content agent completed")

# Trend Analyzer (mock)
if page == "📈 Trend Analyzer":
    st.header("📈 Fashion Trend Updates")
    st.success("Trending: Baggy Jeans, Chunky Sneakers, Monochrome Blazers")
    log_status("Trend analysis simulated")

# Status Log
if page == "📋 Status":
    show_status()
