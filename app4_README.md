#AI Content Creation

Here’s the **complete Streamlit app code** for your **“AI Content Creation Agent”** feature inside **SmartAI**, based on your requirement:

---

### ✅ Full Code: `content_creator.py`

This module lets users upload a photo, generates a fashion blog post, caption, hashtags, and (optionally) simulates posting to Instagram/Twitter.

```python
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama
import time

# Load BLIP image captioning model only once
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def run_content_creator():
    st.subheader("🧠 AI Content Creation Agent")
    st.markdown("""
        Upload a photo, and let SmartAI generate:
        - 📖 Fashion blog-style social media post  
        - 💬 Catchy caption  
        - 🔖 Relevant hashtags and tags  
        - 📤 Option to simulate posting to Instagram or Twitter
    """)

    processor, model = load_blip_model()

    uploaded_file = st.file_uploader("📸 Upload Your Fashion Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("🔍 Analyzing fashion style from image..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            st.success("📝 Image Description Generated:")
            st.markdown(f"> **{description}**")

        with st.spinner("🧠 Creating social content..."):
            prompt_template = PromptTemplate(
                input_variables=["desc"],
                template="""
                You are a top-tier fashion influencer assistant. Based on the following description: "{desc}", do the following:
                1. Write a short 100-word Instagram-style fashion blog post.
                2. Suggest a trendy caption.
                3. List 5 fashion-related hashtags.
                4. Mention 3 relevant tags (e.g., brands, fashion accounts).
                Format output as:
                Blog: ...
                Caption: ...
                Hashtags: ...
                Tags: ...
                """
            )
            llm = Ollama(model="tinyllama")
            chain = LLMChain(prompt=prompt_template, llm=llm)
            response = chain.run(desc=description)

        st.subheader("📢 Generated Social Media Content")
        st.text_area("💬 Full Content", value=response, height=300)

        # Simulate social media post
        st.divider()
        st.subheader("🚀 Post to Social Media")
        platform = st.selectbox("Choose Platform to Simulate Post", ["Instagram", "Twitter"])
        if st.button(f"Simulate Posting to {platform}"):
            with st.spinner(f"Posting to {platform}..."):
                time.sleep(2)
            st.success(f"✅ Post successfully simulated on {platform}!")

        st.info("Note: Actual API posting to Instagram/Twitter requires platform integration and authentication.")
```

---

### 🧭 Usage

Use this module with your `main_app.py` navigation like this:

```python
elif selection == "AI Content Creation Agent":
    from content_creator import run_content_creator
    run_content_creator()
```

---

Would you like me to help with **real Instagram/Twitter API integration** (posting from Streamlit)?




Great! Below is a **complete guide** + code snippets to help you **post to Twitter and Instagram from your AI Content Creation Agent** app.

---

## ✅ Real API Integration Plan

### ⚠️ Important Notes:
- **Instagram Posting via API** only supports **Business Accounts** using the **Facebook Graph API**.
- **Twitter/X posting** is done via **Twitter Developer API v2 or X API**.

---

## 🔧 1. Twitter (X) Posting Setup

### 🔑 Requirements:
- [Apply for a Twitter Developer Account](https://developer.twitter.com/)
- Create a Project & App
- Get API keys:
  - `API_KEY`
  - `API_SECRET`
  - `ACCESS_TOKEN`
  - `ACCESS_TOKEN_SECRET`

---

### ✅ Install Tweepy

```bash
pip install tweepy
```

---

### ✅ Twitter Posting Code

```python
import tweepy

def post_to_twitter(content, api_key, api_secret, access_token, access_token_secret):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    api.update_status(content)
    return "✅ Tweet posted successfully!"
```

---

### 🧪 Usage in Streamlit

```python
if platform == "Twitter":
    api_key = st.text_input("API Key")
    api_secret = st.text_input("API Secret")
    access_token = st.text_input("Access Token")
    access_token_secret = st.text_input("Access Token Secret")
    if st.button("Post to Twitter"):
        status = post_to_twitter(response, api_key, api_secret, access_token, access_token_secret)
        st.success(status)
```

---

## 📷 2. Instagram Posting Setup (via Facebook Graph API)

### 🔑 Requirements:
1. Convert Instagram to **Business Account**
2. Link it to a **Facebook Page**
3. Go to [Facebook Developer Console](https://developers.facebook.com/)
4. Create App → Get:
   - `Page Access Token`
   - `Instagram Business Account ID`

---

### ✅ Install Required Packages

```bash
pip install requests
```

---

### ✅ Instagram Posting Code

```python
import requests

def post_to_instagram(img_path, caption, access_token, ig_user_id):
    # Step 1: Upload image
    image_upload_url = f"https://graph.facebook.com/v18.0/{ig_user_id}/media"
    image_payload = {
        "image_url": img_path,  # Must be publicly accessible
        "caption": caption,
        "access_token": access_token
    }
    res = requests.post(image_upload_url, data=image_payload)
    creation_id = res.json().get("id")

    # Step 2: Publish image
    publish_url = f"https://graph.facebook.com/v18.0/{ig_user_id}/media_publish"
    publish_payload = {
        "creation_id": creation_id,
        "access_token": access_token
    }
    final_res = requests.post(publish_url, data=publish_payload)
    return "✅ Instagram post published!" if final_res.ok else final_res.json()
```

---

### 🧪 Usage in Streamlit

```python
if platform == "Instagram":
    ig_user_id = st.text_input("Instagram Business Account ID")
    ig_access_token = st.text_input("Access Token")
    img_url = st.text_input("Image Public URL (must be accessible)")
    if st.button("Post to Instagram"):
        status = post_to_instagram(img_url, response, ig_access_token, ig_user_id)
        st.success(status)
```

---

## ✅ Summary

| Platform | Support | Notes |
|----------|---------|-------|
| Twitter  | ✅ Yes   | Directly post with Tweepy |
| Instagram| ✅ Limited | Business Account + Public Image URL required |

---

Would you like a **Streamlit UI version with both Twitter & Instagram input fields** included in the code?
