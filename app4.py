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
