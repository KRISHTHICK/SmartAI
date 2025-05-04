#AI Content Creation

Here's a working Streamlit-based implementation for **Feature 4: AI Content Creation Agent** using your SmartAI fashion platform:

---

### ✅ **AI Content Creation Agent - `app.py`**

```python
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms.ollama import Ollama

# Set up Streamlit
st.set_page_config(page_title="SmartAI - Content Creation Agent", layout="wide")
st.title("🧠 AI Content Creation Agent for Fashion Posts")

# Load BLIP model and processor for image captioning
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

# Image Upload
uploaded_file = st.file_uploader("📸 Upload Your Fashion Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Generate image caption
    with st.spinner("🧠 Generating image description..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        st.success("✅ Description Generated:")
        st.markdown(f"> **{description}**")

    # Generate social media post
    with st.spinner("✍️ Generating Blog/Caption/Hashtags..."):
        prompt_template = PromptTemplate(
            input_variables=["desc"],
            template=(
                "You are a fashion social media influencer assistant. Based on the image description: '{desc}', "
                "write a 100-word Instagram blog-style post, generate an attractive caption, and list 5 hashtags."
            )
        )
        llm = Ollama(model="tinyllama")
        chain = LLMChain(prompt=prompt_template, llm=llm)
        result = chain.run(desc=description)

    st.subheader("📢 Generated Social Media Content")
    st.text_area("Full Post (Caption + Hashtags):", result, height=200)

    st.markdown("✅ You can now copy and paste this content directly to Instagram or Twitter!")

```

---

### 🧩 Optional Enhancements (Future Ideas)
- Add a **"Post to Instagram/Twitter"** button via API.
- Store caption history using local database or Firebase.
- Add outfit/style detection and incorporate it in the prompt.

Would you like help adding this as a feature inside your main SmartAI platform with a button on the homepage?
