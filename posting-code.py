# pip install tweepy
# Twitter Posting Code
import tweepy

def post_to_twitter(content, api_key, api_secret, access_token, access_token_secret):
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_token_secret)
    api = tweepy.API(auth)
    api.update_status(content)
    return "✅ Tweet posted successfully!"

# Usage in Streamlit
if platform == "Twitter":
    api_key = st.text_input("API Key")
    api_secret = st.text_input("API Secret")
    access_token = st.text_input("Access Token")
    access_token_secret = st.text_input("Access Token Secret")
    if st.button("Post to Twitter"):
        status = post_to_twitter(response, api_key, api_secret, access_token, access_token_secret)
        st.success(status)


# instragram
# Install Required Packages
# pip install requests

# Instagram Posting Code
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


# Usage in Streamlit
if platform == "Instagram":
    ig_user_id = st.text_input("Instagram Business Account ID")
    ig_access_token = st.text_input("Access Token")
    img_url = st.text_input("Image Public URL (must be accessible)")
    if st.button("Post to Instagram"):
        status = post_to_instagram(img_url, response, ig_access_token, ig_user_id)
        st.success(status)
