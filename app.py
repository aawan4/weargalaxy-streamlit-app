# Final app.py code
import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import os
import threading
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="AI Glasses Advisor", page_icon="👓", layout="wide")

try:
    API_KEY = os.getenv("GEMINI_API_KEY")
    if not API_KEY:
        API_KEY = st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    st.error(f"FATAL ERROR: Could not configure Gemini API. Please set your GEMINI_API_KEY. Error: {e}")
    st.stop()

if 'analysis_text' not in st.session_state:
    st.session_state['analysis_text'] = "Analysis will appear here."

def get_ai_analysis(image_frame):
    rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    prompt = """
    Analyze the face in this image. 
    1. Determine the face shape (e.g., Oval, Round, Square, Heart).
    2. Based on that shape, provide a concise suggestion for suitable glasses in 15 words or less.
    Format your response exactly like this:
    Shape: [Detected Shape]
    Suggestion: [Your Suggestion]
    """
    try:
        response = model.generate_content([prompt, pil_image])
        return response.text.strip()
    except Exception as e:
        return f"API Call Failed: {str(e)}"

def threaded_get_ai_analysis(image_frame):
    result = get_ai_analysis(image_frame)
    st.session_state['analysis_text'] = result

st.title("👓 AI Glasses Style Advisor")
st.markdown("Get personalized glasses recommendations based on your face shape, powered by Google Gemini.")
col1, col2 = st.columns(2)
with col1:
    st.header("Your Image")
    mode = st.radio("Choose your input method:", ("Live Webcam", "Upload an Image"), horizontal=True)
    image_placeholder = st.empty()
with col2:
    st.header("AI Analysis")
    analysis_placeholder = st.empty()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_lock = threading.Lock()
        self.latest_frame = None
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        with self.frame_lock:
            self.latest_frame = img
        return img

if mode == "Live Webcam":
    ctx = webrtc_streamer(key="webcam", video_transformer_factory=VideoTransformer, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    st.sidebar.header("Controls")
    if st.sidebar.button("Analyze My Face"):
        if ctx.video_transformer and ctx.video_transformer.latest_frame is not None:
            frame_to_analyze = ctx.video_transformer.latest_frame
            st.session_state['analysis_text'] = "Analyzing..."
            threading.Thread(target=threaded_get_ai_analysis, args=(frame_to_analyze,)).start()
        else:
            st.sidebar.warning("Webcam is not active or feed has not started.")
    image_placeholder.markdown("<div style='text-align: center;'>Webcam feed will appear here once you press START.</div>", unsafe_allow_html=True)

if mode == "Upload an Image":
    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_placeholder.image(image, channels="BGR", use_container_width=True)
        if st.sidebar.button("Analyze This Image"):
            st.session_state['analysis_text'] = "Analyzing..."
            result = get_ai_analysis(image)
            st.session_state['analysis_text'] = result
    else:
        image_placeholder.markdown("<div style='text-align: center;'>Upload an image to begin.</div>", unsafe_allow_html=True)

analysis_placeholder.markdown(f"**Analysis Result:**\n```\n{st.session_state['analysis_text']}\n```")