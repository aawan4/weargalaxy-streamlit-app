import cv2
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st
import base64
from pathlib import Path

# --- Helper Function to Encode Image ---
def img_to_bytes(img_path):
    """Encodes an image file to a Base64 string for HTML embedding."""
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        st.error(f"Logo file not found at path: {img_path}")
        return None

# --- Page Configuration ---
st.set_page_config(page_title="WeAR Galaxy", page_icon="logo.png", layout="wide")

# --- Custom CSS for a Modern, Branded Look ---
st.markdown("""
    <style>
        /* Import Google Fonts for a modern feel */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=Montserrat:wght@600;700&display=swap');

        /* --- Root Variables (Color Palette from your Image) --- */
        :root {
            --primary-dark: #2f3a47;      /* Deep Blue-Gray */
            --secondary-dark: #4a5a6d;   /* Lighter Blue-Gray */
            --accent-brown: #8b6d5c;     /* Warm Mid-Tone Brown */
            --text-light: #e0e0e0;       /* Light Gray for text */
            --hover-highlight: #a08170;  /* Lighter brown for hovers */
            --font-main: 'Inter', sans-serif;
            --font-heading: 'Montserrat', sans-serif;
            --shadow-medium: rgba(0, 0, 0, 0.2);
        }

        /* --- Main App Styling --- */
        .stApp {
            background-color: var(--primary-dark);
            color: var(--text-light);
            padding-top: 5rem; /* Add padding to prevent content from hiding behind fixed navbar */
        }
        
        /* --- HIDE STREAMLIT HEADER --- */
        [data-testid="stHeader"] {
            display: none;
        }

        /* --- START: Navbar Styling --- */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 40px;
            background-color: var(--primary-dark);
            box-shadow: 0 2px 10px var(--shadow-medium);
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 1000;
            box-sizing: border-box;
        }

        .navbar .logo a {
            text-decoration: none;
            display: flex; /* Helps vertically align the image */
            align-items: center;
        }
        
        .navbar .logo img {
            transition: opacity 0.3s ease;
            height: 40px; /* Set a specific height for the logo */
        }
        
        .navbar .logo a:hover img {
            opacity: 0.8;
        }

        .nav-links {
            list-style: none;
            display: flex;
            align-items: center;
            gap: 25px;
            margin: 0;
        }

        .nav-links li a {
            text-decoration: none;
            color: var(--text-light);
            font-size: 1em;
            font-weight: 500;
            padding: 8px 15px;
            border-radius: 5px;
            transition: background-color 0.3s ease, color 0.3s ease;
            font-family: var(--font-main);
        }

        .nav-links li a:hover {
            background-color: var(--accent-brown);
            color: var(--text-light);
        }

        .nav-links .ai-link {
            background-color: var(--accent-brown);
            font-weight: 700;
        }
        .nav-links .ai-link:hover {
            background-color: var(--hover-highlight);
        }
        /* --- END: Navbar Styling --- */

        /* --- Typography --- */
        h1, h2, h3 {
            font-family: var(--font-heading);
            color: var(--text-light);
        }
        
        p, .stMarkdown, .stRadio, .stSelectbox, .stFileUploader {
            font-family: var(--font-main);
            color: var(--text-light);
        }

        /* --- Component Styling --- */
        
        /* Buttons */
        div[data-testid="stButton"] > button {
            background-color: var(--accent-brown);
            color: var(--text-light);
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
            transition: all 0.3s ease;
        }
        div[data-testid="stButton"] > button:hover {
            background-color: var(--hover-highlight);
            transform: translateY(-2px);
        }
        
        /* Radio Buttons */
        div[data-testid="stRadio"] > div {
            padding: 5px;
            background-color: var(--secondary-dark);
            border-radius: 10px;
        }

        /* Select Box */
        div[data-testid="stSelectbox"] > div {
            background-color: var(--secondary-dark);
            border-radius: 8px;
        }

        /* Analysis Result Box */
        div[data-testid="stMarkdownContainer"] pre {
            background-color: var(--secondary-dark);
            border: 1px solid var(--accent-brown);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text-light);
        }

        /* Chat Interface */
        div[data-testid="stChatInput"] {
            background-color: var(--secondary-dark);
        }
        .stChatMessage {
            background-color: var(--secondary-dark);
            border: 1px solid var(--accent-brown);
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Gemini API Configuration ---
try:
    API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"FATAL ERROR: Could not configure Gemini API. Please set your GEMINI_API_KEY. Error: {e}")
    st.stop()

# --- Initialize Session State ---
if 'analysis_text' not in st.session_state:
    st.session_state['analysis_text'] = "Analysis will appear here."
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat" not in st.session_state:
    st.session_state.chat = None

# --- Core AI Functions ---
def analyze_image_with_gemini(image_frame):
    """Sends an image to Gemini and returns face shape analysis."""
    rgb_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    prompt = """
    Analyze the face in this image. 
    1. Determine the face shape (e.g., Oval, Round, Square, Heart).
    2. Based on that shape, provide a concise suggestion for suitable glasses in 15 words or less.
    Format your response exactly like this:
    Your Face Shape Is: [Detected Shape]
    WeAR AI's Suggestion: [Your Suggestion]
    """
    try:
        st.session_state['analysis_text'] = "Analyzing with AI..."
        response = model.generate_content([prompt, pil_image])
        st.session_state['analysis_text'] = response.text.strip()
    except Exception as e:
        st.session_state['analysis_text'] = f"API Call Failed: {str(e)}"

def get_suggestion_for_shape(shape_name):
    """Sends a face shape name to Gemini and returns a suggestion."""
    prompt = f"""
    You are a helpful and concise fashion assistant named WeAR AI. My face shape is '{shape_name}'.
    In 15 words or less, what are the best types of glasses for me?
    Format your response exactly like this:
    WeAR AI's Suggestion: [Your Suggestion]
    """
    try:
        st.session_state['analysis_text'] = f"Getting suggestion for {shape_name} face..."
        response = model.generate_content(prompt)
        st.session_state['analysis_text'] = f"Your Face Shape Is: {shape_name}\n{response.text.strip()}"
    except Exception as e:
        st.session_state['analysis_text'] = f"API Call Failed: {str(e)}"

# --- UI Layout ---

# --- START: Injected Navbar HTML with Custom Logo ---
LOGO_PATH = "logo.png"
logo_base64 = img_to_bytes(LOGO_PATH)

st.markdown(f"""
    <nav class="navbar">
        <div class="logo">
            <a href="#">
                <img src="data:image/png;base64,{logo_base64}" alt="WeAR Galaxy Logo">
            </a>
        </div>
        <ul class="nav-links">
           <li><a href="https://weargalaxy.me/" target="_self">Home</a></li>
            <li><a href="https://weargalaxy.me/about/" target="_self">About</a></li>
            <li><a href="https://weargalaxy.me/gallery/" target="_self">Gallery</a></li>
            <li><a href="https://ai.weargalaxy.me/" class="ai-link" target="_self">WeAR AI 🚀</a></li>
            <li><a href="https://weargalaxy.me/contact/" target="_self">Contact</a></li>
        </ul>
    </nav>
""", unsafe_allow_html=True)
# --- END: Injected Navbar HTML ---


st.markdown("""
    <div style="text-align: center;">
        <h1>👓 WeAR Galaxy AI Glasses Style Advisor</h1>
        <p>Get personalized glasses recommendations based on your face shape, powered by WeAR AI.</p>
    </div>
""", unsafe_allow_html=True)

st.write("---")

mode = st.radio(
    "Choose your input method:",
    ("Webcam", "Upload Image", "Manual Input", "Chatbot"),
    horizontal=True,
    label_visibility="collapsed"
)

# --- Logic for Analysis Modes (Webcam, Upload, Manual) ---
if mode != "Chatbot":
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.header("Your Input")

        if mode == "Webcam":
            st.write("Position your face in the frame and click the button below.")
            picture = st.camera_input("Webcam Capture", label_visibility="collapsed")
            if picture:
                st.write("Photo Captured! Click 'Analyze Photo' to proceed.")
                if st.button("Analyze Photo"):
                    file_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
                    image_to_analyze = cv2.imdecode(file_bytes, 1)
                    analyze_image_with_gemini(image_to_analyze)

        elif mode == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", width=300)
                if st.button("Analyze Uploaded Image"):
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image_to_analyze = cv2.imdecode(file_bytes, 1)
                    analyze_image_with_gemini(image_to_analyze)

        elif mode == "Manual Input":
            face_shapes = ["Select a Shape", "Oval", "Square", "Round", "Heart"]
            selected_shape = st.selectbox(
                "What is your face shape?",
                face_shapes,
                label_visibility="collapsed"
            )
            if selected_shape != "Select a Shape":
                get_suggestion_for_shape(selected_shape)

    with col2:
        st.header("AI Analysis")
        st.markdown(f"**Analysis Result:**\n```\n{st.session_state['analysis_text']}\n```")

# --- Logic for Chatbot Mode ---
elif mode == "Chatbot":
    st.header("Conversational AI Advisor")
    
    if st.session_state.chat is None:
        system_instruction = """You are a specialized AI fashion assistant for an app called 'WeAR Galaxy'. 
        Your name is WeAR AI. Your ONLY purpose is to answer questions about eyeglass frames, styles, materials, 
        and what frames are suitable for different face shapes. You MUST politely refuse to answer any question that is 
        not related to eyeglasses. If asked an off-topic question, say something like, 'I am the WeAR AI assistant 
        and my expertise is limited to eyeglass frames. How can I help you with glasses today?'"""
        st.session_state.chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_instruction]},
            {'role': 'model', 'parts': ["Okay, I understand. I am the WeAR AI, ready to assist with all questions about eyeglass frames."]}
        ])
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the WeAR AI. How can I help you find the perfect glasses frames today?"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about glasses styles..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = st.session_state.chat.send_message(prompt)
            with st.chat_message("assistant"):
                st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})


