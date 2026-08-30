import os
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import google.generativeai as genai


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WeAR Galaxy",
    page_icon="logo.png",
    layout="wide"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def img_to_bytes(img_path):
    """Encode an image file as Base64 for HTML embedding."""
    try:
        img_bytes = Path(img_path).read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        return encoded
    except FileNotFoundError:
        st.error(f"Logo file not found at path: {img_path}")
        return None


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700'
        '&family=Montserrat:wght@600;700&display=swap'
    );

    :root {
        --primary-dark: #2f3a47;
        --secondary-dark: #4a5a6d;
        --accent-brown: #8b6d5c;
        --text-light: #e0e0e0;
        --hover-highlight: #a08170;
        --font-main: 'Inter', sans-serif;
        --font-heading: 'Montserrat', sans-serif;
        --shadow-medium: rgba(0, 0, 0, 0.2);
    }

    /* Main application */

    .stApp {
        background-color: var(--primary-dark);
        color: var(--text-light);
        padding-top: 5rem;
    }

    /* Hide Streamlit header */

    [data-testid="stHeader"] {
        display: none;
    }

    /* Navbar */

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
        display: flex;
        align-items: center;
    }

    .navbar .logo img {
        transition: opacity 0.3s ease;
        height: 40px;
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
        padding: 0;
    }

    .nav-links li a {
        text-decoration: none;

        color: var(--text-light);

        font-size: 1em;

        font-weight: 500;

        padding: 8px 15px;

        border-radius: 5px;

        transition:
            background-color 0.3s ease,
            color 0.3s ease;

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

    /* Typography */

    h1,
    h2,
    h3 {
        font-family: var(--font-heading);
        color: var(--text-light);
    }

    p,
    .stMarkdown,
    .stRadio,
    .stSelectbox,
    .stFileUploader {
        font-family: var(--font-main);
        color: var(--text-light);
    }

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
    """,
    unsafe_allow_html=True
)


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

try:
    API_KEY = os.getenv("GEMINI_API_KEY")

    if not API_KEY:
        try:
            API_KEY = st.secrets["GEMINI_API_KEY"]
        except Exception:
            API_KEY = None

    if not API_KEY:
        st.error(
            "GEMINI_API_KEY is not configured. "
            "Please add it to your environment variables or Streamlit secrets."
        )
        st.stop()

    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )

except Exception as e:
    st.error(
        f"FATAL ERROR: Could not configure Gemini API. Error: {e}"
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_text" not in st.session_state:
    st.session_state["analysis_text"] = "Analysis will appear here."

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat" not in st.session_state:
    st.session_state["chat"] = None


# ============================================================
# AI FUNCTION - IMAGE ANALYSIS
# ============================================================

def analyze_image_with_gemini(image_frame):
    """
    Send an image to Gemini and return face shape analysis.
    """

    try:

        if image_frame is None:
            st.session_state["analysis_text"] = (
                "Could not read the selected image."
            )
            return

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(
            image_frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to PIL image
        pil_image = Image.fromarray(rgb_frame)

        prompt = """
        Analyze the face in this image.

        1. Determine the face shape.
           Examples:
           Oval, Round, Square, Heart, Diamond, Oblong.

        2. Based on that face shape, provide a concise
           suggestion for suitable eyeglass frames.

        Keep the glasses suggestion to 15 words or less.

        Format your response exactly like this:

        Your Face Shape Is: [Detected Shape]
        WeAR AI's Suggestion: [Your Suggestion]
        """

        st.session_state["analysis_text"] = "Analyzing with AI..."

        response = model.generate_content(
            [
                prompt,
                pil_image
            ]
        )

        if response and response.text:
            st.session_state["analysis_text"] = (
                response.text.strip()
            )
        else:
            st.session_state["analysis_text"] = (
                "The AI did not return an analysis."
            )

    except Exception as e:

        st.session_state["analysis_text"] = (
            f"API Call Failed: {str(e)}"
        )


# ============================================================
# AI FUNCTION - MANUAL FACE SHAPE
# ============================================================

def get_suggestion_for_shape(shape_name):
    """
    Ask Gemini for glasses recommendations
    based on manually selected face shape.
    """

    try:

        prompt = f"""
        You are a helpful and concise fashion assistant
        named WeAR AI.

        The user's face shape is "{shape_name}".

        In 15 words or less, recommend the best types
        of eyeglass frames for this face shape.

        Format your response exactly like this:

        WeAR AI's Suggestion: [Your Suggestion]
        """

        st.session_state["analysis_text"] = (
            f"Getting suggestion for {shape_name} face..."
        )

        response = model.generate_content(prompt)

        if response and response.text:

            st.session_state["analysis_text"] = (
                f"Your Face Shape Is: {shape_name}\n"
                f"{response.text.strip()}"
            )

        else:

            st.session_state["analysis_text"] = (
                "The AI did not return a suggestion."
            )

    except Exception as e:

        st.session_state["analysis_text"] = (
            f"API Call Failed: {str(e)}"
        )


# ============================================================
# NAVBAR
# ============================================================

LOGO_PATH = "logo.png"

logo_base64 = img_to_bytes(LOGO_PATH)

if logo_base64:

    logo_html = f"""
        <img
            src="data:image/png;base64,{logo_base64}"
            alt="WeAR Galaxy Logo"
        >
    """

else:

    logo_html = """
        <span style="font-size: 24px;">
            WeAR Galaxy
        </span>
    """


st.markdown(
    f"""
    <nav class="navbar">

        <div class="logo">

            <a href="https://weargalaxy.me/" target="_self">
                {logo_html}
            </a>

        </div>

        <ul class="nav-links">

            <li>
                <a
                    href="https://weargalaxy.me/"
                    target="_self"
                >
                    Home
                </a>
            </li>

            <li>
                <a
                    href="https://weargalaxy.me/about/"
                    target="_self"
                >
                    About
                </a>
            </li>

            <li>
                <a
                    href="https://weargalaxy.me/gallery/"
                    target="_self"
                >
                    Gallery
                </a>
            </li>

            <li>
                <a
                    href="https://ai.weargalaxy.me/"
                    class="ai-link"
                    target="_self"
                >
                    WeAR AI 🚀
                </a>
            </li>

            <li>
                <a
                    href="https://weargalaxy.me/contact/"
                    target="_self"
                >
                    Contact
                </a>
            </li>

        </ul>

    </nav>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PAGE TITLE
# ============================================================

st.markdown(
    """
    <div style="text-align: center;">

        <h1>
            👓 WeAR Galaxy AI Glasses Style Advisor
        </h1>

        <p>
            Get personalized glasses recommendations
            based on your face shape, powered by WeAR AI.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


st.write("---")


# ============================================================
# INPUT MODE
# ============================================================

mode = st.radio(
    "Choose your input method:",
    (
        "Webcam",
        "Upload Image",
        "Manual Input",
        "Chatbot"
    ),
    horizontal=True,
    label_visibility="collapsed"
)


# ============================================================
# IMAGE / MANUAL INPUT MODES
# ============================================================

if mode != "Chatbot":

    col1, col2 = st.columns(
        2,
        gap="large"
    )

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        st.header("Your Input")

        # ====================================================
        # WEBCAM
        # ====================================================

        if mode == "Webcam":

            st.write(
                "Position your face in the frame "
                "and click the button below."
            )

            picture = st.camera_input(
                "Webcam Capture",
                label_visibility="collapsed"
            )

            if picture:

                st.write(
                    "Photo Captured! "
                    "Click 'Analyze Photo' to proceed."
                )

                if st.button("Analyze Photo"):

                    file_bytes = np.asarray(
                        bytearray(picture.read()),
                        dtype=np.uint8
                    )

                    image_to_analyze = cv2.imdecode(
                        file_bytes,
                        cv2.IMREAD_COLOR
                    )

                    analyze_image_with_gemini(
                        image_to_analyze
                    )

        # ====================================================
        # UPLOAD IMAGE
        # ====================================================

        elif mode == "Upload Image":

            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                label_visibility="collapsed"
            )

            if uploaded_file:

                st.image(
                    uploaded_file,
                    caption="Uploaded Image",
                    width=300
                )

                if st.button(
                    "Analyze Uploaded Image"
                ):

                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()),
                        dtype=np.uint8
                    )

                    image_to_analyze = cv2.imdecode(
                        file_bytes,
                        cv2.IMREAD_COLOR
                    )

                    analyze_image_with_gemini(
                        image_to_analyze
                    )

        # ====================================================
        # MANUAL INPUT
        # ====================================================

        elif mode == "Manual Input":

            face_shapes = [
                "Select a Shape",
                "Oval",
                "Square",
                "Round",
                "Heart",
                "Diamond",
                "Oblong"
            ]

            selected_shape = st.selectbox(
                "What is your face shape?",
                face_shapes,
                label_visibility="collapsed"
            )

            if selected_shape != "Select a Shape":

                if st.button(
                    "Get Glasses Recommendation"
                ):

                    get_suggestion_for_shape(
                        selected_shape
                    )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        st.header("AI Analysis")

        st.markdown(
            f"""
            **Analysis Result:**

            ```
            {st.session_state["analysis_text"]}
            ```
            """
        )


# ============================================================
# CHATBOT MODE
# ============================================================

elif mode == "Chatbot":

    st.header(
        "Conversational AI Advisor"
    )

    # --------------------------------------------------------
    # INITIALIZE CHAT
    # --------------------------------------------------------

    if st.session_state.chat is None:

        system_instruction = """
        You are a specialized AI fashion assistant
        for an app called "WeAR Galaxy".

        Your name is WeAR AI.

        Your ONLY purpose is to answer questions about:

        - Eyeglass frames
        - Eyeglass styles
        - Eyeglass materials
        - Glasses suitable for different face shapes
        - Frame sizing
        - General eyeglass fashion advice

        You MUST politely refuse to answer questions
        unrelated to eyeglasses.

        If asked an off-topic question, respond:

        "I am the WeAR AI assistant and my expertise
        is limited to eyeglass frames.
        How can I help you with glasses today?"
        """

        try:

            st.session_state.chat = model.start_chat(
                history=[
                    {
                        "role": "user",
                        "parts": [system_instruction]
                    },
                    {
                        "role": "model",
                        "parts": [
                            "Okay, I understand. "
                            "I am WeAR AI, ready to assist "
                            "with questions about eyeglass frames."
                        ]
                    }
                ]
            )

        except Exception as e:

            st.error(
                f"Could not initialize chatbot: {e}"
            )

            st.stop()

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! I am the WeAR AI. "
                    "How can I help you find "
                    "the perfect glasses frames today?"
                )
            }
        ]

    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    prompt = st.chat_input(
        "Ask about glasses styles..."
    )

    if prompt:

        # Display user message

        with st.chat_message("user"):

            st.markdown(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        # Generate response

        with st.spinner("Thinking..."):

            try:

                response = (
                    st.session_state.chat
                    .send_message(prompt)
                )

                assistant_response = (
                    response.text
                    if response and response.text
                    else "Sorry, I couldn't generate a response."
                )

            except Exception as e:

                assistant_response = (
                    f"AI response failed: {str(e)}"
                )

        # Display assistant response

        with st.chat_message("assistant"):

            st.markdown(
                assistant_response
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response
            }
        )
