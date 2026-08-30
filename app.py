```python
import os
import base64
import io
import json
import traceback

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import google.generativeai as genai


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="WeAR Galaxy AI API",
    description="AI-powered glasses style advisor",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel(
        "gemini-2.5-flash"
    )
else:
    model = None


# ============================================================
# REQUEST MODELS
# ============================================================

class FaceShapeRequest(BaseModel):
    shape: str


class ChatRequest(BaseModel):
    message: str


# ============================================================
# HELPER
# ============================================================

def get_model():
    """
    Return the Gemini model.

    Raises a clean API error if the Gemini key
    has not been configured.
    """

    if not API_KEY or model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured. "
                "Add GEMINI_API_KEY to your Vercel Environment Variables."
            ),
        )

    return model


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "application": "WeAR Galaxy AI",
        "message": "WeAR Galaxy AI API is running.",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "healthy",
        "gemini_configured": bool(API_KEY),
    }


# ============================================================
# FACE SHAPE ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...)
):
    """
    Analyze an uploaded image using Gemini.

    Expected input:
        multipart/form-data
        file = image
    """

    try:

        ai_model = get_model()

        # ----------------------------------------------------
        # Validate file type
        # ----------------------------------------------------

        if not file.content_type:
            raise HTTPException(
                status_code=400,
                detail="No file type was provided.",
            )

        allowed_types = {
            "image/jpeg",
            "image/jpg",
            "image/png",
            "image/webp",
        }

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image type. "
                    "Please upload JPG, PNG, or WEBP."
                ),
            )

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty.",
            )

        # ----------------------------------------------------
        # Limit image size
        # ----------------------------------------------------

        max_size = 10 * 1024 * 1024

        if len(image_bytes) > max_size:
            raise HTTPException(
                status_code=413,
                detail="Image is too large. Maximum size is 10 MB.",
            )

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.load()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid image.",
            )

        # ----------------------------------------------------
        # Convert image to RGB
        # ----------------------------------------------------

        if image.mode != "RGB":
            image = image.convert("RGB")

        # ----------------------------------------------------
        # Gemini prompt
        # ----------------------------------------------------

        prompt = """
You are WeAR AI, a specialized eyeglass fashion assistant.

Analyze the person's face in the provided image.

Determine the most likely face shape.

Possible face shapes include:

- Oval
- Round
- Square
- Heart
- Diamond
- Oblong

Then recommend suitable eyeglass frame styles for that face shape.

Do not identify the person.
Do not provide personal identity information.
Only analyze visible facial proportions relevant to eyeglass styling.

Keep the glasses recommendation to 15 words or less.

Return ONLY this format:

Your Face Shape Is: [Detected Shape]
WeAR AI's Suggestion: [Short glasses recommendation]
"""

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------

        response = ai_model.generate_content(
            [
                prompt,
                image,
            ]
        )

        if not response:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned no response.",
            )

        result = getattr(
            response,
            "text",
            None,
        )

        if not result:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response.",
            )

        result = result.strip()

        # ----------------------------------------------------
        # Return result
        # ----------------------------------------------------

        return {
            "success": True,
            "filename": file.filename,
            "analysis": result,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "IMAGE ANALYSIS ERROR:",
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"Image analysis failed: {str(e)}",
        )


# ============================================================
# MANUAL FACE SHAPE RECOMMENDATION
# ============================================================

@app.post("/api/suggestion")
async def manual_suggestion(
    request: FaceShapeRequest
):
    """
    Generate glasses recommendations
    based on manually selected face shape.
    """

    try:

        ai_model = get_model()

        shape = request.shape.strip()

        if not shape:
            raise HTTPException(
                status_code=400,
                detail="Face shape is required.",
            )

        allowed_shapes = {
            "Oval",
            "Round",
            "Square",
            "Heart",
            "Diamond",
            "Oblong",
        }

        # Case-insensitive validation

        matched_shape = None

        for allowed_shape in allowed_shapes:

            if allowed_shape.lower() == shape.lower():

                matched_shape = allowed_shape
                break

        if not matched_shape:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid face shape. "
                    "Choose Oval, Round, Square, Heart, "
                    "Diamond, or Oblong."
                ),
            )

        prompt = f"""
You are WeAR AI, a concise eyeglass fashion assistant.

The user's face shape is:

{matched_shape}

Recommend the most suitable eyeglass frame styles.

Keep the recommendation to 15 words or less.

Return ONLY:

WeAR AI's Suggestion: [Your recommendation]
"""

        response = ai_model.generate_content(
            prompt
        )

        if not response or not response.text:

            raise HTTPException(
                status_code=500,
                detail="Gemini returned no recommendation.",
            )

        recommendation = response.text.strip()

        return {
            "success": True,
            "face_shape": matched_shape,
            "recommendation": recommendation,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "MANUAL SUGGESTION ERROR:",
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}",
        )


# ============================================================
# CHATBOT
# ============================================================

@app.post("/api/chat")
async def chatbot(
    request: ChatRequest
):
    """
    WeAR AI glasses chatbot.
    """

    try:

        ai_model = get_model()

        message = request.message.strip()

        if not message:

            raise HTTPException(
                status_code=400,
                detail="Message cannot be empty.",
            )

        if len(message) > 4000:

            raise HTTPException(
                status_code=400,
                detail="Message is too long.",
            )

        system_instruction = """
You are WeAR AI, a specialized AI fashion assistant
for an application called WeAR Galaxy.

Your ONLY area of expertise is eyeglasses.

You can answer questions about:

- Eyeglass frames
- Eyeglass styles
- Eyeglass materials
- Glasses for different face shapes
- Frame sizing
- Eyeglass fashion
- Choosing frames
- Lens/frame appearance
- General glasses styling

You must politely refuse questions unrelated to eyeglasses.

For unrelated questions, say:

"I am the WeAR AI assistant and my expertise is
limited to eyeglass frames. How can I help you
with glasses today?"

Do not claim to diagnose medical conditions.

Keep answers useful and reasonably concise.
"""

        prompt = f"""
{system_instruction}

User question:

{message}

Answer the user now.
"""

        response = ai_model.generate_content(
            prompt
        )

        if not response or not response.text:

            raise HTTPException(
                status_code=500,
                detail="Gemini returned no chatbot response.",
            )

        answer = response.text.strip()

        return {
            "success": True,
            "message": answer,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "CHAT ERROR:",
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"Chatbot failed: {str(e)}",
        )


# ============================================================
# BASE64 IMAGE ANALYSIS
# ============================================================

class Base64ImageRequest(BaseModel):
    image: str


@app.post("/api/analyze-base64")
async def analyze_base64(
    request: Base64ImageRequest
):
    """
    Analyze a Base64 encoded image.

    Useful for webcam images captured
    directly in browser JavaScript.
    """

    try:

        ai_model = get_model()

        encoded_image = request.image

        if not encoded_image:

            raise HTTPException(
                status_code=400,
                detail="Image data is required.",
            )

        # ----------------------------------------------------
        # Remove data URI prefix if present
        # ----------------------------------------------------

        if "," in encoded_image:

            encoded_image = encoded_image.split(
                ",",
                1
            )[1]

        # ----------------------------------------------------
        # Decode
        # ----------------------------------------------------

        try:

            image_bytes = base64.b64decode(
                encoded_image
            )

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Invalid Base64 image data.",
            )

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        try:

            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.load()

        except Exception:

            raise HTTPException(
                status_code=400,
                detail="Invalid image data.",
            )

        if image.mode != "RGB":

            image = image.convert("RGB")

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        prompt = """
You are WeAR AI, an eyeglass fashion assistant.

Analyze the visible face and determine the
most likely face shape.

Choose from:

Oval
Round
Square
Heart
Diamond
Oblong

Then recommend suitable eyeglass frame styles.

Do not identify the person.

Return ONLY:

Your Face Shape Is: [Detected Shape]
WeAR AI's Suggestion: [Short recommendation]

Keep the recommendation to 15 words or less.
"""

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        response = ai_model.generate_content(
            [
                prompt,
                image,
            ]
        )

        if not response or not response.text:

            raise HTTPException(
                status_code=500,
                detail="Gemini returned no analysis.",
            )

        return {
            "success": True,
            "analysis": response.text.strip(),
        }

    except HTTPException:
        raise

    except Exception as e:

        print(
            "BASE64 ANALYSIS ERROR:",
            traceback.format_exc()
        )

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api")
async def api_info():

    return {
        "application": "WeAR Galaxy AI",
        "status": "online",
        "endpoints": {
            "health": "/api/health",
            "analyze_image": "/api/analyze",
            "analyze_base64": "/api/analyze-base64",
            "manual_suggestion": "/api/suggestion",
            "chatbot": "/api/chat",
        },
    }
```
