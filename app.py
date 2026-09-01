```python
from __future__ import annotations

import base64
import binascii
import io
import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="WeAR Galaxy AI API",
    description="AI-powered glasses style advisor",
    version="1.0.0",
)

# Vercel-compatible exports
application = app
handler = app


# ============================================================
# CONFIGURATION
# ============================================================

logging.basicConfig(level=logging.INFO)

LOG = logging.getLogger("wear-galaxy-ai")

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
).strip()

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_EDGE = 1024

DEBUG_ERRORS = (
    os.getenv("DEBUG_ERRORS", "").lower()
    in {"1", "true", "yes"}
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

FACE_SHAPES = (
    "Oval",
    "Round",
    "Square",
    "Heart",
    "Diamond",
    "Oblong",
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
# ERROR HELPER
# ============================================================

def _detail(
    message: str,
    extra: str = "",
) -> str:

    if extra and DEBUG_ERRORS:
        return f"{message} [{extra}]"

    return message


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.middleware("http")
async def catch_unhandled_errors(
    request: Request,
    call_next,
):

    try:
        return await call_next(request)

    except Exception as exc:

        LOG.exception(
            "UNHANDLED ERROR: %s",
            request.url.path,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": _detail(
                    "Internal server error.",
                    f"{type(exc).__name__}: {exc}",
                ),
            },
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class FaceShapeRequest(BaseModel):
    shape: str


class ChatRequest(BaseModel):
    message: str


class Base64ImageRequest(BaseModel):
    image: str


# ============================================================
# GEMINI CLIENT
# ============================================================

_gemini_client = None
_gemini_error: Optional[str] = None


def get_gemini_client():

    global _gemini_client
    global _gemini_error

    if _gemini_client is not None:
        return _gemini_client

    if _gemini_error:
        raise HTTPException(
            status_code=500,
            detail=_gemini_error,
        )

    if not API_KEY:

        _gemini_error = (
            "GEMINI_API_KEY is not configured. "
            "Please add GEMINI_API_KEY to "
            "Vercel Environment Variables."
        )

        raise HTTPException(
            status_code=500,
            detail=_gemini_error,
        )

    try:

        # Lazy import.
        # This prevents Vercel from loading the
        # Google SDK while detecting the FastAPI app.
        from google import genai

        _gemini_client = genai.Client(
            api_key=API_KEY
        )

        return _gemini_client

    except Exception as exc:

        LOG.exception(
            "Gemini initialization failed"
        )

        _gemini_error = _detail(
            "Gemini SDK could not be initialized.",
            f"{type(exc).__name__}: {exc}",
        )

        raise HTTPException(
            status_code=500,
            detail=_gemini_error,
        )


# ============================================================
# GEMINI RESPONSE TEXT
# ============================================================

def extract_response_text(
    response: Any,
) -> str:

    if response is None:

        raise HTTPException(
            status_code=502,
            detail="Gemini returned no response.",
        )

    # Preferred method
    try:

        text = response.text

        if text and text.strip():
            return text.strip()

    except Exception:
        pass

    # Fallback
    chunks: List[str] = []

    candidates = (
        getattr(
            response,
            "candidates",
            None,
        )
        or []
    )

    for candidate in candidates:

        content = getattr(
            candidate,
            "content",
            None,
        )

        parts = (
            getattr(
                content,
                "parts",
                None,
            )
            or []
        )

        for part in parts:

            text = getattr(
                part,
                "text",
                None,
            )

            if text:
                chunks.append(text)

    result = "".join(chunks).strip()

    if result:
        return result

    raise HTTPException(
        status_code=422,
        detail="Gemini returned an empty response.",
    )


# ============================================================
# GEMINI GENERATION
# ============================================================

def generate_content(
    parts: List[Any],
) -> str:

    client = get_gemini_client()

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=parts,
        )

        return extract_response_text(
            response
        )

    except HTTPException:
        raise

    except Exception as exc:

        LOG.exception(
            "Gemini request failed"
        )

        raise HTTPException(
            status_code=502,
            detail=_detail(
                "Gemini request failed. "
                "Check your API key, model name "
                "and quota.",
                f"{type(exc).__name__}: {exc}",
            ),
        )


# ============================================================
# IMAGE SIZE VALIDATION
# ============================================================

def check_image_size(
    raw: bytes,
) -> None:

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(raw) > MAX_IMAGE_BYTES:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 10 MB."
            ),
        )


# ============================================================
# OPEN IMAGE
# ============================================================

def open_image(
    raw: bytes,
) -> Any:

    try:

        from PIL import Image

    except Exception as exc:

        LOG.exception(
            "Pillow is not installed"
        )

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Server dependency missing: Pillow.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    try:

        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "The uploaded file is not "
                "a valid image."
            ),
        )

    # Convert to RGB for Gemini
    if image.mode != "RGB":

        image = image.convert("RGB")

    # Resize large images
    if max(image.size) > MAX_IMAGE_EDGE:

        image.thumbnail(
            (
                MAX_IMAGE_EDGE,
                MAX_IMAGE_EDGE,
            )
        )

    return image


# ============================================================
# BASE64 DECODER
# ============================================================

def decode_base64_image(
    payload: str,
) -> bytes:

    data = (
        payload or ""
    ).strip()

    # Remove data URI prefix
    if "," in data:

        data = data.split(
            ",",
            1,
        )[1]

    # Remove whitespace
    data = "".join(
        data.split()
    )

    # Support base64url
    data = data.replace(
        "-",
        "+",
    ).replace(
        "_",
        "/",
    )

    # Restore padding
    data += "=" * (
        -len(data) % 4
    )

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Image data is required.",
        )

    # Prevent huge Base64 allocations
    max_base64_length = (
        (MAX_IMAGE_BYTES // 3 + 1)
        * 4
    )

    if len(data) > max_base64_length:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 10 MB."
            ),
        )

    try:

        image_bytes = base64.b64decode(
            data,
            validate=True,
        )

    except (
        binascii.Error,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 image data.",
        )

    check_image_size(
        image_bytes
    )

    return image_bytes


# ============================================================
# MULTIPART UPLOAD
# ============================================================

async def read_upload(
    request: Request,
    field: str = "file",
):

    try:

        form = await request.form()

    except Exception as exc:

        LOG.exception(
            "Multipart form parsing failed"
        )

        raise HTTPException(
            status_code=400,
            detail=_detail(
                "Could not read the uploaded "
                "form data. Send the image as "
                "multipart/form-data using a "
                "'file' field.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    upload = form.get(field)

    # Fallback: find first file
    if not hasattr(
        upload,
        "read",
    ):

        upload = next(
            (
                value
                for value in form.values()
                if hasattr(value, "read")
            ),
            None,
        )

    if upload is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "No image was uploaded. "
                "Use a 'file' field."
            ),
        )

    return upload


# ============================================================
# PROMPTS
# ============================================================

FACE_ANALYSIS_PROMPT = """
You are WeAR AI, a specialized eyeglass
fashion assistant.

Analyze the person's face in the provided image.

Determine the most likely face shape.

Possible face shapes:

- Oval
- Round
- Square
- Heart
- Diamond
- Oblong

Then recommend suitable eyeglass frame
styles for that face shape.

Do not identify the person.
Do not provide personal identity information.

Only analyze visible facial proportions
relevant to eyeglass styling.

Keep the glasses recommendation to
15 words or less.

Return ONLY this format:

Your Face Shape Is: [Detected Shape]
WeAR AI's Suggestion: [Short glasses recommendation]
"""


WEBCAM_ANALYSIS_PROMPT = """
You are WeAR AI, an eyeglass fashion assistant.

Analyze the visible face and determine
the most likely face shape.

Choose from:

Oval
Round
Square
Heart
Diamond
Oblong

Then recommend suitable eyeglass
frame styles.

Do not identify the person.

Return ONLY:

Your Face Shape Is: [Detected Shape]
WeAR AI's Suggestion: [Short recommendation]

Keep the recommendation to 15 words or less.
"""


SUGGESTION_PROMPT = """
You are WeAR AI, a concise eyeglass
fashion assistant.

The user's face shape is:

{shape}

Recommend the most suitable eyeglass
frame styles.

Keep the recommendation to 15 words or less.

Return ONLY:

WeAR AI's Suggestion: [Your recommendation]
"""


CHAT_SYSTEM_INSTRUCTION = """
You are WeAR AI, a specialized AI fashion
assistant for an application called
WeAR Galaxy.

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

You must politely refuse questions unrelated
to eyeglasses.

For unrelated questions, say:

"I am the WeAR AI assistant and my expertise
is limited to eyeglass frames. How can I help
you with glasses today?"

Do not claim to diagnose medical conditions.

Keep answers useful and reasonably concise.
"""


# ============================================================
# ROOT
# ============================================================

@app.get("/")
async def root():

    return {
        "status": "online",
        "application": "WeAR Galaxy AI",
        "message": (
            "WeAR Galaxy AI API is running."
        ),
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health():

    return {
        "status": "healthy",
        "gemini_configured": bool(
            API_KEY
        ),
        "model": GEMINI_MODEL,
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze_image(
    request: Request,
):

    try:

        upload = await read_upload(
            request
        )

        content_type = (
            getattr(
                upload,
                "content_type",
                None,
            )
            or ""
        )

        content_type = (
            content_type
            .split(";")[0]
            .strip()
            .lower()
        )

        if not content_type:

            raise HTTPException(
                status_code=400,
                detail="No file type was provided.",
            )

        if (
            content_type
            not in ALLOWED_IMAGE_TYPES
        ):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image type. "
                    "Please upload JPG, PNG, "
                    "or WEBP."
                ),
            )

        image_bytes = await upload.read()

        check_image_size(
            image_bytes
        )

        image = open_image(
            image_bytes
        )

        result = generate_content(
            [
                FACE_ANALYSIS_PROMPT,
                image,
            ]
        )

        return {
            "success": True,
            "filename": getattr(
                upload,
                "filename",
                None,
            ),
            "analysis": result,
        }

    except HTTPException:
        raise

    except Exception as exc:

        LOG.exception(
            "IMAGE ANALYSIS ERROR"
        )

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Image analysis failed.",
                f"{type(exc).__name__}: {exc}",
            ),
        )


# ============================================================
# MANUAL FACE SHAPE SUGGESTION
# ============================================================

@app.post("/api/suggestion")
async def manual_suggestion(
    request: FaceShapeRequest,
):

    try:

        shape = (
            request.shape or ""
        ).strip()

        if not shape:

            raise HTTPException(
                status_code=400,
                detail="Face shape is required.",
            )

        matched_shape = next(
            (
                candidate
                for candidate in FACE_SHAPES
                if candidate.lower()
                == shape.lower()
            ),
            None,
        )

        if not matched_shape:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid face shape. "
                    "Choose Oval, Round, Square, "
                    "Heart, Diamond, or Oblong."
                ),
            )

        recommendation = generate_content(
            [
                SUGGESTION_PROMPT.format(
                    shape=matched_shape
                )
            ]
        )

        return {
            "success": True,
            "face_shape": matched_shape,
            "recommendation": recommendation,
        }

    except HTTPException:
        raise

    except Exception as exc:

        LOG.exception(
            "SUGGESTION ERROR"
        )

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Recommendation failed.",
                f"{type(exc).__name__}: {exc}",
            ),
        )


# ============================================================
# CHATBOT
# ============================================================

@app.post("/api/chat")
async def chatbot(
    request: ChatRequest,
):

    try:

        message = (
            request.message or ""
        ).strip()

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

        prompt = (
            f"{CHAT_SYSTEM_INSTRUCTION}\n\n"
            "User question:\n\n"
            f"{message}\n\n"
            "Answer the user now."
        )

        answer = generate_content(
            [prompt]
        )

        return {
            "success": True,
            "message": answer,
        }

    except HTTPException:
        raise

    except Exception as exc:

        LOG.exception(
            "CHAT ERROR"
        )

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Chatbot failed.",
                f"{type(exc).__name__}: {exc}",
            ),
        )


# ============================================================
# BASE64 IMAGE ANALYSIS
# ============================================================

@app.post("/api/analyze-base64")
async def analyze_base64(
    request: Base64ImageRequest,
):

    try:

        if not request.image:

            raise HTTPException(
                status_code=400,
                detail="Image data is required.",
            )

        image_bytes = decode_base64_image(
            request.image
        )

        image = open_image(
            image_bytes
        )

        analysis = generate_content(
            [
                WEBCAM_ANALYSIS_PROMPT,
                image,
            ]
        )

        return {
            "success": True,
            "analysis": analysis,
        }

    except HTTPException:
        raise

    except Exception as exc:

        LOG.exception(
            "BASE64 ANALYSIS ERROR"
        )

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Analysis failed.",
                f"{type(exc).__name__}: {exc}",
            ),
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


# ============================================================
# VERCEL EXPORTS
# ============================================================

# Keep these at module level.
# Vercel looks for one of these names.

application = app
handler = app


__all__ = [
    "app",
    "application",
    "handler",
]


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api.index:app",
        host="127.0.0.1",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=True,
    )
```
