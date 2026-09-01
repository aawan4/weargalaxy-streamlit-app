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
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("wear-galaxy-ai")


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
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
# FASTAPI APPLICATION
# ============================================================

# IMPORTANT:
# Keep this as a top-level variable.
# Vercel detects this object as the application.

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
# GLOBAL ERROR HANDLING
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
            "Unhandled error on %s",
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
# GEMINI STATE
# ============================================================

_GEMINI: Dict[str, Any] = {
    "ready": False,
    "flavour": None,
    "client": None,
    "error": None,
}


# ============================================================
# ERROR MESSAGE HELPER
# ============================================================

def _detail(
    message: str,
    extra: str = "",
) -> str:

    if extra and DEBUG_ERRORS:
        return f"{message} [{extra}]"

    return message


# ============================================================
# LOAD GEMINI
# ============================================================

def _load_gemini() -> Dict[str, Any]:
    """
    Initialize Gemini lazily.

    Nothing from the Google SDK is imported when Vercel
    imports this module. This is important for deployment.
    """

    if (
        _GEMINI["ready"]
        or _GEMINI["error"]
    ):
        return _GEMINI

    if not API_KEY:

        _GEMINI["error"] = (
            "GEMINI_API_KEY is not configured. "
            "Add GEMINI_API_KEY to your Vercel "
            "Environment Variables."
        )

        return _GEMINI

    failures: List[str] = []

    # --------------------------------------------------------
    # Current Google GenAI SDK
    # --------------------------------------------------------

    try:

        from google import genai

        client = genai.Client(
            api_key=API_KEY
        )

        _GEMINI.update(
            {
                "ready": True,
                "flavour": "google-genai",
                "client": client,
                "error": None,
            }
        )

        return _GEMINI

    except Exception as exc:

        failures.append(
            "google-genai: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # Legacy SDK fallback
    # --------------------------------------------------------

    try:

        import google.generativeai as legacy_genai

        legacy_genai.configure(
            api_key=API_KEY
        )

        model = legacy_genai.GenerativeModel(
            GEMINI_MODEL
        )

        _GEMINI.update(
            {
                "ready": True,
                "flavour": "google-generativeai",
                "client": model,
                "error": None,
            }
        )

        return _GEMINI

    except Exception as exc:

        failures.append(
            "google-generativeai: "
            f"{type(exc).__name__}: {exc}"
        )

    LOG.error(
        "Gemini SDK unavailable: %s",
        " | ".join(failures),
    )

    _GEMINI["error"] = _detail(
        "The Gemini SDK could not be initialized. "
        "Check your requirements.txt.",
        " | ".join(failures),
    )

    return _GEMINI


# ============================================================
# GET GEMINI MODEL
# ============================================================

def get_model() -> Dict[str, Any]:

    state = _load_gemini()

    if not state["ready"]:

        raise HTTPException(
            status_code=500,
            detail=(
                state["error"]
                or "Gemini is not available."
            ),
        )

    return state


# ============================================================
# GENERATE CONTENT
# ============================================================

def _generate(
    parts: List[Any],
) -> str:

    state = get_model()

    try:

        if state["flavour"] == "google-genai":

            response = (
                state["client"]
                .models
                .generate_content(
                    model=GEMINI_MODEL,
                    contents=parts,
                )
            )

        else:

            response = (
                state["client"]
                .generate_content(parts)
            )

    except Exception as exc:

        LOG.exception(
            "Gemini request failed"
        )

        raise HTTPException(
            status_code=502,
            detail=_detail(
                "The Gemini request failed. "
                "Check the API key, model name "
                "and quota.",
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            ),
        )

    return _extract_text(
        response
    )


# ============================================================
# EXTRACT GEMINI RESPONSE
# ============================================================

def _extract_text(
    response: Any,
) -> str:

    if response is None:

        raise HTTPException(
            status_code=502,
            detail="Gemini returned no response.",
        )

    text: Optional[str] = None

    try:

        text = response.text

    except Exception:

        text = None

    # --------------------------------------------------------
    # Fallback candidate extraction
    # --------------------------------------------------------

    if not text:

        chunks: List[str] = []

        for candidate in (
            getattr(
                response,
                "candidates",
                None,
            )
            or []
        ):

            content = getattr(
                candidate,
                "content",
                None,
            )

            for part in (
                getattr(
                    content,
                    "parts",
                    None,
                )
                or []
            ):

                piece = getattr(
                    part,
                    "text",
                    None,
                )

                if piece:

                    chunks.append(
                        piece
                    )

        text = "".join(
            chunks
        )

    if not text or not text.strip():

        raise HTTPException(
            status_code=422,
            detail=_blocked_reason(
                response
            ),
        )

    return text.strip()


# ============================================================
# GEMINI EMPTY RESPONSE REASON
# ============================================================

def _blocked_reason(
    response: Any,
) -> str:

    reasons: List[str] = []

    feedback = getattr(
        response,
        "prompt_feedback",
        None,
    )

    block_reason = getattr(
        feedback,
        "block_reason",
        None,
    )

    if block_reason:

        reasons.append(
            str(
                getattr(
                    block_reason,
                    "name",
                    block_reason,
                )
            )
        )

    for candidate in (
        getattr(
            response,
            "candidates",
            None,
        )
        or []
    ):

        finish = getattr(
            candidate,
            "finish_reason",
            None,
        )

        if finish:

            reasons.append(
                str(
                    getattr(
                        finish,
                        "name",
                        finish,
                    )
                )
            )

    if reasons:

        return (
            "Gemini returned no usable text "
            f"(reason: {', '.join(dict.fromkeys(reasons))}). "
            "Try a clear, well-lit photo showing one face."
        )

    return "Gemini returned an empty response."


# ============================================================
# IMAGE HELPERS
# ============================================================

def _check_size(
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


def _open_image(
    raw: bytes,
) -> Any:

    try:

        # Lazy import
        from PIL import Image

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Pillow is not installed. "
                "Add pillow to requirements.txt.",
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

    if image.mode != "RGB":

        image = image.convert(
            "RGB"
        )

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

def _decode_base64_image(
    payload: str,
) -> bytes:

    data = (
        payload or ""
    ).strip()

    # Remove:
    # data:image/png;base64,
    if "," in data:

        data = data.split(
            ",",
            1,
        )[1]

    data = "".join(
        data.split()
    )

    data = data.replace(
        "-",
        "+",
    ).replace(
        "_",
        "/",
    )

    data += "=" * (
        -len(data) % 4
    )

    if not data:

        raise HTTPException(
            status_code=400,
            detail="Image data is required.",
        )

    # 4 Base64 chars = approximately 3 bytes
    if len(data) > (
        (MAX_IMAGE_BYTES // 3 + 1)
        * 4
    ):

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum size is 10 MB."
            ),
        )

    try:

        image_bytes = (
            base64.b64decode(
                data,
                validate=True,
            )
        )

    except (
        binascii.Error,
        ValueError,
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 image data.",
        )

    _check_size(
        image_bytes
    )

    return image_bytes


# ============================================================
# MULTIPART UPLOAD
# ============================================================

async def _read_upload(
    request: Request,
    field: str = "file",
) -> Any:

    try:

        form = await request.form()

    except Exception as exc:

        LOG.exception(
            "Could not parse multipart form"
        )

        raise HTTPException(
            status_code=400,
            detail=_detail(
                "Could not read the uploaded "
                "form data. Send the image as "
                "multipart/form-data in a field "
                "named 'file'.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    upload = form.get(
        field
    )

    # Fallback to first file
    if not hasattr(
        upload,
        "read",
    ):

        upload = next(
            (
                value
                for value in form.values()
                if hasattr(
                    value,
                    "read",
                )
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
# AI PROMPTS
# ============================================================

FACE_ANALYSIS_PROMPT = """
You are WeAR AI, a specialized eyeglass
fashion assistant.

Analyze the person's face in the provided image.

Determine the most likely face shape.

Possible face shapes include:

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

Keep the glasses recommendation to 15 words
or less.

Return ONLY this format:

Your Face Shape Is: [Detected Shape]
WeAR AI's Suggestion: [Short glasses recommendation]
"""


WEBCAM_ANALYSIS_PROMPT = """
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
assistant for an application called WeAR Galaxy.

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
async def root() -> Dict[str, Any]:

    return {
        "status": "online",
        "application": "WeAR Galaxy AI",
        "message": "WeAR Galaxy AI API is running.",
        "version": "1.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
async def health() -> Dict[str, Any]:

    state = _load_gemini()

    return {
        "status": "healthy",
        "gemini_configured": bool(
            API_KEY
        ),
        "gemini_ready": bool(
            state["ready"]
        ),
        "gemini_sdk": state["flavour"],
        "model": GEMINI_MODEL,
        "error": state["error"],
    }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze_image(
    request: Request,
) -> Dict[str, Any]:

    try:

        upload = await _read_upload(
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

        _check_size(
            image_bytes
        )

        image = _open_image(
            image_bytes
        )

        result = _generate(
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
# MANUAL FACE SHAPE RECOMMENDATION
# ============================================================

@app.post("/api/suggestion")
async def manual_suggestion(
    request: FaceShapeRequest,
) -> Dict[str, Any]:

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

        recommendation = _generate(
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
            "MANUAL SUGGESTION ERROR"
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
) -> Dict[str, Any]:

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

        answer = _generate(
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
# BASE64 / WEBCAM ANALYSIS
# ============================================================

@app.post("/api/analyze-base64")
async def analyze_base64(
    request: Base64ImageRequest,
) -> Dict[str, Any]:

    try:

        if not request.image:

            raise HTTPException(
                status_code=400,
                detail="Image data is required.",
            )

        image_bytes = (
            _decode_base64_image(
                request.image
            )
        )

        image = _open_image(
            image_bytes
        )

        analysis = _generate(
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
async def api_info() -> Dict[str, Any]:

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

application = app
handler = app

__all__ = [
    "app",
    "application",
    "handler",
]
```
