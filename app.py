"""
WeAR Galaxy AI - FastAPI backend, Vercel-ready.

WHY THE VERCEL ERROR HAPPENED
-----------------------------
Vercel's Python runtime imports this module and then looks for a module-level
`app`, `application` or `handler`. If *anything* raises while the module is
being imported, that lookup never succeeds and Vercel reports:

    Found app.py but it does not export a top-level "app", "application",
    or "handler" variable.

In the original file four things ran at import time and could each raise:

  * `@app.post("/api/analyze")` declared `UploadFile = File(...)`. FastAPI
    validates that at *decoration* time and raises RuntimeError when the
    `python-multipart` package is missing. This is the most common cause.
  * `import google.generativeai` fails when the dependency is missing, or when
    requirements.txt pins the newer `google-genai` package instead.
  * `import PIL` fails when `pillow` is missing.
  * `genai.configure()` / `GenerativeModel()` can raise on a bad credential.

THE FIX
-------
Only FastAPI, Pydantic and the standard library are imported at module scope,
and every fragile operation is deferred to request time. Importing this file
cannot fail, so `app` is always exported. All original endpoints, request
shapes, response shapes, prompts and validation rules are preserved.
"""

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

# Read at import time, but never *used* at import time.
API_KEY: str = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or ""
).strip()

GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# Maximum accepted upload size (bytes). 10 MB, same as the original.
MAX_IMAGE_BYTES: int = 10 * 1024 * 1024

# Longest edge sent to Gemini. Keeps serverless payloads and latency small;
# face-shape detection does not need more than this.
MAX_IMAGE_EDGE: int = 1024

# Set DEBUG_ERRORS=1 in Vercel to include exception details in HTTP responses.
DEBUG_ERRORS: bool = os.getenv("DEBUG_ERRORS", "").lower() in {"1", "true", "yes"}

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# Ordered so validation errors and prompts stay deterministic.
FACE_SHAPES = ("Oval", "Round", "Square", "Heart", "Diamond", "Oblong")


# ============================================================
# FASTAPI APPLICATION
# ============================================================
#
# Declared as a plain module-level assignment so both Vercel's static scan and
# its runtime `getattr(module, "app")` lookup find it.

app = FastAPI(
    title="WeAR Galaxy AI API",
    description="AI-powered glasses style advisor",
    version="1.0.0",
)


# ============================================================
# CATCH-ALL ERROR HANDLER
# ============================================================
#
# Middleware rather than @app.exception_handler(Exception): Starlette re-raises
# after an Exception handler runs, and a serverless adapter can turn that into
# an opaque platform error. Swallowing it here means the client always gets
# JSON. Registered *before* CORS so that CORS ends up the outer layer and even
# a 500 carries Access-Control-Allow-Origin.

@app.middleware("http")
async def catch_unhandled_errors(request: Request, call_next):

    try:
        return await call_next(request)

    except Exception as exc:
        LOG.exception("UNHANDLED ERROR on %s", request.url.path)

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
# REQUEST MODELS
# ============================================================

class FaceShapeRequest(BaseModel):
    shape: str


class ChatRequest(BaseModel):
    message: str


class Base64ImageRequest(BaseModel):
    image: str


# ============================================================
# GEMINI CLIENT (lazy, SDK-agnostic)
# ============================================================
#
# Supports both Google SDKs and picks whichever is installed:
#   * "google-genai"          -> from google import genai; genai.Client(...)
#   * "google-generativeai"   -> import google.generativeai; GenerativeModel(...)
#
# Nothing here runs until the first request touches Gemini.

_GEMINI: Dict[str, Any] = {
    "ready": False,
    "flavour": None,
    "client": None,
    "error": None,
}


def _load_gemini() -> Dict[str, Any]:
    """Initialise the Gemini client once per warm instance."""

    if _GEMINI["ready"] or _GEMINI["error"]:
        return _GEMINI

    if not API_KEY:
        _GEMINI["error"] = (
            "GEMINI_API_KEY is not configured. "
            "Add GEMINI_API_KEY to your Vercel Environment Variables."
        )
        return _GEMINI

    failures: List[str] = []

    # Preferred: the current SDK.
    try:
        from google import genai as google_genai

        _GEMINI.update(
            flavour="google-genai",
            client=google_genai.Client(api_key=API_KEY),
            ready=True,
        )
        return _GEMINI
    except Exception as exc:
        failures.append(f"google-genai: {type(exc).__name__}: {exc}")

    # Fallback: the legacy SDK the original code used.
    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=API_KEY)

        _GEMINI.update(
            flavour="google-generativeai",
            client=legacy_genai.GenerativeModel(GEMINI_MODEL),
            ready=True,
        )
        return _GEMINI
    except Exception as exc:
        failures.append(f"google-generativeai: {type(exc).__name__}: {exc}")

    LOG.error("Gemini SDK unavailable: %s", " | ".join(failures))

    _GEMINI["error"] = _detail(
        "The Gemini SDK could not be initialised on the server. "
        "Add 'google-genai' to requirements.txt.",
        " | ".join(failures),
    )
    return _GEMINI


def _detail(message: str, extra: str = "") -> str:
    """Append internal diagnostics only when DEBUG_ERRORS is enabled."""

    if extra and DEBUG_ERRORS:
        return f"{message} [{extra}]"
    return message


def get_model() -> Dict[str, Any]:
    """
    Return a ready Gemini client bundle.

    Raises a clean API error if the Gemini key has not been configured or the
    SDK is missing. Kept as a named helper so existing call sites still read
    the same way.
    """

    state = _load_gemini()

    if not state["ready"]:
        raise HTTPException(
            status_code=500,
            detail=state["error"] or "Gemini is not available.",
        )

    return state


def _generate(parts: List[Any]) -> str:
    """Send `parts` (text and/or PIL images) to Gemini and return its text."""

    state = get_model()

    try:
        if state["flavour"] == "google-genai":
            response = state["client"].models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
            )
        else:
            response = state["client"].generate_content(parts)

    except Exception as exc:
        LOG.exception("Gemini request failed")
        raise HTTPException(
            status_code=502,
            detail=_detail(
                "The Gemini request failed. Check the API key, the model name "
                f"('{GEMINI_MODEL}') and your quota.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    return _extract_text(response)


def _extract_text(response: Any) -> str:
    """
    Pull text out of a Gemini response.

    `response.text` raises (rather than returning empty) when a candidate has
    no parts, e.g. when a safety filter blocked the image. The original code
    would surface that as an opaque 500, so unwrap it properly here.
    """

    if response is None:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned no response.",
        )

    text: Optional[str]
    try:
        text = response.text
    except Exception:
        text = None

    if not text:
        chunks: List[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                piece = getattr(part, "text", None)
                if piece:
                    chunks.append(piece)
        text = "".join(chunks)

    if not text or not text.strip():
        raise HTTPException(
            status_code=422,
            detail=_blocked_reason(response),
        )

    return text.strip()


def _blocked_reason(response: Any) -> str:
    """Build a human-readable message for an empty Gemini response."""

    reasons: List[str] = []

    feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason:
        reasons.append(str(getattr(block_reason, "name", block_reason)))

    for candidate in getattr(response, "candidates", None) or []:
        finish = getattr(candidate, "finish_reason", None)
        if finish:
            reasons.append(str(getattr(finish, "name", finish)))

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

def _open_image(raw: bytes) -> Any:
    """Decode bytes into an RGB PIL image, downscaled to MAX_IMAGE_EDGE."""

    try:
        from PIL import Image
    except Exception as exc:
        LOG.exception("Pillow is not installed")
        raise HTTPException(
            status_code=500,
            detail=_detail(
                "Server dependency missing: Pillow. "
                "Add 'pillow' to requirements.txt.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    if image.mode != "RGB":
        image = image.convert("RGB")

    if max(image.size) > MAX_IMAGE_EDGE:
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))

    return image


def _check_size(raw: bytes) -> None:
    """Reject empty or oversized payloads."""

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )


def _decode_base64_image(payload: str) -> bytes:
    """Decode a raw or data-URI Base64 string into image bytes."""

    data = (payload or "").strip()

    # Strip a "data:image/png;base64," style prefix.
    if "," in data:
        data = data.split(",", 1)[1]

    # Browsers and JSON encoders often introduce newlines; base64url is common
    # too, and canvas.toDataURL output is sometimes missing its padding.
    data = "".join(data.split())
    data = data.replace("-", "+").replace("_", "/")
    data += "=" * (-len(data) % 4)

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Image data is required.",
        )

    # Reject before allocating: 4 Base64 chars decode to 3 bytes.
    if len(data) > (MAX_IMAGE_BYTES // 3 + 1) * 4:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Maximum size is 10 MB.",
        )

    try:
        image_bytes = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 image data.",
        )

    _check_size(image_bytes)
    return image_bytes


# ============================================================
# MULTIPART HELPER
# ============================================================
#
# The upload endpoint reads the form inside the handler instead of declaring
# `file: UploadFile = File(...)`. Starlette only needs `python-multipart` when
# `request.form()` is actually called, so a missing dependency now produces a
# clear 500 on one endpoint instead of breaking the whole module import.

async def _read_upload(request: Request, field: str = "file") -> Any:
    """Return the uploaded file object from a multipart request."""

    try:
        form = await request.form()
    except Exception as exc:
        LOG.exception("Could not parse multipart form data")
        raise HTTPException(
            status_code=400,
            detail=_detail(
                "Could not read the uploaded form data. Send the image as "
                "multipart/form-data in a field named 'file'. If this keeps "
                "happening, add 'python-multipart' to requirements.txt.",
                f"{type(exc).__name__}: {exc}",
            ),
        )

    upload = form.get(field)

    # Fall back to the first file-like value under any field name.
    if not hasattr(upload, "read"):
        upload = next(
            (value for value in form.values() if hasattr(value, "read")),
            None,
        )

    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="No image was uploaded. Use a 'file' field.",
        )

    return upload


# ============================================================
# PROMPTS
# ============================================================

FACE_ANALYSIS_PROMPT = """
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
You are WeAR AI, a concise eyeglass fashion assistant.

The user's face shape is:

{shape}

Recommend the most suitable eyeglass frame styles.

Keep the recommendation to 15 words or less.

Return ONLY:

WeAR AI's Suggestion: [Your recommendation]
"""

CHAT_SYSTEM_INSTRUCTION = """
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


# ============================================================
# ROOT ENDPOINT
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
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
async def health() -> Dict[str, Any]:

    state = _load_gemini()

    return {
        "status": "healthy",
        "gemini_configured": bool(API_KEY),
        "gemini_ready": bool(state["ready"]),
        "gemini_sdk": state["flavour"],
        "model": GEMINI_MODEL,
        "error": state["error"],
    }


# ============================================================
# FACE SHAPE ANALYSIS (multipart upload)
# ============================================================

@app.post("/api/analyze")
async def analyze_image(request: Request) -> Dict[str, Any]:
    """
    Analyze an uploaded image using Gemini.

    Expected input:
        multipart/form-data
        file = image
    """

    try:
        upload = await _read_upload(request)

        # ----------------------------------------------------
        # Validate file type
        # ----------------------------------------------------

        content_type = (getattr(upload, "content_type", None) or "").split(";")[0]
        content_type = content_type.strip().lower()

        if not content_type:
            raise HTTPException(
                status_code=400,
                detail="No file type was provided.",
            )

        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid image type. "
                    "Please upload JPG, PNG, or WEBP."
                ),
            )

        # ----------------------------------------------------
        # Read, size-check and decode
        # ----------------------------------------------------

        image_bytes = await upload.read()

        _check_size(image_bytes)
        image = _open_image(image_bytes)

        # ----------------------------------------------------
        # Gemini request
        # ----------------------------------------------------

        result = _generate([FACE_ANALYSIS_PROMPT, image])

        return {
            "success": True,
            "filename": getattr(upload, "filename", None),
            "analysis": result,
        }

    except HTTPException:
        raise

    except Exception as exc:
        LOG.exception("IMAGE ANALYSIS ERROR")
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
async def manual_suggestion(request: FaceShapeRequest) -> Dict[str, Any]:
    """
    Generate glasses recommendations
    based on manually selected face shape.
    """

    try:
        shape = (request.shape or "").strip()

        if not shape:
            raise HTTPException(
                status_code=400,
                detail="Face shape is required.",
            )

        # Case-insensitive validation, canonical capitalisation preserved.
        matched_shape = next(
            (
                candidate
                for candidate in FACE_SHAPES
                if candidate.lower() == shape.lower()
            ),
            None,
        )

        if not matched_shape:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Invalid face shape. "
                    "Choose Oval, Round, Square, Heart, "
                    "Diamond, or Oblong."
                ),
            )

        recommendation = _generate(
            [SUGGESTION_PROMPT.format(shape=matched_shape)]
        )

        return {
            "success": True,
            "face_shape": matched_shape,
            "recommendation": recommendation,
        }

    except HTTPException:
        raise

    except Exception as exc:
        LOG.exception("MANUAL SUGGESTION ERROR")
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
async def chatbot(request: ChatRequest) -> Dict[str, Any]:
    """
    WeAR AI glasses chatbot.
    """

    try:
        message = (request.message or "").strip()

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
            "Answer the user now.\n"
        )

        answer = _generate([prompt])

        return {
            "success": True,
            "message": answer,
        }

    except HTTPException:
        raise

    except Exception as exc:
        LOG.exception("CHAT ERROR")
        raise HTTPException(
            status_code=500,
            detail=_detail("Chatbot failed.", f"{type(exc).__name__}: {exc}"),
        )


# ============================================================
# BASE64 IMAGE ANALYSIS
# ============================================================

@app.post("/api/analyze-base64")
async def analyze_base64(request: Base64ImageRequest) -> Dict[str, Any]:
    """
    Analyze a Base64 encoded image.

    Useful for webcam images captured
    directly in browser JavaScript.
    """

    try:
        if not request.image:
            raise HTTPException(
                status_code=400,
                detail="Image data is required.",
            )

        image_bytes = _decode_base64_image(request.image)
        image = _open_image(image_bytes)

        analysis = _generate([WEBCAM_ANALYSIS_PROMPT, image])

        return {
            "success": True,
            "analysis": analysis,
        }

    except HTTPException:
        raise

    except Exception as exc:
        LOG.exception("BASE64 ANALYSIS ERROR")
        raise HTTPException(
            status_code=500,
            detail=_detail("Analysis failed.", f"{type(exc).__name__}: {exc}"),
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
# VERCEL ENTRYPOINT EXPORTS
# ============================================================
#
# Vercel accepts any one of these three names. Exporting all three as simple
# module-level assignments satisfies every version of the Python runtime, and
# keeps working if you later move this file to api/index.py.

application = app
handler = app

__all__ = ["app", "application", "handler"]


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
