# WeAR Galaxy AI — Vercel Version

This is the Vercel-ready static frontend for WeAR Galaxy AI.

## Structure

```text
your-project/
├── api/
│   └── index.py
├── index.html
├── styles.css
├── script.js
└── requirements.txt
```

Keep your working `api/index.py` and `requirements.txt`.

## Vercel

No Streamlit is required.

The browser frontend calls these same-origin API routes:

- POST `/api/analyze`
- POST `/api/analyze-base64`
- POST `/api/suggestion`
- POST `/api/chat`
- GET `/api/health`

## Gemini

Add this Vercel Environment Variable:

`GEMINI_API_KEY`

Then redeploy.

## Webcam

Webcam access requires a secure HTTPS deployment. Vercel provides HTTPS automatically.
