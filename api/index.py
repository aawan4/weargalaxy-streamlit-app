```python
from fastapi import FastAPI


# This MUST remain at the top level.
app = FastAPI(
    title="WeAR Galaxy AI",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "status": "online",
        "application": "WeAR Galaxy AI",
        "message": "WeAR Galaxy AI API is running.",
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
    }


# Vercel exports
application = app
handler = app
```
