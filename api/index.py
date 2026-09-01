from fastapi import FastAPI

app = FastAPI(
    title="WeAR Galaxy AI",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "WeAR Galaxy AI API is running"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }
