from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from WeAR Galaxy"}

@app.get("/api/health")
def health():
    return {"status": "ok"}
