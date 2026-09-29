import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pipeline import AudioDetector

app = FastAPI(title="Voice Authenticity Verifier API")

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

detector = None

@app.on_event("startup")
def load_model():
    global detector
    detector = AudioDetector()

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    static_index = os.path.join("static", "index.html")
    if os.path.exists(static_index):
        return FileResponse(static_index)
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return '{"status": "online", "message": "Voice Authenticity Verifier API is running."}'

@app.post("/api/analyze")
@app.post("/predict")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400, 
            detail="Unsupported file format. Please upload WAV, MP3, M4A, FLAC, or OGG."
        )

    try:
        file_bytes = await file.read()
        result = detector.preprocess_and_predict(file_bytes, filename=file.filename)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
