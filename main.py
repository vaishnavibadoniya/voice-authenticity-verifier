import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pipeline import AudioDetector

app = FastAPI(
    title="Voice Authenticity Verifier API",
    description="Detects whether an uploaded voice track is authentic human speech or AI-generated deepfake audio."
)

# Enable CORS for browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector instance at startup
detector = None

@app.on_event("startup")
def load_model():
    global detector
    detector = AudioDetector()

@app.get("/")
def health_check():
    return {
        "status": "online",
        "message": "Voice Authenticity Verifier API is running."
    }

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
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
