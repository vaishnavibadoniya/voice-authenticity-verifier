<<<<<<< HEAD
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pipeline import AudioDetector

# Initialize FastAPI Application
app = FastAPI(title="Voice Authenticity Analyzer")

# Mount static folder for CSS/JS assets and index.html frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate ML Detector (loads model weights into RAM once at startup)
detector = AudioDetector()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serves the web application interface at http://0.0.0.0:7860
    """
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="File static/index.html not found.")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Receives uploaded audio files or live recording blobs, runs preprocessing,
    and executes neural network inference.
    """
    valid_extensions = ('.wav', '.m4a', '.mp3', '.flac', '.ogg', '.webm')
    filename = file.filename or "audio.wav"

    if not filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed formats: {', '.join(valid_extensions)}"
        )

    try:
        # Read raw binary stream from request RAM
        file_bytes = await file.read()
        
        # Execute model inference via pipeline.py
        result = detector.preprocess_and_predict(file_bytes, filename=filename)
        
        return {
            "success": True,
            "filename": filename,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    # Standard launch configuration matching Docker port 7860
    port = int(os.environ.get("PORT", 7860))
=======
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pipeline import AudioDetector

# Initialize FastAPI Application
app = FastAPI(title="Voice Authenticity Analyzer")

# Mount static folder for CSS/JS assets and index.html frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

# Instantiate ML Detector (loads model weights into RAM once at startup)
detector = AudioDetector()


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """
    Serves the web application interface at http://0.0.0.0:7860
    """
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="File static/index.html not found.")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Receives uploaded audio files or live recording blobs, runs preprocessing,
    and executes neural network inference.
    """
    valid_extensions = ('.wav', '.m4a', '.mp3', '.flac', '.ogg', '.webm')
    filename = file.filename or "audio.wav"

    if not filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed formats: {', '.join(valid_extensions)}"
        )

    try:
        # Read raw binary stream from request RAM
        file_bytes = await file.read()
        
        # Execute model inference via pipeline.py
        result = detector.preprocess_and_predict(file_bytes, filename=filename)
        
        return {
            "success": True,
            "filename": filename,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Analysis failed: {str(e)}"
        )


if __name__ == "__main__":
    # Standard launch configuration matching Docker port 7860
    port = int(os.environ.get("PORT", 7860))
>>>>>>> 237f39d72e9d9fb589e02acccf908379494a8313
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
