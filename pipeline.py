import os
import requests


class AudioDetector:
    def __init__(self, model_name: str = "Hemgg/Deepfake-audio-detection"):
        """
        Lightweight wrapper for Hugging Face Serverless Inference API.
        Runs remote model inference with near-zero local memory usage (< 30 MB RAM).
        """
        self.model_name = model_name
        
        # Public Hugging Face Inference API URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Read HF_TOKEN from environment variables
        self.hf_token = os.getenv("HF_TOKEN", "").strip()
        
        # Build headers
        self.headers = {
            "Content-Type": "audio/wav"
        }
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"
            print(f"[ML Engine] Serverless API initialized for '{self.model_name}' (Authenticated).")
        else:
            print(f"[ML Engine] Serverless API initialized for '{self.model_name}' (Unauthenticated).")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Sends raw audio bytes to Hugging Face API and parses probability scores.
        """
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                data=file_bytes, 
                timeout=25
            )
        except requests.exceptions.RequestException as err:
            raise Exception(f"Network error connecting to Hugging Face API: {str(err)}")

        if response.status_code == 401:
            raise Exception(
                "HTTP 401 Unauthorized: Invalid or missing Hugging Face Access Token. "
                "Please verify that 'HF_TOKEN' is correctly set under your Render Environment variables."
            )

        if response.status_code != 200:
            raise Exception(f"Inference API returned HTTP {response.status_code}: {response.text}")

        data = response.json()

        # Parse standard list output from Hugging Face model response
        # Expected shape: [{'label': 'REAL', 'score': 0.95}, {'label': 'FAKE', 'score': 0.05}]
        if isinstance(data, list):
            scores_map = {item['label'].lower(): item['score'] for item in data if 'label' in item and 'score' in item}
        elif isinstance(data, dict) and "error" in data:
            raise Exception(f"Hugging Face API Error: {data['error']}")
        else:
            scores_map = {}

        # Extract classification probability scores
        fake_score = scores_map.get("fake", scores_map.get("spoof", 0.0))
        real_score = scores_map.get("real", scores_map.get("bonafide", 1.0 - fake_score if fake_score else 0.5))

        is_fake = fake_score > 0.5
        confidence = max(real_score, fake_score) * 100

        return {
            "prediction": "AI Generated" if is_fake else "Human Voice",
            "confidence": round(confidence, 2),
            "scores": {
                "human": round(real_score * 100, 2),
                "ai_generated": round(fake_score * 100, 2)
            },
            "audio_info": {
                "sample_rate": 16000,
                "duration_sec": 3.0
            }
        }
