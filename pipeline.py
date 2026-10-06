import os
import requests


class AudioDetector:
    def __init__(self, model_name: str = "Hemgg/Deepfake-audio-detection"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Pull optional HF_TOKEN from environment if present
        self.hf_token = os.getenv("HF_TOKEN", "")
        
        self.headers = {"Content-Type": "audio/wav"}
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"
            
        print(f"[ML Engine] Serverless API client initialized for: {self.model_name}")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                data=file_bytes, 
                timeout=25
            )
        except requests.exceptions.RequestException as err:
            raise Exception(f"Network error contacting Hugging Face API: {str(err)}")

        if response.status_code != 200:
            raise Exception(f"Inference API returned HTTP {response.status_code}: {response.text}")

        data = response.json()

        # Parse output array from Hugging Face response
        if isinstance(data, list):
            scores_map = {item['label'].lower(): item['score'] for item in data if 'label' in item and 'score' in item}
        elif isinstance(data, dict) and "error" in data:
            raise Exception(f"HF API Model Error: {data['error']}")
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
