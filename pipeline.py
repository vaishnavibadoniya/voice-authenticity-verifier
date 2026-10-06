import os
import requests


class AudioDetector:
    def __init__(self, model_name: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        self.model_name = model_name
        # Updated Hugging Face Serverless Router Endpoint
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.model_name}"
        
        # Read API token from Render environment variables
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
                timeout=30
            )
        except requests.exceptions.RequestException as err:
            raise Exception(f"Network error contacting Hugging Face API: {str(err)}")

        if response.status_code != 200:
            raise Exception(f"Inference API returned HTTP {response.status_code}: {response.text}")

        data = response.json()

        # Parse classification scores from API response
        if isinstance(data, list):
            scores_map = {item['label'].lower(): item['score'] for item in data if 'label' in item and 'score' in item}
        elif isinstance(data, dict) and "error" in data:
            raise Exception(f"HF API Model Error: {data['error']}")
        else:
            scores_map = {}

        # Safely extract scores
        fake_score = scores_map.get("fake", scores_map.get("spoof", scores_map.get("label_1", 0.0)))
        real_score = scores_map.get("real", scores_map.get("bonafide", scores_map.get("label_0", 1.0 - fake_score if fake_score else 0.5)))

        is_fake = fake_score > real_score
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
                "duration_sec": 5.0
            }
        }
