import os
import requests

class AudioDetector:
    def __init__(self, model_name: str = "Hemgg/Deepfake-audio-detection"):
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Optional: set HF_TOKEN in Render Environment Variables for higher rate limits
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
        print(f"[ML Engine] Serverless API client initialized for: {model_name}")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Sends audio payload to Hugging Face Serverless API and processes prediction.
        """
        response = requests.post(self.api_url, headers=self.headers, data=file_bytes)

        if response.status_code != 200:
            raise Exception(f"Inference API error ({response.status_code}): {response.text}")

        data = response.json()

        # Parse probability outputs
        scores_map = {item['label'].lower(): item['score'] for item in data}
        
        fake_score = scores_map.get("fake", scores_map.get("spoof", 0.0))
        real_score = scores_map.get("real", scores_map.get("bonafide", 1.0 - fake_score))

        is_fake = fake_score > 0.5
        confidence = max(real_score, fake_score) * 100

        return {
            "prediction": "AI Generated" if is_fake else "Human Voice",
            "confidence": round(confidence, 2),
            "scores": {
                "human": round(real_score * 100, 2),
                "ai_generated": round(fake_score * 100, 2)
            }
        }
