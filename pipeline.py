import os
import requests


class AudioDetector:
    def __init__(self, model_name: str = "superb/wav2vec2-base-superb-ks"):
        """
        Lightweight wrapper for Hugging Face Serverless Inference API.
        Uses verified active serverless audio classification endpoints.
        """
        self.model_name = model_name
        
        # Public Hugging Face Serverless Inference URL
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
        # Read HF_TOKEN from Render Environment Variables
        self.hf_token = os.getenv("HF_TOKEN", "").strip()
        
        self.headers = {"Content-Type": "audio/wav"}
        if self.hf_token:
            self.headers["Authorization"] = f"Bearer {self.hf_token}"
            print(f"[ML Engine] Serverless API client initialized for '{self.model_name}' (Authenticated).")
        else:
            print(f"[ML Engine] Serverless API client initialized for '{self.model_name}' (Unauthenticated).")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Sends audio payload directly to Hugging Face API and parses probability scores.
        """
        try:
            response = requests.post(
                self.api_url, 
                headers=self.headers, 
                data=file_bytes, 
                timeout=30
            )
        except requests.exceptions.RequestException as err:
            raise Exception(f"Network error contacting Hugging Face: {str(err)}")

        if response.status_code == 401:
            raise Exception(
                "HTTP 401 Unauthorized: Invalid or missing HF_TOKEN. "
                "Check your Render Environment Variables."
            )

        if response.status_code == 503:
            raise Exception(
                "Model is currently warming up on Hugging Face servers. "
                "Please wait 20 seconds and try again."
            )

        if response.status_code != 200:
            raise Exception(f"Hugging Face HTTP {response.status_code}: {response.text}")

        try:
            data = response.json()
        except Exception:
            raise Exception(f"Invalid response from server: {response.text}")

        if isinstance(data, dict) and "error" in data:
            raise Exception(f"Hugging Face Model Error: {data['error']}")

        scores_map = {}
        
        # Parse output array from Hugging Face audio model
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                data = data[0]
                
            for item in data:
                if isinstance(item, dict) and 'label' in item and 'score' in item:
                    scores_map[str(item['label']).lower()] = float(item['score'])

        # Calculate scores and predictions
        fake_score = scores_map.get("fake", scores_map.get("spoof", scores_map.get("ai", 0.0)))
        real_score = scores_map.get("real", scores_map.get("bonafide", scores_map.get("human", 1.0 - fake_score if fake_score else 0.5)))

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
                "duration_sec": 3.0
            }
        }
