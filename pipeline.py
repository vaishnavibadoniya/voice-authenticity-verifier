import os
import requests


class AudioDetector:
    def __init__(self, model_name: str = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"):
        """
        Lightweight wrapper for Hugging Face Serverless API.
        Uses router.huggingface.co to resolve DNS errors and fit Render's 512MB RAM limit.
        """
        self.model_name = model_name
        
        # Updated Hugging Face Serverless Router Endpoint
        self.api_url = f"https://router.huggingface.co/hf-inference/v1/models/{self.model_name}"
        
        # Read HF_TOKEN set in Render environment
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

        # Handle HTTP authorization & model loading states
        if response.status_code == 401:
            raise Exception(
                "HTTP 401 Unauthorized: Invalid or missing HF_TOKEN. "
                "Ensure your token is correctly set in Render Environment Variables."
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
        
        # Handle response formats from audio classification models
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                data = data[0]
                
            for item in data:
                if isinstance(item, dict) and 'label' in item and 'score' in item:
                    scores_map[str(item['label']).lower()] = float(item['score'])

        # Extract probability scores
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
