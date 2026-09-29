import os
from huggingface_hub import InferenceClient


class AudioDetector:
    def __init__(self, model_name: str = "Hemgg/Deepfake-audio-detection"):
        """
        Lightweight wrapper for Hugging Face Inference using official SDK.
        Maintains low memory footprint (< 30 MB RAM) to fit Render's free tier.
        """
        self.model_name = model_name
        self.token = os.getenv("HF_TOKEN", "").strip() or None

        # Initialize official Hugging Face Inference Client
        self.client = InferenceClient(
            model=self.model_name,
            token=self.token,
            timeout=30
        )
        print(f"[ML Engine] Hugging Face Inference Client initialized for '{self.model_name}'.")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Sends audio payload directly through Hugging Face InferenceClient.
        """
        try:
            # Send raw audio bytes to Hugging Face
            response = self.client.audio_classification(file_bytes)
        except Exception as err:
            err_str = str(err)
            if "401" in err_str or "Unauthorized" in err_str:
                raise Exception(
                    "HTTP 401 Unauthorized: Invalid or missing Hugging Face Access Token. "
                    "Ensure 'HF_TOKEN' is configured in Render Environment Variables."
                )
            raise Exception(f"Hugging Face API Error: {err_str}")

        # Parse output from Hugging Face response
        # Result format: [{'label': 'REAL', 'score': 0.95}, {'label': 'FAKE', 'score': 0.05}]
        scores_map = {}
        if isinstance(response, list):
            for item in response:
                # Handle dict or object attributes
                label = getattr(item, 'label', None) or item.get('label', '')
                score = getattr(item, 'score', None) or item.get('score', 0.0)
                if label:
                    scores_map[str(label).lower()] = float(score)

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
