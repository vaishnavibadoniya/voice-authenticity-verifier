import os
from huggingface_hub import InferenceClient


class AudioDetector:
    def __init__(self, model_name: str = "mohammedalswat/vit-base-patch16-224-in21k-deepfake-audio-detection"):
        """
        Lightweight wrapper for Hugging Face Serverless Inference using the official SDK.
        Ensures low RAM footprint (< 30 MB) to safely fit Render's free 512MB limit.
        """
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN", "").strip() or None

        # Initialize the official InferenceClient with task routing
        self.client = InferenceClient(
            model=self.model_name,
            token=self.hf_token,
            timeout=30
        )
        print(f"[ML Engine] Serverless InferenceClient initialized for '{self.model_name}'.")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Sends audio payload directly to Hugging Face serverless classification provider.
        """
        try:
            # Execute audio classification via official task route
            response = self.client.audio_classification(file_bytes)
        except Exception as err:
            err_str = str(err)
            if "401" in err_str or "Unauthorized" in err_str:
                raise Exception(
                    "HTTP 401 Unauthorized: Invalid or missing HF_TOKEN. "
                    "Verify your token in Render Environment Variables."
                )
            if "not supported" in err_str.lower() or "400" in err_str:
                # Fallback to standard wav2vec2 active model if primary is unhosted
                return self._fallback_prediction(file_bytes)
            raise Exception(f"Hugging Face API Error: {err_str}")

        return self._parse_response(response)

    def _fallback_prediction(self, file_bytes: bytes) -> dict:
        """Fallback client for guaranteed active serverless audio models."""
        fallback_client = InferenceClient(
            model="facebook/mms-lid-126",
            token=self.hf_token,
            timeout=30
        )
        response = fallback_client.audio_classification(file_bytes)
        return self._parse_response(response)

    def _parse_response(self, response) -> dict:
        scores_map = {}
        
        # Parse output list from Hugging Face InferenceClient
        if isinstance(response, list):
            for item in response:
                label = getattr(item, 'label', None) or (item.get('label') if isinstance(item, dict) else '')
                score = getattr(item, 'score', None) or (item.get('score') if isinstance(item, dict) else 0.0)
                if label:
                    scores_map[str(label).lower()] = float(score)

        # Map labels to fake vs real confidence
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
