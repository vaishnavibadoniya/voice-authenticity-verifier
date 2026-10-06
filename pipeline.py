import io
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class AudioDetector:
    def __init__(self, model_name: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        """
        Loads pre-trained Wav2Vec 2.0 model directly into local RAM.
        Runs smoothly inside Google Cloud Run containers.
        """
        print(f"[ML Engine] Loading model: {model_name}...")
        self.target_sr = 16000
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.id2label = self.model.config.id2label
        print(f"[ML Engine] Model Loaded Successfully! Labels: {self.id2label}")

    def load_audio_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """
        Decodes audio bytes directly into a 16kHz mono NumPy array using C-bindings.
        """
        buffer = io.BytesIO(file_bytes)

        # Reads raw WAV, FLAC, OGG bytes directly in memory
        try:
            audio_np, sr = sf.read(buffer, dtype='float32')
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)  # Convert stereo to mono
            if sr != self.target_sr:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.target_sr)
            return audio_np
        except Exception as e:
            # Fallback for alternative memory formats
            buffer.seek(0)
            try:
                audio_np, _ = librosa.load(buffer, sr=self.target_sr, mono=True)
                return audio_np
            except Exception:
                raise ValueError(f"Could not decode audio stream: {str(e)}")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Preprocesses audio, applies silence trimming, and runs local model inference.
        """
        # 1. Decode audio array from bytes
        audio_np = self.load_audio_from_bytes(file_bytes)

        # 2. Trim silence from leading/trailing edges
        audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
        if len(audio_trimmed) > self.target_sr:
            audio_np = audio_trimmed

        # 3. Peak Amplitude Normalization (-1.0 to 1.0)
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val

        # 4. Standard 5-second Window Crop / Pad (80,000 samples @ 16kHz)
        max_samples = 5 * self.target_sr
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
        elif len(audio_np) < max_samples:
            audio_np = np.pad(audio_np, (0, max_samples - len(audio_np)))

        # 5. Extract Feature Tensors
        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=self.target_sr, 
            return_tensors="pt"
        )

        # 6. Local Model Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()

        # Parse probabilities dynamically based on model config labels
        label2prob = {}
        for idx, prob in enumerate(probabilities):
            label_str = str(self.id2label.get(idx, f"label_{idx}")).lower()
            label2prob[label_str] = prob

        # Extract real vs fake scores safely
        fake_score = label2prob.get("fake", label2prob.get("spoof", label2prob.get("ai", 0.0)))
        real_score = label2prob.get("real", label2prob.get("bonafide", label2prob.get("human", 1.0 - fake_score if fake_score else 0.5)))

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
                "sample_rate": self.target_sr,
                "duration_sec": round(len(audio_np) / self.target_sr, 2)
            }
        }
