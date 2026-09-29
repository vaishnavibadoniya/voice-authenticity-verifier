import io
import gc
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class AudioDetector:
    def __init__(self, model_name: str = "Hemgg/Deepfake-audio-detection"):
        """
        Loads a lightweight audio fake classification model optimized for low-RAM cloud instances (<512MB).
        """
        print(f"[ML Engine] Loading lightweight model: {model_name}...")
        self.target_sr = 16000
        
        # Disable gradient calculations globally
        torch.set_grad_enabled(False)

        # Load feature extractor and model
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        # Reclaim RAM immediately
        gc.collect()

        self.id2label = self.model.config.id2label
        print(f"[ML Engine] Model Loaded Successfully! Labels: {self.id2label}")

    def load_audio_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """
        Decodes raw input audio bytes into a 16kHz mono NumPy array.
        """
        buffer = io.BytesIO(file_bytes)

        try:
            audio_np, sr = sf.read(buffer, dtype='float32')
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)  # Stereo to mono
            if sr != self.target_sr:
                audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=self.target_sr)
            return audio_np
        except Exception as e:
            buffer.seek(0)
            try:
                audio_np, _ = librosa.load(buffer, sr=self.target_sr, mono=True)
                return audio_np
            except Exception:
                raise ValueError(f"Could not decode audio stream: {str(e)}")

    def preprocess_and_predict(self, file_bytes: bytes, filename: str = "audio.wav") -> dict:
        """
        Preprocesses incoming audio array and performs probability inference.
        """
        # 1. Decode audio
        audio_np = self.load_audio_from_bytes(file_bytes)

        # 2. Trim silence from audio
        audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
        if len(audio_trimmed) > self.target_sr:
            audio_np = audio_trimmed

        # 3. Peak amplitude normalization
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val

        # 4. Limit audio window to maximum 3 seconds to keep memory usage low during forward pass
        max_samples = 3 * self.target_sr
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
        elif len(audio_np) < max_samples:
            audio_np = np.pad(audio_np, (0, max_samples - len(audio_np)))

        # 5. Feature extraction
        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=self.target_sr, 
            return_tensors="pt"
        )

        # 6. Perform model inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()

        # Parse probability outputs according to label mapping
        real_score = probabilities[0]
        fake_score = probabilities[1]

        is_fake = fake_score > 0.5
        confidence = max(real_score, fake_score) * 100

        # Cleanup memory
        del inputs, logits
        gc.collect()

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
