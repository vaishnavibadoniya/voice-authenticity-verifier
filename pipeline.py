import io
import gc
import torch
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


class AudioDetector:
    def __init__(self, model_name: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        """
        Loads pre-trained Wav2Vec 2.0 model in FP16 precision to fit 512MB RAM constraints.
        """
        print(f"[ML Engine] Loading model in FP16 mode: {model_name}...")
        self.target_sr = 16000
        
        # Disable gradient tracking globally
        torch.set_grad_enabled(False)

        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Load model weights in float16 to reduce RAM footprint by ~50%
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.model.eval()

        # Reclaim unused RAM immediately
        gc.collect()

        self.id2label = self.model.config.id2label
        print(f"[ML Engine] Model Loaded Successfully in FP16! Labels: {self.id2label}")

    def load_audio_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        """
        Decodes raw audio bytes into a 16kHz mono NumPy array.
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
        Preprocesses audio array and computes AI vs Human probability scores.
        """
        # 1. Decode audio
        audio_np = self.load_audio_from_bytes(file_bytes)

        # 2. Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
        if len(audio_trimmed) > self.target_sr:
            audio_np = audio_trimmed

        # 3. Normalize peak amplitude
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val

        # 4. Limit window size to 3 seconds to restrict tensor allocation during pass
        max_samples = 3 * self.target_sr
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

        # 6. Convert input tensors to float16 to match model weight precision
        inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        # 7. Model Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0].tolist()

        real_score = probabilities[0]
        fake_score = probabilities[1]

        is_fake = fake_score > 0.5
        confidence = max(real_score, fake_score) * 100

        # 8. Clean up intermediate tensors
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
