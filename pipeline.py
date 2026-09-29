import io
import gc
import soundfile as sf
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from optimum.onnxruntime import ORTModelForSequenceClassification


class AudioDetector:
    def __init__(self, model_name: str = "garystafford/wav2vec2-deepfake-voice-detector"):
        print(f"[ML Engine] Loading ONNX optimized model: {model_name}...")
        self.target_sr = 16000

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        
        # Load ONNX model directly (exports to ONNX format on first run & saves RAM)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_name,
            export=True
        )

        gc.collect()
        self.id2label = self.model.config.id2label
        print(f"[ML Engine] ONNX Model Loaded Successfully! Labels: {self.id2label}")

    def load_audio_from_bytes(self, file_bytes: bytes) -> np.ndarray:
        buffer = io.BytesIO(file_bytes)
        try:
            audio_np, sr = sf.read(buffer, dtype='float32')
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=1)
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
        audio_np = self.load_audio_from_bytes(file_bytes)

        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
        if len(audio_trimmed) > self.target_sr:
            audio_np = audio_trimmed

        # Peak amplitude normalization
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val

        # Crop/pad to 3 seconds
        max_samples = 3 * self.target_sr
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
        elif len(audio_np) < max_samples:
            audio_np = np.pad(audio_np, (0, max_samples - len(audio_np)))

        inputs = self.feature_extractor(
            audio_np, 
            sampling_rate=self.target_sr, 
            return_tensors="pt"
        )

        # Run inference through ONNX runtime
        outputs = self.model(**inputs)
        logits = outputs.logits
        
        # Softmax on numpy / tensor logits
        probs = np.exp(logits.detach().numpy()) / np.sum(np.exp(logits.detach().numpy()), axis=-1, keepdims=True)
        probabilities = probs[0].tolist()

        real_score = probabilities[0]
        fake_score = probabilities[1]
        is_fake = fake_score > 0.5
        confidence = max(real_score, fake_score) * 100

        del inputs, outputs, logits
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
