import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from pathlib import Path


class WhisperLarge:
    def __init__(self):
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/whisper-large-v3"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                )
        self.model.to(self.device)

        self.preprocessor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.preprocessor.tokenizer,
                feature_extractor=self.preprocessor.feature_extractor,
                torch_dtype=self.torch_dtype,
                device=self.device,
                return_timestamps=True,
                )

    def transcribe(self, filepath: Path):
        return self.pipe(str(filepath))
