from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from typing import Literal
from TTS.api import TTS
import numpy as np
import soundfile as sf
import torch
import io
import tempfile
import os
import random
import time
from pathlib import Path

app = FastAPI()


@app.post("/create-audio")
def create_audio(text: str, language: Literal["en", "pt", "es"] = "pt") : 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    timestamp = int(time.time())
    output_file = Path(f'outputs/voice_preview_{timestamp}.wav')
    tts.tts_to_file(
        text=text,
        file_path=output_file,
        speaker_wav="audio_base.wav",
        language=language,
    )

    return f'<audio src="file/{output_file.as_posix()}?{timestamp}" controls autoplay></audio>'

    