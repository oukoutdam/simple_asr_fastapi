from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

from fastapi.responses import HTMLResponse

from datetime import datetime
from pathlib import Path
import os
import shutil

from asr.whisper import WhisperLarge

app = FastAPI()

UPLOAD_DIR = Path("upload")
UPLOAD_DIR.mkdir(exist_ok=True)

asr_model = WhisperLarge()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/upload")
async def upload_audio(audio: UploadFile = File(...)):
    if not audio.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are allowed")

    now = datetime.now()
    file_path = UPLOAD_DIR / f"{now.strftime('%Y%m%d_%H%M_')}{audio.filename}"

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    finally:
        audio.file.close()

    result = asr_model.transcribe(file_path)
    # print(result["chunks"][0].text)

    line_list = [f"{chunk['timestamp'][0]}_{chunk['timestamp'][1]}: {chunk['text']}" for chunk in result["chunks"]]
    transcription = "\n".join(line_list)

    return transcription
