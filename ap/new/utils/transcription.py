import os
import whisper

# Ensure ffmpeg path is explicitly added (adjust if your ffmpeg is elsewhere)
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

model = whisper.load_model("base")

def transcribe_audio(file_path):
    """
    Transcribes audio or video file to text using Whisper.
    Supports paths to saved files (use .getbuffer() in Streamlit before calling this).
    """
    result = model.transcribe(file_path)
    return result["text"]
