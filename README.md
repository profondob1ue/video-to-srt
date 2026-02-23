# video-to-srt

Generate `.srt` subtitle files from Italian videos using Groq's Whisper API.

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-api-key"
```

## Usage

```bash
python transcribe.py video.mp4
python transcribe.py video.mp4 -o custom_output.srt
```
