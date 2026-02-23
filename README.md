# video-to-srt

Generate `.srt` subtitle files from videos using Groq's Whisper API.

Handles large video files automatically by extracting audio with ffmpeg and splitting into chunks when needed to stay within API limits.

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and available in PATH
- A [Groq API key](https://console.groq.com/)

## Setup

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-groq-api-key"
```

## Usage

```bash
python transcribe.py video.mp4
python transcribe.py video.mp4 -o custom_output.srt
python transcribe.py video.mp4 -l en   # English instead of default Italian
```

The script will:
1. Extract audio from the video as a compressed mono MP3
2. Split the audio into chunks if it exceeds the 25 MB API limit
3. Transcribe each chunk via Groq's Whisper Large V3 Turbo
4. Merge all segments into a single `.srt` file with correct timestamps
