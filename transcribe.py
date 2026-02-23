#!/usr/bin/env python3
"""Generate .srt subtitle files from Italian videos using Groq's Whisper API."""

import argparse
import os
import sys
import math
from groq import Groq


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_to_srt(video_path: str, output_path: str | None = None) -> str:
    """Transcribe an Italian video and return the .srt content."""
    client = Groq()

    with open(video_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(video_path), f),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language="it",
            temperature=0.0,
        )

    segments = transcription.segments
    if not segments:
        print("No segments found in transcription.", file=sys.stderr)
        sys.exit(1)

    srt_lines = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    srt_content = "\n".join(srt_lines)

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = base + ".srt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"SRT file saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate .srt subtitles from an Italian video using Groq Whisper."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "-o", "--output", help="Output .srt file path (default: same name as video)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    transcribe_to_srt(args.video, args.output)


if __name__ == "__main__":
    main()
