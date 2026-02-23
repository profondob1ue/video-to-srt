#!/usr/bin/env python3
"""Generate .srt subtitle files from Italian videos using Groq's Whisper API.

Handles large video files by extracting audio with ffmpeg and splitting
into chunks when the file exceeds Groq's 25 MB upload limit.
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

from groq import Groq

MAX_FILE_SIZE = 24 * 1024 * 1024  # 24 MB (safe margin under 25 MB limit)
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus"}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def get_duration(file_path: str) -> float:
    """Get the duration of an audio/video file in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            file_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from a video file as a compressed mono MP3."""
    print(f"Extracting audio from: {video_path}")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
            output_path,
        ],
        capture_output=True, check=True,
    )
    print(f"Audio extracted to: {output_path}")
    return output_path


def split_audio(audio_path: str, chunk_dir: str, chunk_duration: int) -> list[str]:
    """Split an audio file into fixed-duration chunks."""
    print(f"Splitting audio into ~{chunk_duration}s chunks...")
    total_duration = get_duration(audio_path)
    chunks = []
    start = 0
    idx = 0

    while start < total_duration:
        chunk_path = os.path.join(chunk_dir, f"chunk_{idx:04d}.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(chunk_duration),
                "-ac", "1", "-ar", "16000", "-b:a", "48k",
                chunk_path,
            ],
            capture_output=True, check=True,
        )
        chunks.append((chunk_path, start))
        start += chunk_duration
        idx += 1

    print(f"Created {len(chunks)} chunk(s)")
    return chunks


def transcribe_file(client: Groq, audio_path: str, language: str) -> list[dict]:
    """Transcribe a single audio file and return its segments."""
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=language,
            temperature=0.0,
        )
    return transcription.segments or []


def transcribe_to_srt(
    input_path: str,
    output_path: str | None = None,
    language: str = "it",
) -> str:
    """Transcribe a video/audio file and generate an .srt subtitle file.

    Automatically extracts audio from video files, and splits large audio
    into chunks to stay within the Groq API upload limit.
    """
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("Error: ffmpeg and ffprobe are required. Install them first.", file=sys.stderr)
        sys.exit(1)

    client = Groq()
    tmpdir = tempfile.mkdtemp(prefix="video-to-srt-")

    try:
        # Step 1: ensure we have an audio file
        ext = os.path.splitext(input_path)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            audio_path = input_path
        else:
            audio_path = os.path.join(tmpdir, "audio.mp3")
            extract_audio(input_path, audio_path)

        audio_size = os.path.getsize(audio_path)

        # Step 2: transcribe directly or split into chunks
        if audio_size <= MAX_FILE_SIZE:
            print("Audio fits within API limit, transcribing directly...")
            all_segments = transcribe_file(client, audio_path, language)
        else:
            # Estimate chunk duration to stay under the size limit
            total_duration = get_duration(audio_path)
            bytes_per_second = audio_size / total_duration
            chunk_duration = int(MAX_FILE_SIZE / bytes_per_second)

            chunk_dir = os.path.join(tmpdir, "chunks")
            os.makedirs(chunk_dir)
            chunks = split_audio(audio_path, chunk_dir, chunk_duration)

            all_segments = []
            for i, (chunk_path, offset) in enumerate(chunks):
                print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
                segments = transcribe_file(client, chunk_path, language)
                # Shift timestamps by the chunk's offset in the original audio
                for seg in segments:
                    seg["start"] += offset
                    seg["end"] += offset
                all_segments.extend(segments)

        if not all_segments:
            print("No segments found in transcription.", file=sys.stderr)
            sys.exit(1)

        # Step 3: build SRT content
        srt_lines = []
        for i, seg in enumerate(all_segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            text = seg["text"].strip()
            srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

        srt_content = "\n".join(srt_lines)

        # Step 4: write output
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = base + ".srt"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        print(f"SRT file saved to: {output_path}")
        return output_path

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate .srt subtitles from a video/audio file using Groq Whisper."
    )
    parser.add_argument("video", help="Path to the input video or audio file")
    parser.add_argument(
        "-o", "--output", help="Output .srt file path (default: same name as input)"
    )
    parser.add_argument(
        "-l", "--language", default="it",
        help="Language code for transcription (default: it)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Error: file not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    transcribe_to_srt(args.video, args.output, args.language)


if __name__ == "__main__":
    main()
