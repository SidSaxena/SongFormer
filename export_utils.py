"""Serialize SongFormer analysis results to downloadable files.

Pure, UI-agnostic helpers used by app.py. No model or Gradio imports, so
these can be unit-tested without loading any checkpoint.
"""

import csv
import io
import os
import zipfile


def format_time(t: float) -> str:
    """Render seconds as mm:ss.mmm (e.g. 61.5 -> '01:01.500')."""
    minutes = int(t // 60)
    seconds = t % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def stem_of(audio_path: str) -> str:
    """Return the audio filename without directory or extension."""
    return os.path.splitext(os.path.basename(audio_path))[0]
