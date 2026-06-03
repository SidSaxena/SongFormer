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


def segments_to_csv(segments) -> str:
    """Build CSV text from segment dicts.

    Each segment is {"start": str|float, "end": str|float, "label": str}.
    Columns: start_sec, start_mmss, end_sec, end_mmss, label.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["start_sec", "start_mmss", "end_sec", "end_mmss", "label"])
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        writer.writerow(
            [
                f"{start:.2f}",
                format_time(start),
                f"{end:.2f}",
                format_time(end),
                seg["label"],
            ]
        )
    return buf.getvalue()


def write_exports(audio_path, segments, json_str, msa_str, fig, out_dir) -> dict:
    """Write json/msa/csv/png into out_dir; return {format: path}.

    Reuses the already-built json_str/msa_str from app.py rather than
    re-serializing. Saves the matplotlib figure as PNG.
    """
    stem = stem_of(audio_path)
    paths = {
        "json": os.path.join(out_dir, f"{stem}.json"),
        "msa": os.path.join(out_dir, f"{stem}.msa.txt"),
        "csv": os.path.join(out_dir, f"{stem}.csv"),
        "png": os.path.join(out_dir, f"{stem}.png"),
    }
    with open(paths["json"], "w", encoding="utf-8") as f:
        f.write(json_str)
    with open(paths["msa"], "w", encoding="utf-8") as f:
        f.write(msa_str)
    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        f.write(segments_to_csv(segments))
    fig.savefig(paths["png"], dpi=150, bbox_inches="tight")
    return paths
