"""Serialize SongFormer analysis results to downloadable files.

Pure, UI-agnostic helpers used by app.py. No model or Gradio imports, so
these can be unit-tested without loading any checkpoint.
"""

import csv
import io
import json
import os
import shutil
import time
import zipfile

# Per-run export directories older than this (seconds) are swept at the start
# of each analysis. Recent runs are kept so their download files stay servable.
DEFAULT_EXPORT_TTL_SECONDS = 3600


def format_time(t: float) -> str:
    """Render seconds as mm:ss.mmm (e.g. 61.5 -> '01:01.500')."""
    minutes = int(t // 60)
    seconds = t % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def stem_of(audio_path: str) -> str:
    """Return the audio filename without directory or extension."""
    return os.path.splitext(os.path.basename(audio_path))[0]


def segments_to_table(segments) -> list:
    """Build display table rows: [start "(mm:ss.mmm)", end "(mm:ss.mmm)", label]."""
    rows = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        rows.append(
            [
                f"{start:.2f} ({format_time(start)})",
                f"{end:.2f} ({format_time(end)})",
                seg["label"],
            ]
        )
    return rows


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


def segments_to_combined_csv(named) -> str:
    """Build a combined CSV across files.

    `named` is a list of (filename, segments). Columns:
    filename, start_sec, start_mmss, end_sec, end_mmss, label.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["filename", "start_sec", "start_mmss", "end_sec", "end_mmss", "label"]
    )
    for filename, segments in named:
        for seg in segments:
            start = float(seg["start"])
            end = float(seg["end"])
            writer.writerow(
                [
                    filename,
                    f"{start:.2f}",
                    format_time(start),
                    f"{end:.2f}",
                    format_time(end),
                    seg["label"],
                ]
            )
    return buf.getvalue()


def combined_json(named) -> str:
    """Build a combined JSON mapping {filename: segments} across files."""
    return json.dumps(
        {filename: segments for filename, segments in named},
        indent=2,
        ensure_ascii=False,
    )


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


def make_zip(paths, zip_path) -> str:
    """Bundle the given files into zip_path using their basenames.

    Returns zip_path.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path


def zip_dir(src_dir, zip_path) -> str:
    """Zip the contents of src_dir into zip_path.

    Arcnames are relative to src_dir, preserving subfolders. Returns zip_path.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(src_dir):
            for name in files:
                full = os.path.join(root, name)
                arcname = os.path.relpath(full, src_dir)
                zf.write(full, arcname=arcname)
    return zip_path


def cleanup_old_exports(parent_dir, max_age_seconds, now=None) -> list:
    """Remove run subdirectories of parent_dir older than max_age_seconds.

    Only directories are swept (stray files are left alone). A missing
    parent_dir is a no-op. Recent runs are preserved so their download files
    remain servable. Returns the list of removed directory paths.
    """
    if now is None:
        now = time.time()
    removed = []
    if not os.path.isdir(parent_dir):
        return removed
    cutoff = now - max_age_seconds
    for name in sorted(os.listdir(parent_dir)):
        path = os.path.join(parent_dir, name)
        if not os.path.isdir(path):
            continue
        if os.path.getmtime(path) < cutoff:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed
