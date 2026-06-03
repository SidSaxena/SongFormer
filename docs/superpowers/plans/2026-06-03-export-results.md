# Result Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user download a single-file analysis from the Gradio app as JSON, MSA text, CSV, and plot PNG files, plus a ZIP bundle, via per-format buttons and a "Download all" button.

**Architecture:** A new pure, model-free module `export_utils.py` (beside `app.py`) serializes results to files in a per-run temp directory and returns their paths. `app.py`'s analyze handler calls it and feeds the paths to five `gr.DownloadButton`s. Serialization stays out of the UI and inference code so it can be unit-tested without loading any model.

**Tech Stack:** Python 3.10 (conda env `songformer`), stdlib `csv`/`json`/`zipfile`/`tempfile`, matplotlib (Agg backend), Gradio 5.38, pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-export-results-design.md`

**Interpreter for all commands:** `/opt/miniconda3/envs/songformer/bin/python`
(equivalently, `conda activate songformer` first, then use `python`).

---

## File Structure

- **Create** `export_utils.py` (repo root) — pure serialization helpers: `format_time`, `stem_of`, `segments_to_csv`, `write_exports`, `make_zip`. No Gradio/model imports.
- **Create** `tests/test_export_utils.py` — unit tests for the above.
- **Create** `conftest.py` (repo root) — empty; makes pytest put the repo root on `sys.path` so `import export_utils` resolves from `tests/`.
- **Create** `requirements-dev.txt` — `pytest` (dev-only test dependency).
- **Modify** `app.py` — add `import tempfile` and `import export_utils`; replace the local `format_time` with `export_utils.format_time`; extend `process_and_analyze` to write exports and return 5 paths; add a download-button row; extend the analyze click `outputs`.

---

## Task 0: Test tooling setup

**Files:**
- Create: `requirements-dev.txt`
- Create: `conftest.py`
- Create: `tests/` (directory, via the test file in Task 1)

- [ ] **Step 1: Create `requirements-dev.txt`**

```
# Development / test dependencies (not needed at runtime)
pytest>=8.0
```

- [ ] **Step 2: Create empty `conftest.py` at repo root**

```python
# Presence of this file makes pytest add the repo root to sys.path,
# so tests can `import export_utils`.
```

- [ ] **Step 3: Install pytest into the songformer env**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pip install -r requirements-dev.txt`
Expected: pytest installs successfully (or "Requirement already satisfied").

- [ ] **Step 4: Verify pytest runs**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest --version`
Expected: prints `pytest 8.x`.

- [ ] **Step 5: Commit**

```bash
git add requirements-dev.txt conftest.py
git commit -m "test: add pytest dev dependency and conftest for import path"
```

---

## Task 1: `format_time` and `stem_of`

**Files:**
- Create: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_utils.py`:

```python
import json
import os
import zipfile

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

import export_utils


def test_format_time():
    assert export_utils.format_time(0.0) == "00:00.000"
    assert export_utils.format_time(12.34) == "00:12.340"
    assert export_utils.format_time(61.5) == "01:01.500"


def test_stem_of():
    assert export_utils.stem_of("/a/b/Song.mp3") == "Song"
    assert export_utils.stem_of("Track.wav") == "Track"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_utils'`.

- [ ] **Step 3: Create `export_utils.py` with the two functions**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -v`
Expected: PASS (`test_format_time`, `test_stem_of`).

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.format_time and stem_of"
```

---

## Task 2: `segments_to_csv`

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_export_utils.py`:

```python
def test_segments_to_csv_basic():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": "12.34", "end": "45.67", "label": "verse"},
    ]
    csv_text = export_utils.segments_to_csv(segments)
    lines = csv_text.strip().split("\n")
    assert lines[0] == "start_sec,start_mmss,end_sec,end_mmss,label"
    assert lines[1] == "0.00,00:00.000,12.34,00:12.340,intro"
    assert lines[2] == "12.34,00:12.340,45.67,00:45.670,verse"
    assert len(lines) == 3


def test_segments_to_csv_quotes_label_with_comma():
    segments = [{"start": "0.0", "end": "1.0", "label": "a,b"}]
    csv_text = export_utils.segments_to_csv(segments)
    assert '"a,b"' in csv_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -k csv -v`
Expected: FAIL — `AttributeError: module 'export_utils' has no attribute 'segments_to_csv'`.

- [ ] **Step 3: Add `segments_to_csv` to `export_utils.py`**

Add after `stem_of`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -k csv -v`
Expected: PASS (both csv tests).

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.segments_to_csv"
```

---

## Task 3: `write_exports`

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_export_utils.py`:

```python
def test_write_exports(tmp_path):
    segments = [{"start": "0.0", "end": "1.0", "label": "intro"}]
    json_str = json.dumps(segments)
    msa_str = "0.00 intro\n1.00 end"
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])

    paths = export_utils.write_exports(
        "/some/dir/Song.mp3", segments, json_str, msa_str, fig, str(tmp_path)
    )

    assert os.path.basename(paths["json"]) == "Song.json"
    assert os.path.basename(paths["msa"]) == "Song.msa.txt"
    assert os.path.basename(paths["csv"]) == "Song.csv"
    assert os.path.basename(paths["png"]) == "Song.png"
    # Every file exists and lives in tmp_path
    for p in paths.values():
        assert os.path.isfile(p)
    # Contents round-trip
    with open(paths["json"], encoding="utf-8") as f:
        assert json.load(f) == segments
    with open(paths["msa"], encoding="utf-8") as f:
        assert f.read() == msa_str
    assert os.path.getsize(paths["png"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_write_exports -v`
Expected: FAIL — `AttributeError: module 'export_utils' has no attribute 'write_exports'`.

- [ ] **Step 3: Add `write_exports` to `export_utils.py`**

Add after `segments_to_csv`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_write_exports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.write_exports"
```

---

## Task 4: `make_zip`

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_export_utils.py`:

```python
def test_make_zip(tmp_path):
    f1 = tmp_path / "a.json"
    f1.write_text("{}")
    f2 = tmp_path / "a.csv"
    f2.write_text("x")
    zip_path = str(tmp_path / "a_songformer.zip")

    returned = export_utils.make_zip([str(f1), str(f2)], zip_path)

    assert returned == zip_path
    with zipfile.ZipFile(zip_path) as zf:
        assert sorted(zf.namelist()) == ["a.csv", "a.json"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_make_zip -v`
Expected: FAIL — `AttributeError: module 'export_utils' has no attribute 'make_zip'`.

- [ ] **Step 3: Add `make_zip` to `export_utils.py`**

Add after `write_exports`:

```python
def make_zip(paths, zip_path) -> str:
    """Bundle the given files into zip_path using their basenames.

    Returns zip_path.
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))
    return zip_path
```

- [ ] **Step 4: Run the full test file to verify all pass**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.make_zip"
```

---

## Task 5: Wire export into `app.py`

**Files:**
- Modify: `app.py`

No automated test (the inference path requires loading models and audio). Verification is a syntax compile plus a manual app launch.

- [ ] **Step 1: Add the new imports**

In `app.py`, the stdlib import block currently includes (around the top, after the monkey-patch):

```python
import json
import math
import importlib
```

Add `import tempfile` next to `import json`:

```python
import json
import tempfile
import math
import importlib
```

And add `import export_utils` immediately after the existing project imports (after the line `from utils.fetch_pretrained import download_all`):

```python
from utils.fetch_pretrained import download_all

import export_utils
```

(Note: `app.py` calls `os.chdir(...)` at startup, but the script's own directory — the repo root — stays on `sys.path`, so `import export_utils` resolves.)

- [ ] **Step 2: Replace the local `format_time` with the shared one**

In `process_and_analyze`, delete the nested helper:

```python
    def format_time(t: float) -> str:
        minutes = int(t // 60)
        seconds = t % 60
        return f"{minutes:02d}:{seconds:06.3f}"

```

Then change the two table-cell calls from `format_time(...)` to `export_utils.format_time(...)`:

```python
        table_data = [
            [
                f"{float(seg['start']):.2f} ({export_utils.format_time(float(seg['start']))})",
                f"{float(seg['end']):.2f} ({export_utils.format_time(float(seg['end']))})",
                seg["label"],
            ]
            for seg in segments
        ]
```

- [ ] **Step 3: Extend the success return path to generate and return export paths**

In `process_and_analyze`, the success branch currently ends:

```python
        # Create visualization
        fig = create_visualization(logits, msa_output)

        return table_data, json_format, msa_format, fig
```

Replace with:

```python
        # Create visualization
        fig = create_visualization(logits, msa_output)

        # Write downloadable export files into a per-run temp directory
        out_dir = tempfile.mkdtemp(prefix="songformer_")
        export_paths = export_utils.write_exports(
            audio_file, segments, json_format, msa_format, fig, out_dir
        )
        zip_path = os.path.join(
            out_dir, export_utils.stem_of(audio_file) + "_songformer.zip"
        )
        export_utils.make_zip(list(export_paths.values()), zip_path)

        return (
            table_data,
            json_format,
            msa_format,
            fig,
            export_paths["json"],
            export_paths["msa"],
            export_paths["csv"],
            export_paths["png"],
            zip_path,
        )
```

- [ ] **Step 4: Update the no-input and error return paths**

Change the early no-input return:

```python
    if audio_file is None:
        return None, "", "", None
```

to:

```python
    if audio_file is None:
        return None, "", "", None, None, None, None, None, None
```

And the `except` block return:

```python
        return None, "", error_msg, None
```

to:

```python
        return None, "", error_msg, None, None, None, None, None, None
```

- [ ] **Step 5: Add the download-button row in the Blocks layout**

In the `with gr.Blocks(...)` body, the plot row currently reads:

```python
    # Visualization plot
    with gr.Row():
        plot_output = gr.Plot(label="Activation Curves Visualization")
```

Add a download row immediately after it:

```python
    # Visualization plot
    with gr.Row():
        plot_output = gr.Plot(label="Activation Curves Visualization")

    # Export / download buttons (populated after analysis)
    with gr.Row():
        download_json_btn = gr.DownloadButton("⬇️ JSON")
        download_msa_btn = gr.DownloadButton("⬇️ MSA (.txt)")
        download_csv_btn = gr.DownloadButton("⬇️ CSV")
        download_png_btn = gr.DownloadButton("⬇️ Plot (.png)")
        download_zip_btn = gr.DownloadButton(
            "⬇️ Download all (ZIP)", variant="primary"
        )
```

- [ ] **Step 6: Extend the analyze click outputs**

The handler is currently:

```python
    analyze_btn.click(
        fn=process_and_analyze,
        inputs=[audio_input],
        outputs=[segments_table, json_output, msa_output, plot_output],
    )
```

Replace the `outputs` list:

```python
    analyze_btn.click(
        fn=process_and_analyze,
        inputs=[audio_input],
        outputs=[
            segments_table,
            json_output,
            msa_output,
            plot_output,
            download_json_btn,
            download_msa_btn,
            download_csv_btn,
            download_png_btn,
            download_zip_btn,
        ],
    )
```

- [ ] **Step 7: Verify the file compiles**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py`
Expected: no output (success).

- [ ] **Step 8: Re-run the unit tests (regression check)**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -v`
Expected: all 6 tests still PASS.

- [ ] **Step 9: Manual verification (launch the app)**

Run: `conda activate songformer && python app.py`
(First launch downloads checkpoints and loads models; allow a few minutes.)
Then in the browser at the printed local URL:
1. Click an example (e.g. `BC_5cd6a6.mp3`) and press **Analyze Music Structure**.
2. Confirm the table, JSON, MSA, and plot populate.
3. Click each of the five download buttons; confirm files download named `BC_5cd6a6.json`, `BC_5cd6a6.msa.txt`, `BC_5cd6a6.csv`, `BC_5cd6a6.png`, and `BC_5cd6a6_songformer.zip`.
4. Open the CSV: header `start_sec,start_mmss,end_sec,end_mmss,label`; open the ZIP: contains the four files.

- [ ] **Step 10: Commit**

```bash
git add app.py
git commit -m "feat: add result export download buttons to Gradio app"
```

---

## Self-Review notes (author checklist — already applied)

- **Spec coverage:** JSON/MSA/CSV/PNG + ZIP (Tasks 3–5); per-format + Download-all buttons (Task 5); CSV with seconds + mm:ss.mmm (Task 2); eager temp-dir generation, audio-stem names (Tasks 3, 5); error/no-input clears buttons (Task 5 Step 4); Agg backend already on `main`; unit tests model-free (Tasks 1–4). All covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Type consistency:** `write_exports` returns dict keyed `json`/`msa`/`csv`/`png`, consumed by that exact key set in `app.py` (Task 5 Step 3); `make_zip(paths, zip_path)` signature matches its call; `stem_of`/`format_time` names consistent across module, tests, and `app.py`.
- **Handler arity:** `process_and_analyze` returns 9 values on every path (success, no-input, error), matching the 9 `outputs` components.
```
