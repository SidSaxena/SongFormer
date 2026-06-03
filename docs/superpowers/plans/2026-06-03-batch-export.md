# Batch Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Batch" tab to the Gradio app that analyzes multiple uploaded audio files sequentially, shows live per-file status, lets the user download all results as one ZIP (per-stem subfolders + combined CSV/JSON manifests), and inspect any file's result in a detail viewer.

**Architecture:** Four new pure functions in `export_utils.py` (serialization/zip) are unit-tested model-free. `app.py` gains a `gr.Tabs` layout (existing UI → "Single File" tab, new "Batch" tab), a generator handler `process_batch` that loops inference and yields live status, and an `on_select_file` handler that re-renders a stored per-file result without recomputing.

**Tech Stack:** Python 3.10 (conda env `songformer`), Gradio 5.38, matplotlib (Agg), stdlib `csv`/`json`/`zipfile`/`os`, pytest.

**Spec:** `docs/superpowers/specs/2026-06-03-batch-export-design.md`

**Interpreter for all commands:** `/opt/miniconda3/envs/songformer/bin/python`

---

## File Structure

- **Modify** `export_utils.py` — add `segments_to_table`, `segments_to_combined_csv`, `combined_json`, `zip_dir`; add `import json`.
- **Modify** `tests/test_export_utils.py` — add unit tests for the four functions.
- **Modify** `app.py` — refactor single-file table-building to use `segments_to_table`; wrap existing UI in a "Single File" tab and add a "Batch" tab; add `process_batch` and `on_select_file` handlers and wire their events.

Tasks 1–4 are pure/tested. Tasks 5–6 are UI/handler wiring verified by compile + test regression + a final manual launch.

---

## Task 1: `segments_to_table` (+ refactor single-file handler)

**Files:**
- Modify: `export_utils.py`
- Modify: `app.py` (`process_and_analyze` table-building)
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_export_utils.py`:

```python
def test_segments_to_table():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": "12.34", "end": "45.67", "label": "verse"},
    ]
    rows = export_utils.segments_to_table(segments)
    assert rows == [
        ["0.00 (00:00.000)", "12.34 (00:12.340)", "intro"],
        ["12.34 (00:12.340)", "45.67 (00:45.670)", "verse"],
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_table -v`
Expected: FAIL — `AttributeError: module 'export_utils' has no attribute 'segments_to_table'`.

- [ ] **Step 3: Add `segments_to_table` to `export_utils.py`** — insert AFTER `stem_of` (before `segments_to_csv`):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_table -v`
Expected: PASS.

- [ ] **Step 5: Refactor `app.py` to use it** — in `process_and_analyze`, replace this block:

```python
        # Create table data
        table_data = [
            [
                f"{float(seg['start']):.2f} ({export_utils.format_time(float(seg['start']))})",
                f"{float(seg['end']):.2f} ({export_utils.format_time(float(seg['end']))})",
                seg["label"],
            ]
            for seg in segments
        ]
```

with:

```python
        # Create table data
        table_data = export_utils.segments_to_table(segments)
```

- [ ] **Step 6: Verify compile + full test suite**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py && /opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: compile OK; all tests pass (10 total).

- [ ] **Step 7: Commit**

```bash
git add export_utils.py app.py tests/test_export_utils.py
git commit -m "feat: add export_utils.segments_to_table and reuse it in single-file handler"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 2: `segments_to_combined_csv`

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_export_utils.py`:

```python
def test_segments_to_combined_csv():
    named = [
        ("Song1", [{"start": "0.0", "end": "1.0", "label": "intro"}]),
        ("Song2", [{"start": "0.0", "end": "2.5", "label": "a,b"}]),
    ]
    csv_text = export_utils.segments_to_combined_csv(named)
    lines = csv_text.strip().split("\n")
    assert lines[0] == "filename,start_sec,start_mmss,end_sec,end_mmss,label"
    assert lines[1] == "Song1,0.00,00:00.000,1.00,00:01.000,intro"
    assert lines[2].startswith("Song2,0.00,00:00.000,2.50,00:02.500,")
    assert '"a,b"' in csv_text
    assert len(lines) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_combined_csv -v`
Expected: FAIL — `AttributeError: ... has no attribute 'segments_to_combined_csv'`.

- [ ] **Step 3: Add `segments_to_combined_csv` to `export_utils.py`** — insert AFTER `segments_to_csv`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_combined_csv -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.segments_to_combined_csv"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 3: `combined_json`

**Files:**
- Modify: `export_utils.py` (also add `import json`)
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_export_utils.py`:

```python
def test_combined_json():
    named = [
        ("Song1", [{"start": 0.0, "end": 1.0, "label": "intro"}]),
        ("Song2", [{"start": 0.0, "end": 2.0, "label": "verse"}]),
    ]
    data = json.loads(export_utils.combined_json(named))
    assert set(data.keys()) == {"Song1", "Song2"}
    assert data["Song1"] == [{"start": 0.0, "end": 1.0, "label": "intro"}]
```

(`json` is already imported at the top of the test file.)

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_combined_json -v`
Expected: FAIL — `AttributeError: ... has no attribute 'combined_json'`.

- [ ] **Step 3: Add `import json` and the function to `export_utils.py`**

First add `import json` to the import block so it reads:

```python
import csv
import io
import json
import os
import shutil
import time
import zipfile
```

Then insert AFTER `segments_to_combined_csv`:

```python
def combined_json(named) -> str:
    """Build a combined JSON mapping {filename: segments} across files."""
    return json.dumps(
        {filename: segments for filename, segments in named},
        indent=2,
        ensure_ascii=False,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_combined_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.combined_json"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 4: `zip_dir`

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test** — append to `tests/test_export_utils.py`:

```python
def test_zip_dir(tmp_path):
    src = tmp_path / "bundle"
    (src / "Song1").mkdir(parents=True)
    (src / "Song1" / "Song1.json").write_text("{}")
    (src / "summary.csv").write_text("x")
    zip_path = str(tmp_path / "batch.zip")

    returned = export_utils.zip_dir(str(src), zip_path)

    assert returned == zip_path
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(zf.namelist())
    assert names == ["Song1/Song1.json", "summary.csv"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_zip_dir -v`
Expected: FAIL — `AttributeError: ... has no attribute 'zip_dir'`.

- [ ] **Step 3: Add `zip_dir` to `export_utils.py`** — insert AFTER `make_zip`:

```python
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
```

- [ ] **Step 4: Run the full module test file to verify all pass**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -v`
Expected: PASS — all tests green (13 total).

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.zip_dir"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 5: Restructure `app.py` UI into tabs + add Batch tab components

**Files:**
- Modify: `app.py`

No automated test (Gradio layout). Verify with `py_compile`. The single-file UI is moved verbatim into a "Single File" tab (only indentation changes); the existing `analyze_btn.click` handler stays where it is (component variables remain in scope).

- [ ] **Step 1: Wrap the single-file UI in tabs and add the Batch tab**

In `app.py`, find the block that currently begins with `    # Main input area` and ends with the single-file download-buttons row (the `with gr.Row():` containing `download_json_btn` … `download_zip_btn`). Replace that ENTIRE block with the following. (The single-file content is identical, just indented under `gr.Tabs` → `gr.Tab("Single File")`; the new `gr.Tab("Batch")` is appended.)

```python
    with gr.Tabs():
        with gr.Tab("Single File"):
            # Main input area
            with gr.Row():
                with gr.Column(scale=3):
                    audio_input = gr.Audio(
                        label="Upload Audio File", type="filepath", elem_id="audio-input"
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### 📌 Examples")
                    gr.Examples(
                        examples=[
                            ["examples/BC_5cd6a6.mp3"],
                            ["examples/BC_282ece.mp3"],
                            ["examples/BHX_0158_letitrock.wav"],
                            ["examples/BHX_0374_drunkonyou.wav"],
                        ],
                        inputs=[audio_input],
                        label="Click to load example",
                    )

            # Analyze button
            with gr.Row():
                analyze_btn = gr.Button(
                    "🚀 Analyze Music Structure", variant="primary", scale=1
                )

            # Results display area
            with gr.Row():
                with gr.Column(scale=13):
                    segments_table = gr.Dataframe(
                        headers=["Start / s (m:s.ms)", "End / s (m:s.ms)", "Label"],
                        label="Detected Music Segments",
                        interactive=False,
                        elem_id="result-table",
                    )
                with gr.Column(scale=8):
                    with gr.Row():
                        with gr.Accordion("📄 JSON Output", open=False):
                            json_output = gr.Textbox(
                                label="JSON Format",
                                lines=15,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )
                    with gr.Row():
                        with gr.Accordion("📋 MSA Text Output", open=False):
                            msa_output = gr.Textbox(
                                label="MSA Format",
                                lines=15,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )

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

        with gr.Tab("Batch"):
            gr.Markdown(
                "Upload multiple audio files, analyze them sequentially, "
                "and download all results as a single ZIP."
            )
            batch_files = gr.File(
                label="Upload Audio Files",
                file_count="multiple",
                type="filepath",
            )
            batch_analyze_btn = gr.Button("🚀 Analyze Batch", variant="primary")
            batch_status = gr.Dataframe(
                headers=["File", "Status", "Segments", "Duration"],
                label="Batch Status",
                interactive=False,
            )
            batch_zip_btn = gr.DownloadButton("⬇️ Download all (ZIP)")
            batch_results_state = gr.State({})
            gr.Markdown("### Inspect a file")
            batch_file_selector = gr.Dropdown(
                label="Processed file", choices=[], interactive=True
            )
            batch_detail_table = gr.Dataframe(
                headers=["Start / s (m:s.ms)", "End / s (m:s.ms)", "Label"],
                label="Detected Music Segments",
                interactive=False,
            )
            with gr.Accordion("📄 JSON Output", open=False):
                batch_detail_json = gr.Textbox(
                    label="JSON Format",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
            with gr.Accordion("📋 MSA Text Output", open=False):
                batch_detail_msa = gr.Textbox(
                    label="MSA Format",
                    lines=15,
                    max_lines=20,
                    interactive=False,
                    show_copy_button=True,
                )
            batch_detail_plot = gr.Image(label="Activation Curves Visualization")
```

- [ ] **Step 2: Verify compile + tests still pass**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py && /opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: compile OK; all tests pass. (The existing `analyze_btn.click` handler is unchanged and its component variables — `audio_input`, `segments_table`, etc. — are still in scope because Python `with` blocks do not create new scopes.)

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: split Gradio UI into Single File and Batch tabs"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 6: Add `process_batch` + `on_select_file` handlers and wire events

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the two handler functions**

In `app.py`, insert these two functions immediately AFTER the `process_and_analyze` function definition (i.e. after its `return None, "", error_msg, ...` except block ends) and BEFORE the `# Create Gradio interface` / `with gr.Blocks(...)` section:

```python
def process_batch(files, progress=gr.Progress()):
    """Analyze multiple files sequentially, yielding live status.

    Outputs (per yield): status rows, ZIP download update, file-selector
    update, per-file results dict (for the detail viewer).
    """
    if not files:
        yield (
            [["(no files uploaded)", "", "", ""]],
            gr.update(value=None),
            gr.update(choices=[], value=None),
            {},
        )
        return

    exports_parent = os.path.join(tempfile.gettempdir(), "songformer_exports")
    os.makedirs(exports_parent, exist_ok=True)
    export_utils.cleanup_old_exports(
        exports_parent, export_utils.DEFAULT_EXPORT_TTL_SECONDS
    )
    run_dir = tempfile.mkdtemp(prefix="run_", dir=exports_parent)
    bundle = os.path.join(run_dir, "bundle")
    os.makedirs(bundle, exist_ok=True)

    status_rows = []
    results = {}
    named = []

    for audio_file in progress.tqdm(files, desc="Analyzing"):
        stem = export_utils.stem_of(audio_file)
        try:
            logits, msa_output = process_audio(audio_file)
            msa_output = rule_post_processing(msa_output)
            segments = format_as_segments(msa_output)
            json_str = format_as_json(segments)
            msa_str = format_as_msa(msa_output)
            fig = create_visualization(logits, msa_output)
            file_dir = os.path.join(bundle, stem)
            os.makedirs(file_dir, exist_ok=True)
            paths = export_utils.write_exports(
                audio_file, segments, json_str, msa_str, fig, file_dir
            )
            plt.close(fig)
            duration = (
                export_utils.format_time(float(segments[-1]["end"]))
                if segments
                else ""
            )
            status_rows.append([stem, "✅", len(segments), duration])
            results[stem] = {
                "segments": segments,
                "json": json_str,
                "msa": msa_str,
                "png": paths["png"],
            }
            named.append((stem, segments))
        except Exception as e:
            status_rows.append([stem, "❌ " + str(e)[:80], 0, ""])
        yield status_rows, gr.update(), gr.update(), results

    if named:
        with open(
            os.path.join(bundle, "summary.csv"), "w", encoding="utf-8", newline=""
        ) as f:
            f.write(export_utils.segments_to_combined_csv(named))
        with open(
            os.path.join(bundle, "combined.json"), "w", encoding="utf-8"
        ) as f:
            f.write(export_utils.combined_json(named))
        zip_path = os.path.join(run_dir, "songformer_batch.zip")
        export_utils.zip_dir(bundle, zip_path)
    else:
        zip_path = None

    yield (
        status_rows,
        gr.update(value=zip_path),
        gr.update(choices=list(results.keys()), value=None),
        results,
    )


def on_select_file(stem, results):
    """Render a previously-computed file's result in the batch detail viewer."""
    results = results or {}
    if not stem or stem not in results:
        return None, "", "", None
    r = results[stem]
    return (
        export_utils.segments_to_table(r["segments"]),
        r["json"],
        r["msa"],
        r["png"],
    )
```

- [ ] **Step 2: Wire the batch events**

In the `with gr.Blocks(...)` body, find the existing `analyze_btn.click(...)` call (single-file). Immediately AFTER it (same indentation level, still inside the Blocks `with`), add:

```python
    batch_analyze_btn.click(
        fn=process_batch,
        inputs=[batch_files],
        outputs=[
            batch_status,
            batch_zip_btn,
            batch_file_selector,
            batch_results_state,
        ],
    )
    batch_file_selector.change(
        fn=on_select_file,
        inputs=[batch_file_selector, batch_results_state],
        outputs=[
            batch_detail_table,
            batch_detail_json,
            batch_detail_msa,
            batch_detail_plot,
        ],
    )
```

- [ ] **Step 3: Verify compile + tests**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py && /opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: compile OK; all tests pass.

- [ ] **Step 4: Manual verification (launch the app)** — SKIP during subagent execution; the controller runs this separately.

`conda activate songformer && python app.py`, then in the browser:
1. Open the **Batch** tab; upload 2–3 example files from `src/SongFormer/examples/`.
2. Click **Analyze Batch**; confirm the status table fills row-by-row (✅ with segment counts + durations), progress bar advances.
3. Click **Download all (ZIP)**; confirm `songformer_batch.zip` contains per-stem subfolders, `summary.csv`, and `combined.json`.
4. Pick a file in the **Inspect a file** dropdown; confirm its table/JSON/MSA/plot render.
5. Confirm the **Single File** tab still works end-to-end.

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: add batch analysis handler and detail viewer to Gradio app"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Self-Review notes (author checklist — already applied)

- **Spec coverage:** Batch tab + multi-file upload (Task 5); sequential analysis with live status + progress (Task 6 `process_batch` generator + `progress.tqdm`); per-stem-subfolder ZIP (`zip_dir`, Task 4 + Task 6) with `summary.csv` (`segments_to_combined_csv`, Task 2) and `combined.json` (`combined_json`, Task 3); detail viewer (`on_select_file` + `segments_to_table`, Tasks 1 & 6); per-file error isolation + empty-upload notice (Task 6); temp-dir reuse via `cleanup_old_exports` (Task 6); `plt.close` to bound memory (Task 6); status columns `[File, Status, Segments, Duration]` with duration as `mm:ss.mmm` (Task 6). All covered.
- **Placeholder scan:** none — every code/command step is concrete.
- **Type consistency:** `process_batch` yields a 4-tuple on every path, matching the 4 `batch_analyze_btn.click` outputs (`batch_status`, `batch_zip_btn`, `batch_file_selector`, `batch_results_state`). `on_select_file` returns a 4-tuple matching its 4 outputs (`batch_detail_table/json/msa/plot`). `segments_to_table` returns `list[list]` consumed by the table components; `segments_to_combined_csv`/`combined_json` take the same `named` list of `(filename, segments)`; `zip_dir(src_dir, zip_path)` signature matches its call. `results[stem]` dict keys (`segments`/`json`/`msa`/`png`) are written in Task 6 and read in `on_select_file`.
- **Scope:** single focused feature; no inference refactor (explicitly deferred).
