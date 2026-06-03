# Batch Export in the Gradio Interface — Design

**Date:** 2026-06-03
**Branch:** to be created off `main` (e.g. `batch-export`)
**Status:** Approved design — pending implementation plan
**Builds on:** single-file export (`export_utils.py`, shipped) and the
age-based temp-dir cleanup (`export_utils.cleanup_old_exports`, shipped).

## Goal

Let a user analyze **multiple audio files in one go** from the Gradio app
and download all results as a single ZIP, with a live per-file status
table and an on-demand per-file detail viewer. Single-file analysis stays
exactly as it is today, in its own tab.

## Constraints (what shapes the design)

- Inference is **heavy and sequential**: the models are global singletons
  on one GPU/MPS device, so files are processed one at a time and a batch
  of N files takes roughly N × (single-file time). The UI must show
  progress.
- The existing `export_utils` serialization helpers (`write_exports`,
  `make_zip`, `segments_to_csv`) are **pure and reusable per file**.
- The HF Spaces ZeroGPU per-call duration cap is a `hf-space`-branch
  concern, **not** `main`. This design targets `main` (local CUDA/MPS/CPU)
  and only notes the limitation.

## Scope

In scope:
- A new **"Batch"** tab (single-file UI moves into a "Single File" tab,
  otherwise unchanged).
- Multi-file upload, sequential analysis with live status + progress.
- A master ZIP with per-stem subfolders plus combined CSV and JSON
  manifests.
- A per-file detail viewer (dropdown → table/JSON/MSA/plot).

Out of scope (future work):
- Parallel inference, server-side folder-path input, result persistence
  beyond the temp dir, HF-Space-specific batching within ZeroGPU limits.

## UI structure

Wrap the existing Blocks body in `gr.Tabs`:

- **Tab "Single File"** — the current UI verbatim (audio input, analyze
  button, table/JSON/MSA/plot, the five single-file download buttons).
- **Tab "Batch"**:
  - `gr.File(file_count="multiple", type="filepath")` — upload
  - `gr.Button("Analyze Batch", variant="primary")`
  - status `gr.Dataframe` — headers **`["File", "Status", "Segments", "Duration"]`**, non-interactive
  - `gr.DownloadButton("⬇️ Download all (ZIP)")` — the master bundle
  - **Detail viewer**: `gr.Dropdown(label="Inspect a file")` + the four
    single-file-style displays:
    - `gr.Dataframe` (segments table)
    - `gr.Textbox` (JSON, copy button)
    - `gr.Textbox` (MSA, copy button)
    - `gr.Image` (the saved plot PNG)

## Processing flow

`process_batch(files, progress=gr.Progress())` — a **generator** handler in
`app.py`:

1. If `files` is empty/None: yield an empty status table + a notice and
   return (download button cleared, dropdown empty).
2. `exports_parent = <tempdir>/songformer_exports`; `os.makedirs(...,
   exist_ok=True)`; `cleanup_old_exports(exports_parent,
   DEFAULT_EXPORT_TTL_SECONDS)`; `run_dir = tempfile.mkdtemp(prefix="run_",
   dir=exports_parent)`; `bundle = run_dir/"bundle"`; `os.makedirs(bundle)`.
3. For each `audio_file` (sequential), updating `progress`:
   - `try`:
     - `logits, msa_output = process_audio(audio_file)`
     - `msa_output = rule_post_processing(msa_output)` (same as single-file)
     - `segments = format_as_segments(msa_output)`
     - `json_str = format_as_json(segments)`; `msa_str = format_as_msa(msa_output)`
     - `fig = create_visualization(logits, msa_output)`
     - `stem = export_utils.stem_of(audio_file)`; `file_dir = bundle/stem`;
       `os.makedirs(file_dir, exist_ok=True)`
     - `paths = export_utils.write_exports(audio_file, segments, json_str, msa_str, fig, file_dir)`
     - `plt.close(fig)` to free the figure
     - record result: status `"✅"`, segment count `len(segments)`, and
       duration displayed as `export_utils.format_time(float(segments[-1]["end"]))`
       (mm:ss.mmm; `""` if no segments); stash
       `{stem: {"segments": segments, "json": json_str, "msa": msa_str, "png": paths["png"]}}`
   - `except Exception as e`: record status `"❌ " + str(e)[:80]`, segments
     `0`, duration `""`; do **not** add to the bundle.
   - `yield` the current status-table rows (live update); other outputs
     unchanged until the end.
4. After the loop, for the **successful** files only:
   - `named = [(stem, segments) for each success]`
   - write `bundle/summary.csv = export_utils.segments_to_combined_csv(named)`
   - write `bundle/combined.json = export_utils.combined_json(named)`
   - `zip_path = run_dir/"songformer_batch.zip"`;
     `export_utils.zip_dir(bundle, zip_path)` (zip lives outside `bundle`,
     so it never self-includes; `run_dir` remains a normal cleanup target)
5. Final `yield`: status table, dropdown updated to the successful stems,
   the per-file results `gr.State`, and the master-ZIP download
   (`zip_path`, or `None` if there were zero successes).

**Detail viewer** — `on_select_file(stem, results_state)`:
- Returns `(segments_to_table(results[stem]["segments"]),
  results[stem]["json"], results[stem]["msa"], results[stem]["png"])`.
- Wired via `dropdown.change`.

## `export_utils` additions (pure, unit-tested)

- `segments_to_table(segments) -> list[list[str]]`
  Build the on-screen table rows
  `[f"{start:.2f} ({mmss})", f"{end:.2f} ({mmss})", label]`. Extracted from
  the current inline code in `process_and_analyze`; the single-file handler
  is refactored to call it, so single + batch render identically.

- `segments_to_combined_csv(named) -> str`
  `named` is a list of `(filename, segments)`. Columns:
  `filename,start_sec,start_mmss,end_sec,end_mmss,label`, one row per
  segment across all files. Uses `csv.writer(lineterminator="\n")` for
  correct quoting.

- `combined_json(named) -> str`
  `json.dumps({filename: segments}, indent=2, ensure_ascii=False)` over the
  same `named` list.

- `zip_dir(src_dir, zip_path) -> str`
  Walk `src_dir`, add each file with an arcname relative to `src_dir`
  (preserving the per-stem subfolders), `ZIP_DEFLATED`. Returns `zip_path`.
  The existing flat `make_zip` stays for the single-file flow.

Note: `combined_json` uses `json`, so `export_utils.py` will import `json`
for the first time here — legitimately used, unlike the earlier dead import.

## Master ZIP layout

```
songformer_batch.zip
  Song1/
    Song1.json
    Song1.msa.txt
    Song1.csv
    Song1.png
  Song2/
    Song2.json
    ...
  summary.csv      # combined CSV across all successful files
  combined.json    # {filename: [segments...]} across all successful files
```

## Error / edge handling

- **Per-file isolation:** a file that raises is recorded as `❌` in the
  status table and skipped; the batch continues. Failed files are excluded
  from the bundle and manifests.
- **Empty upload:** friendly notice, empty table, cleared download/dropdown.
- **All files fail:** no ZIP (download cleared); the status table shows each
  failure.
- **Temp lifecycle:** `run_dir` is created under `songformer_exports/` and
  swept by the existing age-based cleanup on later runs. The master ZIP is
  inside `run_dir` (not inside `bundle/`), so it is removed with its run dir.
- **Memory:** only per-file data (segments + short strings) and on-disk PNG
  paths are kept in `gr.State`; matplotlib figures are closed after saving
  (`plt.close`) so a large batch does not accumulate figures.

## Testing

`tests/test_export_utils.py` (model-free, fast):
- `segments_to_table`: row count and the `"{sec:.2f} ({mmss})"` formatting.
- `segments_to_combined_csv`: header, one row per segment across two files,
  `filename` column correct, a label with a comma is quoted.
- `combined_json`: parses back to `{filename: segments}` for multiple files.
- `zip_dir`: namelist contains the nested arcnames
  (e.g. `Song1/Song1.json`, `summary.csv`) and is rooted at `src_dir`.

The `process_batch` generator and the detail-viewer wiring require models
and Gradio runtime, so they are verified by a manual app launch (upload 2–3
files, watch the status table fill, download the ZIP, inspect a file in the
viewer), consistent with how single-file export was verified.

## Note on `app.py` size

`app.py` is already ~650 lines. This feature keeps **all** pure logic in
`export_utils` and adds only the tab layout + the `process_batch` /
`on_select_file` handlers to `app.py`. A future refactor extracting the
inference path (`process_audio`, model init) into its own module would
reduce `app.py` further; that is explicitly **not** part of this work.
