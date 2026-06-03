# Export Results from the Gradio Interface — Design

**Date:** 2026-06-03
**Branch:** `export-results` (off `main`)
**Status:** Approved design — pending implementation plan

## Goal

Let a user download the results of a single-file analysis from the Gradio
app (`app.py`) as files. Today the app only displays results on screen
(segments table, JSON box, MSA text box, plot) with copy-to-clipboard on
the two text boxes — nothing can be saved to disk.

Batch (multi-file) inference in the UI is **out of scope** for this work;
it is a planned follow-up. This spec covers single-file export only.

## Scope

Four downloadable artifacts, plus a bundle:

| Format | File | Source |
|--------|------|--------|
| JSON | `<stem>.json` | the segments list already shown in the JSON box |
| MSA text | `<stem>.msa.txt` | the `time label` lines shown in the MSA box |
| CSV | `<stem>.csv` | the segments table (new serialization) |
| Plot PNG | `<stem>.png` | the activation/boundary figure already rendered |
| ZIP bundle | `<stem>_songformer.zip` | all four files above |

`<stem>` is the uploaded audio file's basename without extension
(e.g. `BC_5cd6a6.mp3` → `BC_5cd6a6`). Exports reflect exactly what is
displayed: the **rule-post-processed** segments and the rendered figure.

### CSV columns

Both machine-friendly seconds and human-friendly `mm:ss.mmm`:

```
start_sec,start_mmss,end_sec,end_mmss,label
0.00,00:00.000,12.34,00:12.340,intro
12.34,00:12.340,45.67,00:45.670,verse
```

`mm:ss.mmm` uses the same `format_time` rendering already used by the
on-screen table (`f"{minutes:02d}:{seconds:06.3f}"`).

## Approach

Generate all export files **eagerly** during analysis. Temp files are
cheap, and eager generation makes both the per-format buttons and the ZIP
button download instantly with no extra round-trip or hidden state. The
analyze handler writes artifacts to a per-run temp directory and returns
their paths to the download components.

Rejected alternative — lazy generation (store results in `gr.State`, each
button generates its file on click): more moving parts and more failure
surface for no real benefit at single-file scale.

## Components

### New module: `export_utils.py` (repo root, beside `app.py`)

Pure, UI-agnostic, unit-testable functions. Keeping serialization out of
`app.py` (already ~600 lines) and out of the inference code makes each
piece independently testable without loading any model.

- `format_time(t: float) -> str`
  Render seconds as `mm:ss.mmm`. (Moved/shared from the local helper
  currently nested inside `process_and_analyze`.)

- `segments_to_csv(segments: list[dict]) -> str`
  Build the CSV text (header + rows) from the segment dicts
  (`{"start","end","label"}`). Uses `csv` module for correct quoting/
  escaping of labels.

- `write_exports(audio_path, segments, json_str, msa_str, fig, out_dir) -> dict`
  Write `<stem>.json`, `<stem>.msa.txt`, `<stem>.csv`, `<stem>.png` into
  `out_dir`. Reuses the already-built `json_str`/`msa_str`; saves the
  figure via `fig.savefig(..., dpi=...)`. Returns
  `{"json": path, "msa": path, "csv": path, "png": path}`.

- `make_zip(paths: list[str], zip_path: str) -> str`
  Bundle the given files into `zip_path` (flat, basenames only). Returns
  `zip_path`.

### `app.py` changes

1. Add an "⬇️ Export" `gr.Row` below the plot with five
   `gr.DownloadButton`s: JSON, MSA, CSV, PNG, and **Download all (ZIP)**.
   They start with no value (inactive) and are populated as outputs of the
   analyze click.

2. `process_and_analyze(audio_file)` gains five return values (the five
   file paths) appended after the existing four
   (`table_data, json_format, msa_format, fig`). It:
   - creates a per-run temp dir via `tempfile.mkdtemp()`,
   - calls `write_exports(...)` then `make_zip(...)`,
   - returns the five paths.

3. The local `format_time` is replaced by the shared
   `export_utils.format_time` so table and CSV stay in sync.

4. `analyze_btn.click(...)` `outputs=` list extends to include the five
   download buttons.

## Data flow

```
audio file
  └─ process_audio ─▶ logits, msa_output
       └─ rule_post_processing ─▶ segments (displayed + exported)
            ├─ format_as_json / format_as_msa ─▶ JSON box, MSA box
            ├─ create_visualization ─────────▶ Plot
            └─ write_exports + make_zip ──────▶ 5 file paths ─▶ DownloadButtons
```

## Error / edge handling

- **No input / exception:** `process_and_analyze` returns `None` for all
  five download paths (buttons clear; nothing stale is downloadable),
  alongside the existing `None`/`""` display returns.
- **Concurrency / repeat runs:** each run gets its own `mkdtemp()`
  directory, so paths never collide across analyses.
- **Plot saving:** the `matplotlib.use("Agg")` backend (added during the
  branch consolidation) makes `fig.savefig` safe in the server thread.
- **Label escaping:** CSV is written with the `csv` module so commas/
  quotes in labels are handled (labels are a fixed class set today, but
  this keeps the serializer robust).

## Testing

`tests/test_export_utils.py` — model-free, fast, deterministic, run with
`pytest`:

- `format_time` formats representative values correctly
  (`0.0 → 00:00.000`, `12.34 → 00:12.340`, `61.5 → 01:01.500`).
- `segments_to_csv` produces the exact header and one row per segment,
  with both seconds and `mm:ss.mmm` columns, and quotes a label
  containing a comma.
- `write_exports` writes all four files with the correct `<stem>.*`
  names into `tmp_path`; JSON parses back to the input segments; MSA text
  matches input; PNG is a non-empty file (use a tiny real
  `matplotlib.figure.Figure`).
- `make_zip` produces a zip whose members are exactly the four expected
  basenames.

The Gradio wiring (buttons populate and download) is verified by launching
the app manually, not in automated tests.

## Out of scope (future work)

- Batch / multi-file upload and export in the UI.
- Server-side persistence of results beyond the temp dir.
- Configurable export format selection in the UI.
