# Audacity Export + Colab Drive Persistence QoL — Design

**Date:** 2026-06-04
**Branch:** `worktree-qol-exports-drive` (off `main`)
**Status:** Approved design — pending implementation plan

## Goal

Three small, independent quality-of-life features:

1. **Audacity labels export** — every analysis also produces an
   Audacity-importable label file; one-click download in the Single File
   tab.
2. **MuQ HF cache on Drive** (Colab) — `STORE_ON_DRIVE` persists the MuQ
   encoder cache too, not just the checkpoints.
3. **Auto-save batch results to Drive** (Colab) — optional toggle copies
   the B4 results ZIP to Drive with a timestamped name.

After merge to `main`: sync `hf-space` per the runbook and redeploy the
Space (it inherits feature 1; features 2–3 are notebook-only and the
notebook is pruned from that branch).

## Feature 1: Audacity labels export

### `export_utils.py`

New pure serializer:

```python
def segments_to_audacity(segments) -> str:
    """Build an Audacity label-track file from segment dicts.

    One line per segment: start<TAB>end<TAB>label, seconds with 6
    decimals (Audacity's File > Import > Labels format).
    """
```

Output example:

```
0.000000	12.340000	intro
12.340000	45.670000	verse
```

`write_exports(...)` writes a fifth artifact `<stem>.audacity.txt` using
the serializer and returns it under the `"audacity"` key. Because the
single-file ZIP (`make_zip(list(paths.values()))`), the batch per-stem
folders, and the batch master ZIP all consume this dict/folder, the file
propagates to every bundle automatically.

### `app.py`

- Single File tab download row gains a sixth button:
  `download_audacity_btn = gr.DownloadButton("⬇️ Audacity labels")`,
  placed between the CSV and Plot buttons.
- `process_and_analyze` returns **10 values** on all three paths
  (success adds `export_paths["audacity"]`; no-input and error paths add
  one more `None`).
- `analyze_btn.click` outputs list grows to 10, same order as returns.
- Batch detail viewer (`on_select_file`) unchanged.

### Tests (TDD)

- `test_segments_to_audacity`: exact lines, tab separation, 6-decimal
  formatting, str/float coercion of start/end.
- `test_write_exports` extended: `<stem>.audacity.txt` exists with the
  expected basename; `test_write_exports_custom_stem` extended likewise
  (`Song_2.audacity.txt`).

## Feature 2: MuQ HF cache on Drive (notebook)

**Cell 11 (`code-checkpoints`)**, inside the existing `if STORE_ON_DRIVE:`
branch, BEFORE any model download:

- `HF_CACHE = '/root/.cache/huggingface'`,
  `DRIVE_HF = '/content/drive/MyDrive/SongFormer/hf_cache'`.
- If `HF_CACHE` is a real directory (not a symlink): move its contents
  into `DRIVE_HF` (create it first; skip files that already exist), remove
  the local dir, then `os.symlink(DRIVE_HF, HF_CACHE)`.
- If `HF_CACHE` doesn't exist: create `DRIVE_HF` and symlink.
- If already a symlink: no-op (idempotent across re-runs).
- Print where the HF cache lives.

Symlinking the cache directory is used instead of `HF_HOME` because
`huggingface_hub` resolves its env-config at import time and earlier cells
have already imported it. On Drive's FUSE filesystem `huggingface_hub`
detects missing symlink support and falls back to copying files inside
the cache — documented upstream behavior, costs some duplicate bytes.

**Cell 10 heading** (always visible): totals updated to **~2.7 GB**
(1.33 GB checkpoints + 1.33 GB MuQ encoder HF cache) and an honest
caveat added: Drive reads via FUSE can be *slower* than re-downloading
from HF — the toggle saves bandwidth and re-download flakiness, not
necessarily startup time.

## Feature 3: Auto-save batch results to Drive (notebook)

**Cell 23 (`code-download-results`)**, already form-view:

- Add `SAVE_TO_DRIVE = False  #@param {type:"boolean"}` under the
  existing `#@title`.
- After zipping: if `SAVE_TO_DRIVE`, mount Drive if needed, create
  `/content/drive/MyDrive/SongFormer/results/`, and copy the ZIP to
  `songformer_results_<YYYYMMDD-HHMMSS>.zip` (timestamped — never
  overwrites). Print the Drive path.
- The browser `files.download` happens regardless of the toggle.
- On any Drive failure, print the error and continue to the browser
  download.

**Cell 22 (B4 heading)**: one line mentioning the optional Drive save
toggle inside the cell.

## Error handling

- Serializer: same `float()` coercion as the CSV builders; labels pass
  through verbatim (tabs in labels are not escaped — labels come from a
  fixed class vocabulary with no tabs).
- Notebook cells print explicit success/failure messages; Drive failures
  never block the browser download path.

## Verification

- Unit tests green (17+); `py_compile app.py`.
- Local app relaunch; user clicks the new Audacity button and imports the
  file into Audacity if convenient.
- Notebook: JSON validity + structural assertions (params present,
  cross-references intact). The two Drive behaviors require a real Colab
  session — flagged for the user's next run (Drive mounting cannot be
  exercised locally).

## hf-space sync + redeploy (final phase)

Per the runbook in `2026-06-03-hf-space-zerogpu-design.md`:

1. `git checkout hf-space && git merge main` — expected conflicts:
   - pruned paths re-appearing (`notebooks/`, `tests/`, `docs/`, etc.)
     → `git rm` again;
   - `README.md` / `requirements.txt` → keep ours;
   - `app.py` → feature 1's regions (button row, handler returns/outputs)
     do not overlap the ZeroGPU overlay regions, so this merge is
     expected to be clean or near-clean.
2. `git push --force origin hf-space`.
3. Redeploy via the `hf upload` snapshot (including the musicfm
   submodule export).
4. Verify the Space rebuilds to `RUNNING` and the Single File tab shows
   the new Audacity button.

## Out of scope

- yt-dlp / URL ingestion (explicitly dropped).
- Spotify in any form.
- Per-segment playback, waveform timeline, segment audio export (future
  candidates).
