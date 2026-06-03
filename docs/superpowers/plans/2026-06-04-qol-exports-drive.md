# Audacity Export + Colab Drive Persistence QoL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Audacity label-track export (in all ZIPs + a Single File download button), persist the MuQ HF cache to Drive under the existing Colab `STORE_ON_DRIVE` toggle, and add a timestamped auto-save of Colab batch results to Drive — then merge to `main` and sync/redeploy `hf-space`.

**Architecture:** Feature 1 extends the pure `export_utils` serialization layer (TDD) and wires one new button into `app.py` (handler arity 9→10). Features 2–3 are scripted edits to `notebooks/SongFormer-Colab.ipynb` cells located by stable cell id, with structural assertions. The final phase is controller-driven: merge to `main`, push, then the documented `hf-space` merge + `hf upload` snapshot redeploy.

**Tech Stack:** Python 3.10 (conda env `songformer`), pytest, Gradio 5.38, Colab notebook JSON.

**Spec:** `docs/superpowers/specs/2026-06-04-qol-exports-drive-design.md`

**Working directory:** `/Users/sid/Developer/Python/SongFormer/.claude/worktrees/qol-exports-drive` (branch `worktree-qol-exports-drive`). All commands run here.

**Interpreter:** `/opt/miniconda3/envs/songformer/bin/python`

---

## File Structure

- **Modify** `export_utils.py` — add `segments_to_audacity`; extend `write_exports` with the `"audacity"` artifact.
- **Modify** `tests/test_export_utils.py` — new serializer test; extend the two `write_exports` tests.
- **Modify** `app.py` — sixth download button, 10-value returns, 10-component outputs.
- **Modify** `notebooks/SongFormer-Colab.ipynb` — cells `sec3-checkpoints` (heading), `code-checkpoints`, `b4-download` (heading), `code-download-results`.

---

## Task 1: `segments_to_audacity` (TDD)

**Files:**
- Modify: `export_utils.py`
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Add the failing test** — APPEND to `tests/test_export_utils.py`:

```python
def test_segments_to_audacity():
    segments = [
        {"start": "0.0", "end": "12.34", "label": "intro"},
        {"start": 12.34, "end": 45.67, "label": "verse"},
    ]
    text = export_utils.segments_to_audacity(segments)
    lines = text.strip().split("\n")
    assert lines[0] == "0.000000\t12.340000\tintro"
    assert lines[1] == "12.340000\t45.670000\tverse"
    assert len(lines) == 2
    assert text.endswith("\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_audacity -v`
Expected: FAIL — `AttributeError: module 'export_utils' has no attribute 'segments_to_audacity'`.

- [ ] **Step 3: Add the serializer** — in `export_utils.py`, insert AFTER `segments_to_csv` (before `segments_to_combined_csv`):

```python
def segments_to_audacity(segments) -> str:
    """Build an Audacity label-track file from segment dicts.

    One line per segment: start<TAB>end<TAB>label, seconds with six
    decimals (Audacity's File > Import > Labels format).
    """
    lines = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        lines.append(f"{start:.6f}\t{end:.6f}\t{seg['label']}")
    return "\n".join(lines) + ("\n" if lines else "")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py::test_segments_to_audacity -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: add export_utils.segments_to_audacity"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 2: `write_exports` gains the audacity artifact (TDD)

**Files:**
- Modify: `export_utils.py` (`write_exports`)
- Test: `tests/test_export_utils.py`

- [ ] **Step 1: Extend the two existing tests to fail first**

In `test_write_exports`, after the four basename asserts, ADD:

```python
    assert os.path.basename(paths["audacity"]) == "Song.audacity.txt"
```

and after the MSA round-trip block, ADD:

```python
    with open(paths["audacity"], encoding="utf-8") as f:
        assert f.read() == "0.000000\t1.000000\tintro\n"
```

In `test_write_exports_custom_stem`, after the four basename asserts, ADD:

```python
    assert os.path.basename(paths["audacity"]) == "Song_2.audacity.txt"
```

- [ ] **Step 2: Run to verify they fail**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/test_export_utils.py -k write_exports -v`
Expected: both FAIL with `KeyError: 'audacity'`.

- [ ] **Step 3: Extend `write_exports`** — in `export_utils.py`, change the `paths` dict to:

```python
    paths = {
        "json": os.path.join(out_dir, f"{stem}.json"),
        "msa": os.path.join(out_dir, f"{stem}.msa.txt"),
        "csv": os.path.join(out_dir, f"{stem}.csv"),
        "audacity": os.path.join(out_dir, f"{stem}.audacity.txt"),
        "png": os.path.join(out_dir, f"{stem}.png"),
    }
```

and after the CSV write block, ADD:

```python
    with open(paths["audacity"], "w", encoding="utf-8") as f:
        f.write(segments_to_audacity(segments))
```

(The PNG `fig.savefig` line stays last.)

- [ ] **Step 4: Run the full suite**

Run: `/opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: **17 passed** (16 prior + 1 new from Task 1; the two extended tests count within the 17).

- [ ] **Step 5: Commit**

```bash
git add export_utils.py tests/test_export_utils.py
git commit -m "feat: write_exports emits an Audacity label file in every bundle"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 3: Sixth download button in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add the button** — replace:

```python
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

with:

```python
            # Export / download buttons (populated after analysis)
            with gr.Row():
                download_json_btn = gr.DownloadButton("⬇️ JSON")
                download_msa_btn = gr.DownloadButton("⬇️ MSA (.txt)")
                download_csv_btn = gr.DownloadButton("⬇️ CSV")
                download_audacity_btn = gr.DownloadButton("⬇️ Audacity labels")
                download_png_btn = gr.DownloadButton("⬇️ Plot (.png)")
                download_zip_btn = gr.DownloadButton(
                    "⬇️ Download all (ZIP)", variant="primary"
                )
```

- [ ] **Step 2: Success return 9→10** — in `process_and_analyze`, replace:

```python
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

with:

```python
        return (
            table_data,
            json_format,
            msa_format,
            fig,
            export_paths["json"],
            export_paths["msa"],
            export_paths["csv"],
            export_paths["audacity"],
            export_paths["png"],
            zip_path,
        )
```

- [ ] **Step 3: No-input and error returns 9→10**

Replace `    if audio_file is None:` block's return:

```python
        return None, "", "", None, None, None, None, None, None
```

with:

```python
        return None, "", "", None, None, None, None, None, None, None
```

and the `except` block's:

```python
        return None, "", error_msg, None, None, None, None, None, None
```

with:

```python
        return None, "", error_msg, None, None, None, None, None, None, None
```

- [ ] **Step 4: Outputs list 9→10** — in `analyze_btn.click`, replace the outputs list:

```python
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
```

with:

```python
        outputs=[
            segments_table,
            json_output,
            msa_output,
            plot_output,
            download_json_btn,
            download_msa_btn,
            download_csv_btn,
            download_audacity_btn,
            download_png_btn,
            download_zip_btn,
        ],
```

- [ ] **Step 5: Verify**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py && /opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: compile OK; 17 passed.
Self-check: all three return paths have 10 values; outputs list has 10 components; return order (…csv, audacity, png, zip) matches button order (…csv_btn, audacity_btn, png_btn, zip_btn).

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: Audacity labels download button on the Single File tab"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 4: Notebook — MuQ HF cache on Drive

**Files:**
- Modify: `notebooks/SongFormer-Colab.ipynb` (cells `sec3-checkpoints`, `code-checkpoints`)

- [ ] **Step 1: Apply the scripted edit** — run exactly:

```bash
/opt/miniconda3/envs/songformer/bin/python - <<'PYEOF'
import json

path = "notebooks/SongFormer-Colab.ipynb"
nb = json.load(open(path))
by_id = {c["metadata"].get("id"): c for c in nb["cells"]}

def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def src(cell):
    return "".join(cell["source"])

# --- Heading (sec3-checkpoints): sizes + speed caveat ---
set_src(by_id["sec3-checkpoints"], """\
---
## 3. Download model checkpoints

Models total **~2.7 GB** per session:
- `pretrained_msd.pt` (MusicFM) — 1.23 GB
- `SongFormer.safetensors` — 99.6 MB
- MuQ encoder (HF cache) — 1.33 GB
- `msd_stats.json` — 2.2 KB

> **Optional — persist models on Google Drive:** saves re-downloading ~2.7 GB every session, at the cost of the same Drive space. Note: Drive reads can be *slower* than re-downloading from Hugging Face — this saves bandwidth and avoids download flakiness, not necessarily startup time. The toggle lives inside this collapsed section: expand it and tick **`STORE_ON_DRIVE`** *before* running the cell (with `Run all` it defaults to off).
""")

# --- code-checkpoints: relocate HF cache to Drive inside STORE_ON_DRIVE ---
cell = by_id["code-checkpoints"]
s = src(cell)
anchor = '    print(f"Checkpoints will be stored at: {DRIVE_CKPTS}")\n'
assert anchor in s, "anchor line not found in code-checkpoints"
insertion = anchor + """
    # Persist the MuQ encoder cache (HF hub cache) on Drive too, so the
    # ~1.33 GB MuQ download survives across sessions. Symlink the cache
    # dir (env vars are resolved at import time, too late here).
    HF_CACHE = '/root/.cache/huggingface'
    DRIVE_HF = '/content/drive/MyDrive/SongFormer/hf_cache'
    os.makedirs(DRIVE_HF, exist_ok=True)
    if not os.path.islink(HF_CACHE):
        if os.path.isdir(HF_CACHE):
            for item in os.listdir(HF_CACHE):
                dst = os.path.join(DRIVE_HF, item)
                if not os.path.exists(dst):
                    shutil.move(os.path.join(HF_CACHE, item), dst)
            shutil.rmtree(HF_CACHE)
        os.makedirs(os.path.dirname(HF_CACHE), exist_ok=True)
        os.symlink(DRIVE_HF, HF_CACHE)
    print(f"HF model cache (MuQ) stored at: {DRIVE_HF}")
"""
set_src(cell, s.replace(anchor, insertion))

json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)

# validate
nb2 = json.load(open(path))
j = json.dumps(nb2)
assert "DRIVE_HF" in j and "~2.7 GB" in j and "os.symlink(DRIVE_HF, HF_CACHE)" in j
assert len(nb2["cells"]) == 24
print("Task 4 applied, JSON valid")
PYEOF
```

Expected output: `Task 4 applied, JSON valid`.

- [ ] **Step 2: Sanity-check the inserted python parses** (extract the cell and compile it minus magics):

```bash
/opt/miniconda3/envs/songformer/bin/python - <<'PYEOF'
import json, ast
nb = json.load(open("notebooks/SongFormer-Colab.ipynb"))
cell = next(c for c in nb["cells"] if c["metadata"].get("id") == "code-checkpoints")
code = "".join(l for l in cell["source"] if not l.lstrip().startswith(("!", "%", "#@")))
ast.parse(code)
print("code-checkpoints parses OK")
PYEOF
```

Expected: `code-checkpoints parses OK`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/SongFormer-Colab.ipynb
git commit -m "feat(colab): STORE_ON_DRIVE also persists the MuQ HF cache"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 5: Notebook — auto-save batch results to Drive

**Files:**
- Modify: `notebooks/SongFormer-Colab.ipynb` (cells `b4-download`, `code-download-results`)

- [ ] **Step 1: Apply the scripted edit** — run exactly:

```bash
/opt/miniconda3/envs/songformer/bin/python - <<'PYEOF'
import json

path = "notebooks/SongFormer-Colab.ipynb"
nb = json.load(open(path))
by_id = {c["metadata"].get("id"): c for c in nb["cells"]}

def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)

def src(cell):
    return "".join(cell["source"])

# --- Heading (b4-download): mention the toggle ---
set_src(by_id["b4-download"], """\
### B4. Download results

The cell below zips the results for browser download, and can optionally also save the ZIP to your Drive (`SAVE_TO_DRIVE` toggle — timestamped, never overwrites).
""")

# --- code-download-results: SAVE_TO_DRIVE param + Drive copy ---
cell = by_id["code-download-results"]
s = src(cell)

title = "#@title ⬇️ Zip and download results\n"
assert s.startswith(title), "title line not first in code-download-results"
s = s.replace(
    title,
    title
    + "SAVE_TO_DRIVE = False  #@param {type:\"boolean\"}\n"
    + "#@markdown Tick to also copy the ZIP to `MyDrive/SongFormer/results/` (timestamped).\n",
    1,
)

dl = "    files.download(ZIP_PATH)\n"
assert dl in s, "files.download line not found"
drive_block = """\
    if SAVE_TO_DRIVE:
        try:
            if not os.path.exists('/content/drive/MyDrive'):
                from google.colab import drive
                drive.mount('/content/drive')
            from datetime import datetime
            results_dir = '/content/drive/MyDrive/SongFormer/results'
            os.makedirs(results_dir, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            drive_zip = os.path.join(results_dir, f'songformer_results_{stamp}.zip')
            shutil.copy2(ZIP_PATH, drive_zip)
            print(f"Saved to Drive: {drive_zip}")
        except Exception as e:
            print(f"Drive save failed ({e}); continuing with browser download.")
    files.download(ZIP_PATH)
"""
s = s.replace(dl, drive_block, 1)
set_src(cell, s)

json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)

# validate
nb2 = json.load(open(path))
j = json.dumps(nb2)
assert "SAVE_TO_DRIVE" in j and "songformer_results_{stamp}" in j
assert j.count("files.download(ZIP_PATH)") == 1
print("Task 5 applied, JSON valid")
PYEOF
```

Expected output: `Task 5 applied, JSON valid`.

- [ ] **Step 2: Sanity-check the cell parses** (magics/forms stripped):

```bash
/opt/miniconda3/envs/songformer/bin/python - <<'PYEOF'
import json, ast
nb = json.load(open("notebooks/SongFormer-Colab.ipynb"))
cell = next(c for c in nb["cells"] if c["metadata"].get("id") == "code-download-results")
code = "".join(l for l in cell["source"] if not l.lstrip().startswith(("!", "%", "#@")))
code = code.replace('SAVE_TO_DRIVE = False  ', 'SAVE_TO_DRIVE = False')
ast.parse(code)
print("code-download-results parses OK")
PYEOF
```

Expected: `code-download-results parses OK`.
(Note: the `#@param` suffix lives on the same line as the assignment, so the line survives the magic-strip filter and parses as plain Python with a trailing comment.)

- [ ] **Step 3: Commit**

```bash
git add notebooks/SongFormer-Colab.ipynb
git commit -m "feat(colab): optional timestamped save of batch results ZIP to Drive"
```

End the commit body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 6 (controller): verify, merge to `main`, push

- [ ] Full suite + compile in the worktree: 17 passed, `py_compile app.py` OK.
- [ ] Relaunch the app locally from the worktree; user clicks **⬇️ Audacity labels** and confirms the file imports into Audacity (tab-separated, opens as a label track).
- [ ] Merge: from the MAIN checkout (`/Users/sid/Developer/Python/SongFormer`): `git checkout main`, delete the stale empty `qol-exports-drive` branch, `git merge --ff-only worktree-qol-exports-drive`, re-run tests on merged main, `git push origin main`, delete the worktree branch.

## Task 7 (controller): hf-space sync + redeploy

- [ ] `git checkout hf-space && git merge main` — resolve per runbook: re-`git rm` pruned paths (`notebooks/`, `tests/`, `docs/`, `conftest.py`, etc. if re-added), keep ours for `README.md`/`requirements.txt`; `app.py` expected clean/near-clean (feature regions don't overlap overlay regions). Verify after merge: `import spaces` + `@spaces.GPU` still present; audacity button present; `py_compile`.
- [ ] `git push --force origin hf-space`.
- [ ] Redeploy: `git archive` snapshot + musicfm submodule export + `hf upload ... --delete "*"` per runbook.
- [ ] Watch Space → `RUNNING`; confirm the Single File tab shows the Audacity button.

---

## Self-Review notes (author checklist — already applied)

- **Spec coverage:** serializer + format (Task 1); `"audacity"` artifact in all bundles (Task 2 — single ZIP/batch propagate via `write_exports`); sixth button + 10-arity (Task 3); MuQ cache symlink + heading caveat (Task 4); SAVE_TO_DRIVE timestamped + browser download preserved + failure tolerance (Task 5); verification + merge (Task 6); hf-space sync/redeploy (Task 7). All covered.
- **Placeholder scan:** none.
- **Consistency:** `"audacity"` key written in Task 2, consumed in Task 3; return order (csv, audacity, png, zip) matches outputs order; notebook edits keyed by the stable cell ids assigned earlier (`sec3-checkpoints`, `code-checkpoints`, `b4-download`, `code-download-results`); Task 5's asserts match Task 4's output state (title line first, single `files.download`).
- **Arithmetic:** tests 16 → 17 (one new test; two tests extended in place).
