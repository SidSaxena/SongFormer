# HF Space ZeroGPU Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate the `hf-space` branch from `main` plus a small ZeroGPU overlay (decorators, Space README/requirements, pruning) and deploy it to the `SidSaxena/SongFormer` Space — keeping `main` free of all ZeroGPU code.

**Architecture:** One neutral commit lands on `main` (header-link fixes). A fresh branch from `main` then receives three overlay commits: (1) the `app.py` ZeroGPU patch, (2) Space `README.md` + `requirements.txt`, (3) pruning. The branch replaces `hf-space` (force-push) and is pushed to the HF Space git remote.

**Tech Stack:** Gradio 5.38, `spaces` package (preinstalled on the Space — never in requirements), torch 2.8.0 (Space pin), git.

**Spec:** `docs/superpowers/specs/2026-06-03-hf-space-zerogpu-design.md`

**Interpreter for checks:** `/opt/miniconda3/envs/songformer/bin/python`

**Note on testing:** This work is platform glue with no new pure logic — no new unit tests. Gates are: the existing 16-test suite stays green on `main`-derived code, `py_compile` after every `app.py` edit, and live verification on the Space (the only environment where ZeroGPU runs).

---

## Task 1: Neutral link fixes on `main` (via `hf-space-prep`)

**Files:**
- Modify: `app.py` (links HTML block inside `gr.Blocks`)

Work on the existing `hf-space-prep` branch (it already carries the spec commit).

- [ ] **Step 1: Fix the header links**

In `app.py`, replace the body of the links `gr.HTML` block. Replace this exact block:

```python
        <div class="links-container">
            <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python">
            <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightblue" alt="License">
            <a href="https://arxiv.org/abs/">
            <img src="https://img.shields.io/badge/arXiv-com.svg?logo=arXiv" alt="arXiv">
            </a>
            <a href="https://github.com/ASLP-lab/SongFormer">
            <img src="https://img.shields.io/badge/GitHub-SongFormer-black" alt="GitHub">
            </a>
            <a href="https://huggingface.co/spaces/ASLP-lab/SongFormer">
            <img src="https://img.shields.io/badge/HuggingFace-space-yellow" alt="HuggingFace Space">
            </a>
            <a href="https://huggingface.co/ASLP-lab/SongFormer">
            <img src="https://img.shields.io/badge/HuggingFace-model-blue" alt="HuggingFace Model">
            </a>
            <a href="https://huggingface.co/datasets/ASLP-lab/SongFormDB">
            <img src="https://img.shields.io/badge/HF%20Dataset-SongFormDB-green" alt="Dataset SongFormDB">
            </a>
            <a href="https://huggingface.co/datasets/ASLP-lab/SongFormBench">
            <img src="https://img.shields.io/badge/HF%20Dataset-SongFormBench-orange" alt="Dataset SongFormBench">
            </a>
            <a href="https://discord.gg/rwcqh7Em">
            <img src="https://img.shields.io/badge/Discord-join%20us-purple?logo=discord&logoColor=white" alt="Discord">
            </a>
            <a href="http://www.npu-aslp.org/">
            <img src="https://img.shields.io/badge/🏫-ASLP-grey?labelColor=lightgrey" alt="ASLP">
            </a>
        </div>
```

with:

```python
        <div class="links-container">
            <img src="https://img.shields.io/badge/Python-3.10-brightgreen" alt="Python">
            <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightblue" alt="License">
            <a href="https://arxiv.org/abs/2510.02797">
            <img src="https://img.shields.io/badge/arXiv-2510.02797-blue" alt="arXiv">
            </a>
            <a href="https://github.com/ASLP-lab/SongFormer">
            <img src="https://img.shields.io/badge/GitHub-SongFormer-black" alt="GitHub">
            </a>
            <a href="https://huggingface.co/spaces/SidSaxena/SongFormer">
            <img src="https://img.shields.io/badge/HuggingFace-space-yellow" alt="HuggingFace Space">
            </a>
            <a href="https://huggingface.co/ASLP-lab/SongFormer">
            <img src="https://img.shields.io/badge/HuggingFace-model-blue" alt="HuggingFace Model">
            </a>
            <a href="https://huggingface.co/datasets/ASLP-lab/SongFormDB">
            <img src="https://img.shields.io/badge/HF%20Dataset-SongFormDB-green" alt="Dataset SongFormDB">
            </a>
            <a href="https://huggingface.co/datasets/ASLP-lab/SongFormBench">
            <img src="https://img.shields.io/badge/HF%20Dataset-SongFormBench-orange" alt="Dataset SongFormBench">
            </a>
            <a href="https://discord.gg/p5uBryC4Zs">
            <img src="https://img.shields.io/badge/Discord-join%20us-purple?logo=discord&logoColor=white" alt="Discord">
            </a>
            <a href="http://www.npu-aslp.org/">
            <img src="https://img.shields.io/badge/🏫-ASLP-grey?labelColor=lightgrey" alt="ASLP">
            </a>
        </div>
```

(Three changes: arXiv href + badge now carry `2510.02797`; Space link points at `SidSaxena/SongFormer`; Discord invite is `p5uBryC4Zs`.)

- [ ] **Step 2: Verify**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py && /opt/miniconda3/envs/songformer/bin/python -m pytest tests/ -q`
Expected: compile OK; 16 passed.

- [ ] **Step 3: Commit, merge to `main`, push**

```bash
git add app.py
git commit -m "fix: correct arXiv, Space, and Discord links in app header"
git checkout main
git merge --ff-only hf-space-prep
git push origin main
git branch -d hf-space-prep
```

End the commit message body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 2: Overlay commit 1 — `app.py` ZeroGPU patch

**Files:**
- Modify: `app.py` (on a new branch `hf-space-new`)

- [ ] **Step 1: Create the overlay branch from updated `main`**

```bash
git checkout -b hf-space-new main
```

- [ ] **Step 2: Add the `spaces` import**

In `app.py`, the project imports end with:

```python
from utils.fetch_pretrained import download_all

import export_utils
```

Replace with:

```python
from utils.fetch_pretrained import download_all

import export_utils

# ZeroGPU (Hugging Face Spaces). Preinstalled on the Space; this branch
# is Space-only and never runs locally.
import spaces
```

- [ ] **Step 3: Add the dynamic-duration callable and decorate `process_audio`**

Find the line `def process_audio(audio_path, win_size=420, hop_size=420, num_classes=128):` and insert ABOVE it:

```python
def _gpu_duration(audio_path, win_size=420, hop_size=420, num_classes=128):
    """Estimate GPU seconds for one file (ZeroGPU dynamic duration).

    Conservative: 30s base + 0.2s per audio second, clamped to [60, 300].
    Tune the constants from observed Space timings.
    """
    try:
        audio_secs = librosa.get_duration(path=audio_path)
    except Exception:
        return 120
    return int(min(300, max(60, 30 + 0.2 * audio_secs)))


@spaces.GPU(duration=_gpu_duration)
```

so that the decorator line sits directly above `def process_audio(...)`.

- [ ] **Step 4: Simplify the launch for Spaces**

Replace:

```python
    # Launch interface
    # Use SONGFORMER_SHARE=1 to get a public gradio.live URL
    share = os.environ.get("SONGFORMER_SHARE", "0") == "1"
    server_name = "0.0.0.0" if share else "127.0.0.1"
    demo.launch(share=share, server_name=server_name, server_port=7891, debug=True)
```

with:

```python
    # Launch interface (Spaces injects its own server settings; an explicit
    # port would break the platform health check)
    demo.launch()
```

- [ ] **Step 5: Batch quota-abort**

In `process_batch`, the per-file except handler currently reads:

```python
        except Exception as e:
            import traceback

            print(f"Batch error for {stem}:\n{traceback.format_exc()}")
            status_rows[idx] = [stem, "❌ " + str(e)[:80], 0, ""]
```

Replace with:

```python
        except Exception as e:
            import traceback

            print(f"Batch error for {stem}:\n{traceback.format_exc()}")
            status_rows[idx] = [stem, "❌ " + str(e)[:80], 0, ""]
            # ZeroGPU quota exhausted: every remaining file would fail the
            # same way, so skip them. (Message heuristic — ZeroGPU does not
            # document a stable exception class.)
            if "quota" in str(e).lower():
                for j in range(idx + 1, len(queue)):
                    status_rows[j] = [queue[j][1], "⏭️ skipped (GPU quota)", "", ""]
                yield (
                    status_rows,
                    gr.update(),
                    gr.update(choices=list(results.keys())),
                    results,
                )
                break
```

(The `else:` clause that follows the except is untouched; after `break`, the post-loop normalize yield still runs, so the partial ZIP stays downloadable.)

- [ ] **Step 6: Batch tab quota note**

The Batch tab Markdown currently reads:

```python
            gr.Markdown(
                "Upload multiple audio files, analyze them sequentially, "
                "and download all results as a single ZIP."
            )
```

Replace with:

```python
            gr.Markdown(
                "Upload multiple audio files, analyze them sequentially, "
                "and download all results as a single ZIP.\n\n"
                "*This Space runs on ZeroGPU: each file consumes your daily "
                "GPU quota (2–40 min depending on account tier). The ZIP "
                "below always contains everything analyzed so far.*"
            )
```

- [ ] **Step 7: Verify and commit**

Run: `/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py`
Expected: compile OK. (Do NOT run app.py — `import spaces` is absent locally; compile does not import.)

Also confirm scope: `git diff main --stat` shows ONLY `app.py`, and `grep -c "spaces" app.py` shows the import + decorator (plus the HF links containing "spaces" in URLs).

```bash
git add app.py
git commit -m "feat(space): ZeroGPU support - dynamic-duration GPU decorator, Space launch, batch quota-abort"
```

End the commit message body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 3: Overlay commit 2 — Space `README.md` + `requirements.txt`

**Files:**
- Replace: `README.md`
- Replace: `requirements.txt`

- [ ] **Step 1: Replace `README.md`** with exactly:

````markdown
---
title: SongFormer
emoji: "\U0001F3B5"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.38.2
python_version: "3.10"
app_file: app.py
pinned: false
tags:
  - music-structure-annotation
  - transformer
short_description: State-of-the-art music analysis with multi-scale datasets
fullWidth: true
---

<p align="center">
  <img src="https://github.com/ASLP-lab/SongFormer/blob/main/figs/logo.png?raw=true" width="50%" />
</p>

# SONGFORMER: SCALING MUSIC STRUCTURE ANALYSIS WITH HETEROGENEOUS SUPERVISION

![Python](https://img.shields.io/badge/Python-3.10-brightgreen)
![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightblue)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2510.02797-blue)](https://arxiv.org/abs/2510.02797)
[![GitHub](https://img.shields.io/badge/GitHub-SongFormer-black)](https://github.com/ASLP-lab/SongFormer)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-space-yellow)](https://huggingface.co/spaces/SidSaxena/SongFormer)
[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-model-blue)](https://huggingface.co/ASLP-lab/SongFormer)
[![Dataset SongFormDB](https://img.shields.io/badge/HF%20Dataset-SongFormDB-green)](https://huggingface.co/datasets/ASLP-lab/SongFormDB)
[![Dataset SongFormBench](https://img.shields.io/badge/HF%20Dataset-SongFormBench-orange)](https://huggingface.co/datasets/ASLP-lab/SongFormBench)
[![Discord](https://img.shields.io/badge/Discord-join%20us-purple?logo=discord&logoColor=white)](https://discord.gg/p5uBryC4Zs)
[![lab](https://img.shields.io/badge/%F0%9F%8F%AB-ASLP-grey?labelColor=lightgrey)](http://www.npu-aslp.org/)

Chunbo Hao<sup>&ast;</sup>, Ruibin Yuan<sup>&ast;</sup>, Jixun Yao, Qixin Deng, Xinyi Bai, Wei Xue, Lei Xie<sup>&dagger;</sup>

----

**For more information, please visit our [github repository](https://github.com/ASLP-lab/SongFormer)**

SongFormer is a music structure analysis framework that leverages multi-resolution self-supervised representations and heterogeneous supervision, accompanied by the large-scale multilingual dataset SongFormDB and the high-quality benchmark SongFormBench to foster fair and reproducible research.

**This Space offers two modes:** analyze a *Single File* with downloadable results (JSON / MSA / CSV / plot / ZIP), or use the *Batch* tab to process multiple files with live per-file status, a combined ZIP (downloadable mid-run), and a per-file inspector with audio playback. Runs on ZeroGPU — each analyzed file consumes daily GPU quota.

![](https://github.com/ASLP-lab/SongFormer/blob/main/figs/songformer.png?raw=true)

## Citation

If our work and codebase is useful for you, please cite as:

```
@misc{hao2025songformer,
  title         = {SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision},
  author        = {Chunbo Hao and Ruibin Yuan and Jixun Yao and Qixin Deng and Xinyi Bai and Wei Xue and Lei Xie},
  year          = {2025},
  eprint        = {2510.02797},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url           = {https://arxiv.org/abs/2510.02797}
}
```

## License

Our code is released under CC-BY-4.0 License.
````

(This is old `hf-space`'s README verbatim plus one new paragraph — "This Space offers two modes…" — describing the tabs. Deeper README reorganization is deferred by user request.)

- [ ] **Step 2: Replace `requirements.txt`** with exactly (old `hf-space`'s proven set, verbatim — note: NO `spaces` entry, it is preinstalled and pinning it breaks pip):

```text
# Core Deep Learning Framework (torch>=2.8.0 required for ZeroGPU)
torch==2.8.0
torchaudio==2.8.0

# ML/DL Libraries
transformers==4.51.1
accelerate==1.5.2
safetensors==0.5.3

# Scientific Computing
numpy==1.25.0
scipy==1.15.2

# Audio Processing
librosa==0.11.0
soundfile==0.13.1

# Configuration
omegaconf==2.3.0

# Deep Learning Utils
einops==0.8.1
x-transformers==2.4.14
ema-pytorch==0.7.7

# Visualization
matplotlib==3.10.1

# Music Structure Analysis
msaf==0.1.80
muq==0.1.0

# App
gradio
tqdm==4.67.1
setuptools<70
```

- [ ] **Step 3: Commit**

```bash
git add README.md requirements.txt
git commit -m "feat(space): Space README metadata and pinned requirements"
```

End the commit message body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 4: Overlay commit 3 — pruning

**Files:** deletions only.

- [ ] **Step 1: Remove non-Space files**

```bash
git rm -q .dockerignore Dockerfile docker-compose.yml .gitmodules README_ZH.md \
  requirements-colab.txt requirements-dev.txt requirements-docker.txt \
  requirements-local.txt requirements-no-build.txt \
  setup-linux.sh setup-mac.sh setup-windows.bat \
  songformer-env.yml songformer-pip-freeze.txt conftest.py \
  src/SongFormer/ckpts/md5sum.txt
git rm -rq notebooks figs docs tests \
  src/data_pipeline/obtain_SSL_representation/MuQ \
  src/data_pipeline/obtain_SSL_representation/MusicFM
```

KEEP (do not remove): `export_utils.py` (app.py imports it), `LICENSE`, `src/SongFormer/infer.sh`, `src/SongFormer/infer/infer.py`, `src/SongFormer/examples/`.

- [ ] **Step 2: Verify the tree**

```bash
git status --porcelain | head -30
ls   # expect: LICENSE README.md app.py export_utils.py requirements.txt src/ (+untracked local artifacts)
/opt/miniconda3/envs/songformer/bin/python -m py_compile app.py export_utils.py
```

Expected: only deletions staged; `export_utils.py` and `LICENSE` present; compile OK.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(space): prune non-Space files from deployment branch"
```

End the commit message body with (after a blank line):
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

---

## Task 5: Replace `hf-space` and push to origin

- [ ] **Step 1: Record the old tip for rollback, then swap**

```bash
git rev-parse origin/hf-space   # note this SHA (expected 29ebdeb...) for rollback
git branch -f hf-space hf-space-new
git checkout hf-space
git branch -d hf-space-new
```

- [ ] **Step 2: Force-push to origin**

```bash
git push --force origin hf-space
```

Expected: `+ <old>...<new> hf-space -> hf-space (forced update)`.

---

## Task 6: Deploy to the HF Space and verify live

This task is interactive (auth + browser verification). The controller/user drives it; do not dispatch a subagent for Step 3 onward.

- [ ] **Step 1: Check HF auth**

```bash
hf auth whoami 2>/dev/null || echo "NOT LOGGED IN"
```

If not logged in, the user runs `hf auth login` (or provides a write token; with an HTTPS remote, git will prompt — username is the HF username, password is a `write`-scoped token).

- [ ] **Step 2: Add the Space remote (once) and push**

```bash
git remote add space https://huggingface.co/spaces/SidSaxena/SongFormer 2>/dev/null || true
git push --force space hf-space:main
```

- [ ] **Step 3: Watch the build**

Open https://huggingface.co/spaces/SidSaxena/SongFormer — check Logs → Build for pip resolution errors, then Runtime for `download_all` checkpoint fetching (~1.3 GB; several minutes) and `Models loaded successfully!`.

- [ ] **Step 4: Live verification checklist**

1. Single File tab: analyze an example; table/JSON/MSA/plot render; all 5 download buttons serve files.
2. Batch tab: upload 2–3 examples; statuses flip queued → processing → ✅; ZIP button activates after first file with live count; mid-run ZIP download works; detail viewer + audio player work.
3. Duration: a ~4-minute song completes without a "GPU task aborted" error (if aborted, raise `_gpu_duration` constants).
4. Quota note visible in the Batch tab.
5. (If quota allows) burn-down behavior: on quota exhaustion the batch marks remaining files `⏭️ skipped (GPU quota)` instead of erroring file-by-file.

- [ ] **Step 5: Rollback (only if needed)**

```bash
git push --force space <old-sha>:main
```

---

## Self-Review notes (author checklist — already applied)

- **Spec coverage:** main neutral links fix (Task 1); `import spaces` + dynamic duration + decorator (Task 2 Steps 2–3); Space launch (Step 4); quota-abort + note (Steps 5–6); Space README/requirements (Task 3); pruning incl. the two keep-exceptions (Task 4); branch swap + origin push (Task 5); deploy + live verification + rollback (Task 6). AoTI and README-overhaul deferrals need no tasks. All covered.
- **Placeholder scan:** none — every step has exact content/commands.
- **Consistency:** `_gpu_duration` signature mirrors `process_audio`'s exactly (required by ZeroGPU's dynamic-duration contract); quota-abort yield is a 4-tuple matching the established outputs; pruning list matches the spec's, with the same keep-list.
- **Safety:** Task 2 Step 7 explicitly forbids running `app.py` locally on this branch (no `spaces` package); compile-only gates.
