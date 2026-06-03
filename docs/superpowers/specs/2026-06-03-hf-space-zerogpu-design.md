# HF Space ZeroGPU Deployment — Design

**Date:** 2026-06-03
**Status:** Approved design — pending implementation plan
**Branches:** `main` (one neutral fix only) and a regenerated `hf-space`
(fresh from `main` + one overlay commit; old `hf-space` superseded by
force-push).

## Goal

Bring the Hugging Face Space (`SidSaxena/SongFormer`, ZeroGPU hardware) up
to date with everything on `main` — single-file export, the Batch tab,
incremental ZIP, detail viewer — while keeping `main` completely free of
ZeroGPU-specific code. Future feature syncs become a routine
`git merge main` instead of re-porting by hand.

## Verified platform facts (sources)

From the [ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu),
the [quickstart](https://huggingface.co/spaces/cbensimon/zerogpu-quickstart),
[hysts' ZeroGPU notes](https://github.com/hysts/huggingface-skills/blob/main/skills/huggingface-zerogpu/SKILL.md),
and the [AoTI blog](https://huggingface.co/blog/zerogpu-aoti):

- `@spaces.GPU` requests a GPU per call and releases it on completion.
  Outside ZeroGPU it returns the function unchanged (effect-free).
- A CUDA **emulation mode** is active outside decorated functions:
  `torch.cuda.is_available()` is True at module level, and models **must**
  be placed on `cuda` at startup (lazy placement inside the decorated
  function is explicitly discouraged). `main`'s `get_device()` therefore
  picks `cuda` on the Space and `mps` locally with **no change**.
- `duration`: default 60s; supports a **callable** receiving the decorated
  function's args and returning seconds (dynamic duration). Shorter
  reservations get better queue priority. Quota burns by *effective*
  runtime, not reservation.
- Daily GPU quotas: unauthenticated 2 min, free account 5 min, PRO 40 min.
- Hardware: half NVIDIA RTX Pro 6000 Blackwell, 48 GB (default `large`).
- Supported: Gradio 4+, **torch 2.8.0–2.11**, Python 3.10.13/3.12.12.
  `torch.compile` unsupported.
- The `spaces` package is **preinstalled** on ZeroGPU Spaces and must NOT
  be pinned in `requirements.txt` (a conflicting pin breaks pip).
- Each `@spaces.GPU` entry has per-call overhead (pickle boundary, worker
  warm-up, queue pass). Per-file decoration is still correct for us: each
  file is minutes of inference (overhead amortizes), live batch progress
  requires the generator staying in the main process, and generator
  support inside `@spaces.GPU` is undocumented.
- GPU-boundary rules already satisfied by `process_audio`: picklable args
  (a path) and returns (CPU tensors via `torch.from_numpy`, lists); no
  mutable-global writes; no fixed output paths inside the GPU function.
- AoT compilation (AoTI): worthwhile for shape-stable heavy transformers.
  Our encoders (MuQ, MusicFM) see highly variable-length chunks and are
  non-standard export targets — **out of scope**; see Future work.

## Changes to `main` (platform-neutral only)

One small commit:
- Fix the header links in `app.py`'s HTML block (real arXiv id
  `2510.02797`, Space URL `SidSaxena/SongFormer`, working Discord invite
  `p5uBryC4Zs`) — parity with what old `hf-space` already fixed.

Nothing else. No `spaces` import, no decorators, no Space conditionals.

## The `hf-space` overlay (fresh branch from `main`, a short commit series)

### 1. `app.py` ZeroGPU patch (this branch only — it never runs locally)

- `import spaces` (direct; the package is preinstalled on the Space).
- Dynamic duration callable, placed immediately above `process_audio`:

  ```python
  def _gpu_duration(audio_path, win_size=420, hop_size=420, num_classes=128):
      """Estimate GPU seconds for one file (ZeroGPU dynamic duration)."""
      try:
          audio_secs = librosa.get_duration(path=audio_path)
      except Exception:
          return 120
      return int(min(300, max(60, 30 + 0.2 * audio_secs)))

  @spaces.GPU(duration=_gpu_duration)
  def process_audio(audio_path, win_size=420, hop_size=420, num_classes=128):
      ...
  ```

  Constants (30 s base + 0.2 s/audio-second, clamped 60–300) are
  deliberately conservative; tune after observing real timings on the
  Blackwell hardware.
- Launch block: replace the local launch (port 7891 / `SONGFORMER_SHARE`)
  with plain `demo.launch()` — Spaces inject their own server settings; an
  explicit port breaks the platform health check.
- Batch quota-abort in `process_batch`'s except handler: if
  `"quota" in str(e).lower()`, mark all remaining queued rows
  `⏭️ skipped (GPU quota)`, yield once, and `break` (the post-loop
  normalize yield still runs, so the partial ZIP stays downloadable).
  Heuristic on the message text — ZeroGPU does not document a stable
  exception class.
- One Markdown line added to the Batch tab: this Space runs on ZeroGPU;
  every file consumes daily GPU quota (2–40 min by account tier); the ZIP
  always contains everything analyzed so far.

### 2. `README.md` → Space metadata

Replace with the Space-frontmatter README (ported from old `hf-space`):
`sdk: gradio`, `sdk_version: 5.38.2`, `python_version: "3.10"`,
`app_file: app.py`, title/emoji/colors/tags/short_description, plus brief
usage text mentioning both tabs.

### 3. `requirements.txt` → Space pins

Replace with old `hf-space`'s proven set verbatim (`torch==2.8.0`,
`torchaudio==2.8.0`, pinned ML stack, unpinned `gradio`, `setuptools<70`).
No `spaces` entry (preinstalled). torch 2.8.0 is within the verified
supported range.

### 4. Pruning (parity with old `hf-space`, two exceptions)

Delete: `.dockerignore`, `Dockerfile`, `docker-compose.yml`,
`.gitmodules`, `README_ZH.md`, `notebooks/`, `requirements-colab.txt`,
`requirements-dev.txt`, `requirements-docker.txt`,
`requirements-local.txt`, `requirements-no-build.txt`, `setup-linux.sh`,
`setup-mac.sh`, `setup-windows.bat`, `songformer-env.yml`,
`songformer-pip-freeze.txt`, `figs/`, `docs/`, `tests/`, `conftest.py`,
`src/SongFormer/ckpts/md5sum.txt`,
`src/data_pipeline/obtain_SSL_representation/MuQ/*`,
`src/data_pipeline/obtain_SSL_representation/MusicFM/*` (the four
embedding scripts and their .sh wrappers, as before).

**Kept (deviations from old hf-space):** `export_utils.py` (required by
`app.py`) and `LICENSE` (a public mirror should carry its license). Also
kept: `main`'s `infer.sh` / `infer/infer.py` (old hf-space carried
simplified copies; the full versions are inert on the Space and keeping
them removes a merge-conflict source).

## Branch mechanics

- Build: `git checkout -b hf-space-new main` → apply overlay commit →
  `git branch -f hf-space hf-space-new` → `git push --force origin hf-space`.
  Old `hf-space` tip (`29ebdeb`) remains reachable by SHA for rollback.
- Deploy — **via `hf upload` snapshot, NOT git push.** (Verified during the
  first deployment: the Hub's pre-receive hooks reject a direct git push
  twice over — the branch history carries `figs/aslp.png` >10 MiB, and the
  example .mp3/.wav files must be stored via Xet/LFS, which `hf upload`
  handles automatically.)

  ```
  rm -rf /tmp/songformer-space-deploy && mkdir -p /tmp/songformer-space-deploy
  git archive hf-space | tar -x -C /tmp/songformer-space-deploy
  # git archive does NOT include submodules: src/third_party/{MuQ,musicfm}
  # are gitlinks. The app imports `musicfm` from there (MuQ comes from the
  # pip `muq` package — its 26MB submodule is NOT needed). Export it too:
  mkdir -p /tmp/songformer-space-deploy/src/third_party/musicfm
  git -C src/third_party/musicfm archive HEAD \
    | tar -x -C /tmp/songformer-space-deploy/src/third_party/musicfm
  hf upload SidSaxena/SongFormer /tmp/songformer-space-deploy . \
    --repo-type space --delete "*" \
    --commit-message "Deploy hf-space @ $(git rev-parse --short hf-space)"
  ```

  `git archive` exports tracked files only (no local artifacts); the
  `--delete "*"` makes the upload a true sync (remote files absent from
  the export are removed — which is also why the musicfm step is
  mandatory: the first deploy omitted it and the Space failed at startup
  with `ModuleNotFoundError: No module named 'musicfm'`). Auth via
  `hf auth login` (token needs write access to the Space). Rollback:
  re-run the same upload from a checkout of any earlier `hf-space` SHA,
  or revert the commit in the Space UI.

## Sync runbook (future `main` features → Space)

```
git checkout hf-space
git merge main
```

Expected conflicts and resolutions:
- `README.md`, `requirements.txt` → keep ours (`git checkout --ours`).
- `app.py` → conflicts only when `main` changed the overlay's regions
  (imports, the lines around `process_audio`, the launch block, the batch
  except-handler/Markdown). Resolution: take `main`'s logic, re-apply the
  overlay's additions on top. The overlay deliberately keeps these
  insertions small and localized.
- Files pruned by the overlay re-appear as "added by them" → delete again
  (`git rm`).

Then re-deploy with the `hf upload` snapshot from "Branch mechanics" above
(and `git push --force origin hf-space` to keep GitHub in sync).

## Error handling / edge cases

- **Quota exhaustion mid-batch**: handled by quota-abort; remaining files
  marked skipped; partial ZIP remains valid.
- **Per-call GPU queue waits mid-batch**: visible as a longer
  `🔄 processing…` state; no code change needed.
- **Cold start**: `download_all()` fetches ~1.3 GB of checkpoints on each
  Space restart (existing behavior; startup takes minutes). Persistent
  storage is out of scope.
- **Examples**: `gr.Examples` here has no `fn`/`outputs`, so no example
  caching runs at startup (no GPU needed at boot).
- **`gr.Examples` paths**: relative `examples/...` resolve because app.py
  `os.chdir`s into `src/SongFormer` — same as local; example files are
  committed on `main` already.

## Verification

- Local (before push): `python -m pytest tests/` still 16-pass on `main`;
  `py_compile` of the overlay `app.py` on `hf-space`; grep-level check
  that the overlay only touches the declared regions.
- On the Space (the only place ZeroGPU runs): after push, watch build
  logs; then verify live: single-file analyze + 5 downloads; batch of 2–3
  examples with live status; mid-run ZIP; detail viewer + audio player;
  duration behavior (no "GPU task aborted" errors on a ~4 min song).
- Rollback path verified by keeping the old tip SHA in the spec (29ebdeb).

## Future work (explicitly out of scope)

- **AoTI compilation** of the MuQ/MusicFM encoder forward passes
  (`spaces.aoti_capture` around `muq_model(...)` /
  `musicfm_model.get_predictions(...)`, `torch.export` with bounded
  `Dim` ranges for the 30 s / 420 s chunk shapes, `spaces.aoti_compile`).
  Revisit if Space latency matters; expected ≤1.3–1.8× on the encoder
  portion, at high engineering risk for these architectures.
- Persistent storage for checkpoints (faster cold boots).
- Tuning `_gpu_duration` constants from observed Space timings.
- README overhaul on both branches (reorganize, declutter) and Colab
  badge + notebook refresh — explicitly deferred until after this
  deployment lands (user request).
