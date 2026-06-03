import os
import sys

os.chdir(os.path.join("src", "SongFormer"))
sys.path.append(os.path.join("..", "third_party"))
sys.path.append(".")

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import gradio as gr
import torch
import librosa
import json
import math
import importlib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe for rendering plots off the main thread
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from argparse import Namespace
from omegaconf import OmegaConf
from ema_pytorch import EMA
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from postprocessing.functional import postprocess_functional_structure
from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from utils.fetch_pretrained import download_all

import export_utils

# ZeroGPU (Hugging Face Spaces). Preinstalled on the Space; this branch
# is Space-only and never runs locally.
import spaces

# Constants
MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")
BEFORE_DOWNSAMPLING_FRAME_RATES = 25
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333
DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]
TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000

# Hardware-aware usage note shown on both tabs. ZeroGPU containers set
# SPACES_ZERO_GPU; without it the Space is on plain CPU hardware.
if os.environ.get("SPACES_ZERO_GPU"):
    USAGE_NOTE = (
        "*Running on ZeroGPU: each analyzed file consumes your daily GPU "
        "quota — anonymous visitors 2 min, free accounts 5 min, PRO 40 min, "
        "Team/Enterprise members 40/60 min. Remaining quota also sets your "
        "queue priority.*"
    )
else:
    USAGE_NOTE = (
        "*Running on CPU hardware: analysis takes a few minutes per song. "
        "On ZeroGPU hardware each file would consume daily GPU quota "
        "(anonymous 2 min, free 5 min, PRO 40 min).*"
    )

# Global model variables
muq_model = None
musicfm_model = None
msa_model = None
device = None


def get_device():
    """Select the best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clear_device_cache(device):
    """Clear GPU memory cache for the given device type."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint from path"""
    if device is None:
        device = "cpu"

    if checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")
    return checkpoint


def initialize_models(model_name: str, checkpoint: str, config_path: str):
    """Initialize all models"""
    global muq_model, musicfm_model, msa_model, device

    # Set device
    device = get_device()

    # Load MuQ
    muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq_model = muq_model.to(device).eval()

    # Load MusicFM
    musicfm_model = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm_model = musicfm_model.to(device).eval()

    # Load MSA model
    module = importlib.import_module("models." + str(model_name))
    Model = getattr(module, "Model")
    hp = OmegaConf.load(os.path.join("configs", config_path))
    msa_model = Model(hp)

    ckpt = load_checkpoint(checkpoint_path=os.path.join("ckpts", checkpoint))
    if ckpt.get("model_ema", None) is not None:
        model_ema = EMA(msa_model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        msa_model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        msa_model.load_state_dict(ckpt["model"])

    msa_model.to(device).eval()

    return hp


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
def process_audio(audio_path, win_size=420, hop_size=420, num_classes=128):
    """Process audio file and return structure analysis results"""
    global muq_model, musicfm_model, msa_model, device

    if muq_model is None:
        hp = initialize_models()
    else:
        hp = OmegaConf.load(os.path.join("configs", "SongFormer.yaml"))

    # Load audio
    wav, sr = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
    audio = torch.tensor(wav).to(device)

    # Prepare output
    total_len = (
        (audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR * TIME_DUR
    ) + TIME_DUR
    total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

    logits = {
        "function_logits": np.zeros([total_frames, num_classes]),
        "boundary_logits": np.zeros([total_frames]),
    }
    logits_num = {
        "function_logits": np.zeros([total_frames, num_classes]),
        "boundary_logits": np.zeros([total_frames]),
    }

    # Prepare label masks
    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    lens = 0
    i = 0

    with torch.no_grad():
        while True:
            start_idx = i * INPUT_SAMPLING_RATE
            end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
            if start_idx >= audio.shape[-1]:
                break
            if end_idx - start_idx <= 1024:
                continue

            audio_seg = audio[start_idx:end_idx]

            # Get embeddings
            muq_output = muq_model(audio_seg.unsqueeze(0), output_hidden_states=True)
            muq_embd_420s = muq_output["hidden_states"][10]
            del muq_output
            clear_device_cache(device)

            _, musicfm_hidden_states = musicfm_model.get_predictions(
                audio_seg.unsqueeze(0)
            )
            musicfm_embd_420s = musicfm_hidden_states[10]
            del musicfm_hidden_states
            clear_device_cache(device)

            # Process 30-second segments
            wraped_muq_embd_30s = []
            wraped_musicfm_embd_30s = []

            for idx_30s in range(i, i + hop_size, 30):
                start_idx_30s = idx_30s * INPUT_SAMPLING_RATE
                end_idx_30s = min(
                    (idx_30s + 30) * INPUT_SAMPLING_RATE,
                    audio.shape[-1],
                    (i + hop_size) * INPUT_SAMPLING_RATE,
                )
                if start_idx_30s >= audio.shape[-1]:
                    break
                if end_idx_30s - start_idx_30s <= 1024:
                    continue

                wraped_muq_embd_30s.append(
                    muq_model(
                        audio[start_idx_30s:end_idx_30s].unsqueeze(0),
                        output_hidden_states=True,
                    )["hidden_states"][10]
                )
                clear_device_cache(device)

                wraped_musicfm_embd_30s.append(
                    musicfm_model.get_predictions(
                        audio[start_idx_30s:end_idx_30s].unsqueeze(0)
                    )[1][10]
                )
                clear_device_cache(device)

            if wraped_muq_embd_30s:
                wraped_muq_embd_30s = torch.concatenate(wraped_muq_embd_30s, dim=1)
                wraped_musicfm_embd_30s = torch.concatenate(
                    wraped_musicfm_embd_30s, dim=1
                )

                all_embds = [
                    wraped_musicfm_embd_30s,
                    wraped_muq_embd_30s,
                    musicfm_embd_420s,
                    muq_embd_420s,
                ]

                # Align embedding lengths
                if len(all_embds) > 1:
                    embd_lens = [x.shape[1] for x in all_embds]
                    min_embd_len = min(embd_lens)
                    for idx in range(len(all_embds)):
                        all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                embd = torch.concatenate(all_embds, axis=-1)

                # Inference
                dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)
                msa_info, chunk_logits = msa_model.infer(
                    input_embeddings=embd,
                    dataset_ids=dataset_ids,
                    label_id_masks=torch.Tensor(
                        dataset_id2label_mask[
                            DATASET_LABEL_TO_DATASET_ID[DATASET_LABEL]
                        ]
                    )
                    .to(device, dtype=bool)
                    .unsqueeze(0)
                    .unsqueeze(0),
                    with_logits=True,
                )

                # Accumulate logits
                start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                end_frame = start_frame + min(
                    math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                    chunk_logits["boundary_logits"][0].shape[0],
                )

                logits["function_logits"][start_frame:end_frame, :] += (
                    chunk_logits["function_logits"][0].detach().cpu().numpy()
                )
                logits["boundary_logits"][start_frame:end_frame] = (
                    chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                )
                logits_num["function_logits"][start_frame:end_frame, :] += 1
                logits_num["boundary_logits"][start_frame:end_frame] += 1
                lens += end_frame - start_frame

            i += hop_size

    # Average logits
    logits["function_logits"] /= np.maximum(logits_num["function_logits"], 1)
    logits["boundary_logits"] /= np.maximum(logits_num["boundary_logits"], 1)

    logits["function_logits"] = torch.from_numpy(
        logits["function_logits"][:lens]
    ).unsqueeze(0)
    logits["boundary_logits"] = torch.from_numpy(
        logits["boundary_logits"][:lens]
    ).unsqueeze(0)

    # Post-process
    msa_infer_output = postprocess_functional_structure(logits, hp)

    return logits, msa_infer_output


def format_as_segments(msa_output):
    """Format as list of segments"""
    segments = []
    for idx in range(len(msa_output) - 1):
        segments.append(
            {
                "start": str(round(msa_output[idx][0], 2)),
                "end": str(round(msa_output[idx + 1][0], 2)),
                "label": msa_output[idx][1],
            }
        )
    return segments


def format_as_msa(msa_output):
    """Format as MSA format"""
    lines = []
    for time, label in msa_output:
        lines.append(f"{time:.2f} {label}")
    return "\n".join(lines)


def format_as_json(segments):
    """Format as JSON"""
    return json.dumps(segments, indent=2, ensure_ascii=False)


def create_visualization(
    logits, msa_output, label_num=8, frame_rates=AFTER_DOWNSAMPLING_FRAME_RATES
):
    """Create visualization plot"""
    # Assume ID_TO_LABEL mapping exists
    try:
        from dataset.label2id import ID_TO_LABEL
    except:
        ID_TO_LABEL = {i: f"Class_{i}" for i in range(128)}

    function_vals = logits["function_logits"].squeeze().cpu().numpy()
    boundary_vals = logits["boundary_logits"].squeeze().cpu().numpy()

    top_classes = np.argsort(function_vals.mean(axis=0))[-label_num:]
    T = function_vals.shape[0]
    time_axis = np.arange(T) / frame_rates

    fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Plot function logits
    for cls in top_classes:
        ax[1].plot(
            time_axis,
            function_vals[:, cls],
            label=f"{ID_TO_LABEL.get(cls, f'Class_{cls}')}",
        )

    ax[1].set_title("Top 8 Function Logits by Mean Activation")
    ax[1].set_xlabel("Time (seconds)")
    ax[1].set_ylabel("Logit")
    ax[1].xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax[1].xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax[1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax[1].legend()
    ax[1].grid(True)

    # Plot boundary logits
    ax[0].plot(time_axis, boundary_vals, label="Boundary Logit", color="orange")
    ax[0].set_title("Boundary Logits")
    ax[0].set_ylabel("Logit")
    ax[0].legend()
    ax[0].grid(True)

    # Add vertical lines for markers
    for t_sec, label in msa_output:
        for a in ax:
            a.axvline(x=t_sec, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        if label != "end":
            ax[1].text(
                t_sec + 0.3,
                ax[1].get_ylim()[1] * 0.85,
                label,
                rotation=90,
                fontsize=8,
                color="red",
            )

    plt.suptitle("Music Structure Analysis - Logits Overview", fontsize=16)
    plt.tight_layout()

    return fig


def rule_post_processing(msa_list):
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def analyze_one(audio_file, out_dir, stem=None):
    """Run the full per-file analysis pipeline and write export files.

    Shared by the single-file and batch handlers so the two paths cannot
    drift. Returns (segments, json_str, msa_str, fig, export_paths). The
    caller owns the returned figure (single-file displays it via gr.Plot;
    batch saves+closes it); on a write failure the figure is closed here
    before re-raising so it never leaks.
    """
    logits, msa_output = process_audio(audio_file)
    # Apply rule-based post-processing, if not needed, use in cli infer
    msa_output = rule_post_processing(msa_output)
    segments = format_as_segments(msa_output)
    msa_str = format_as_msa(msa_output)
    json_str = format_as_json(segments)
    fig = create_visualization(logits, msa_output)
    try:
        export_paths = export_utils.write_exports(
            audio_file, segments, json_str, msa_str, fig, out_dir, stem=stem
        )
    except Exception:
        plt.close(fig)
        raise
    return segments, json_str, msa_str, fig, export_paths


def process_and_analyze(audio_file):
    """Main processing function"""

    if audio_file is None:
        return None, "", "", None, None, None, None, None, None

    try:
        # Shared pipeline; exports land in a fresh per-run temp directory
        # (stale runs are swept automatically by the bootstrap).
        out_dir = export_utils.new_run_dir()
        segments, json_format, msa_format, fig, export_paths = analyze_one(
            audio_file, out_dir
        )

        # Create table data
        table_data = export_utils.segments_to_table(segments)

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

    except Exception as e:
        import traceback

        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)  # 在命令行输出完整错误
        return None, "", error_msg, None, None, None, None, None, None


def process_batch(files):
    """Analyze multiple files sequentially, yielding live status.

    The status table itself is the progress display: every file is listed
    as queued upfront, flips to processing, then to done/failed. Dropdown
    choices update as files finish so completed results can be inspected
    while the rest of the batch is still running.

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

    run_dir = export_utils.new_run_dir()
    bundle = os.path.join(run_dir, "bundle")
    os.makedirs(bundle, exist_ok=True)

    # De-duplicate stems upfront so same-named uploads don't overwrite each
    # other and the queued list shows the final names.
    used_stems = set()
    queue = []
    for audio_file in files:
        base = export_utils.stem_of(audio_file)
        stem = base
        n = 2
        while stem in used_stems:
            stem = f"{base}_{n}"
            n += 1
        used_stems.add(stem)
        queue.append((audio_file, stem))

    status_rows = [[stem, "⏳ queued", "", ""] for _, stem in queue]
    results = {}
    zipped_count = 0  # how many files the on-disk ZIP actually contains
    zip_path = os.path.join(run_dir, "songformer_batch.zip")

    def _rebuild_bundle_zip():
        """Rewrite manifests and atomically swap in an updated ZIP.

        Called after each completed file so the download button always
        serves "everything so far". os.replace is atomic, so a click can
        never observe a half-written archive. The (stem, segments) pairs
        are derived from `results` — the single source of truth.
        """
        named = [(s, r["segments"]) for s, r in results.items()]
        with open(
            os.path.join(bundle, "summary.csv"), "w", encoding="utf-8", newline=""
        ) as f:
            f.write(export_utils.segments_to_combined_csv(named))
        with open(
            os.path.join(bundle, "combined.json"), "w", encoding="utf-8"
        ) as f:
            f.write(export_utils.combined_json(named))
        part = zip_path + ".part"
        export_utils.zip_dir(bundle, part)
        os.replace(part, zip_path)

    # List every file as queued; clear any previous run's results
    yield (
        status_rows,
        gr.update(value=None, interactive=False, label="⬇️ Download all (ZIP)"),
        gr.update(choices=[], value=None),
        {},
    )

    for idx, (audio_file, stem) in enumerate(queue):
        status_rows[idx] = [stem, "🔄 processing…", "", ""]
        yield status_rows, gr.update(), gr.update(), results
        try:
            file_dir = os.path.join(bundle, stem)
            os.makedirs(file_dir, exist_ok=True)
            segments, json_str, msa_str, fig, paths = analyze_one(
                audio_file, file_dir, stem=stem
            )
            plt.close(fig)
            duration = (
                export_utils.format_time(float(segments[-1]["end"]))
                if segments
                else ""
            )
            status_rows[idx] = [stem, "✅", len(segments), duration]
            results[stem] = {
                "segments": segments,
                "json": json_str,
                "msa": msa_str,
                "png": paths["png"],
                "audio": audio_file,
            }
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
        else:
            # A ZIP rebuild failure must NOT mark the analyzed file as
            # failed: its exports exist and the next successful rebuild
            # will include it (pairs derive from `results`).
            try:
                # Keep the ZIP downloadable mid-run with everything so far
                _rebuild_bundle_zip()
                zipped_count = len(results)
            except Exception:
                import traceback

                print(f"ZIP rebuild error after {stem}:\n{traceback.format_exc()}")
        if zipped_count:
            zip_update = gr.update(
                value=zip_path,
                interactive=True,
                label=f"⬇️ Download all (ZIP) — {zipped_count}/{len(queue)} files",
            )
        else:
            zip_update = gr.update()
        # Completed files become inspectable while the batch continues
        yield status_rows, zip_update, gr.update(choices=list(results.keys())), results

    # Manifests + ZIP were rebuilt incrementally per file; just normalize
    # the button label now that the batch is complete. The button is only
    # active if at least one rebuild actually produced a ZIP on disk.
    yield (
        status_rows,
        gr.update(
            value=zip_path if zipped_count else None,
            interactive=bool(zipped_count),
            label="⬇️ Download all (ZIP)",
        ),
        gr.update(choices=list(results.keys())),
        results,
    )


def on_select_file(stem, results):
    """Render a previously-computed file's result in the batch detail viewer."""
    # A selection can race an in-flight batch iteration under rare scheduler
    # timings (choices reach the browser just before the state lands); the
    # guard degrades to an empty view, recoverable by re-selecting.
    results = results or {}
    if not stem or stem not in results:
        return None, "", "", None, None
    r = results[stem]
    return (
        export_utils.segments_to_table(r["segments"]),
        r["json"],
        r["msa"],
        r["png"],
        r.get("audio"),
    )


# Create Gradio interface
with gr.Blocks(
    title="Music Structure Analysis",
    css="""
    .logo-container {
        text-align: center;
        margin-bottom: 20px;
    }
    .links-container {
        display: flex;
        justify-content: center;
        column-gap: 10px;
        margin-bottom: 10px;
    }
    .model-title {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 30px;
    }
    """,
) as demo:
    # Top Logo
    gr.HTML("""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="https://raw.githubusercontent.com/ASLP-lab/SongFormer/refs/heads/main/figs/logo.png" style="max-width: 300px; height: auto;" />
        </div>
    """)

    # Model title
    gr.HTML("""
        <div class="model-title">
            SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision
        </div>
    """)

    # Links
    gr.HTML("""
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
    """)

    with gr.Tabs():
        with gr.Tab("Single File"):
            gr.Markdown(USAGE_NOTE)
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
                "and download all results as a single ZIP — it always "
                "contains everything analyzed so far, so you can download "
                "mid-run."
            )
            gr.Markdown(USAGE_NOTE)
            with gr.Row():
                with gr.Column(scale=3):
                    batch_files = gr.File(
                        label="Upload Audio Files",
                        file_count="multiple",
                        type="filepath",
                    )
                with gr.Column(scale=1):
                    batch_analyze_btn = gr.Button(
                        "🚀 Analyze Batch", variant="primary"
                    )
                    batch_zip_btn = gr.DownloadButton(
                        "⬇️ Download all (ZIP)", variant="primary", interactive=False
                    )
            with gr.Row():
                batch_status = gr.Dataframe(
                    headers=["File", "Status", "Segments", "Duration"],
                    label="Batch Status",
                    interactive=False,
                )
            batch_results_state = gr.State({})
            gr.Markdown("### Inspect a file")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_file_selector = gr.Dropdown(
                        label="Processed File", choices=[], interactive=True
                    )
                with gr.Column(scale=2):
                    batch_detail_audio = gr.Audio(
                        label="Listen", type="filepath", interactive=False
                    )
            with gr.Row():
                with gr.Column(scale=13):
                    batch_detail_table = gr.Dataframe(
                        headers=["Start / s (m:s.ms)", "End / s (m:s.ms)", "Label"],
                        label="Detected Music Segments",
                        interactive=False,
                    )
                with gr.Column(scale=8):
                    with gr.Row():
                        with gr.Accordion("📄 JSON Output", open=False):
                            batch_detail_json = gr.Textbox(
                                label="JSON Format",
                                lines=15,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )
                    with gr.Row():
                        with gr.Accordion("📋 MSA Text Output", open=False):
                            batch_detail_msa = gr.Textbox(
                                label="MSA Format",
                                lines=15,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                            )
            with gr.Row():
                batch_detail_plot = gr.Image(label="Activation Curves Visualization")

    gr.HTML("""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="https://raw.githubusercontent.com/ASLP-lab/SongFormer/refs/heads/main/figs/aslp.png" style="max-width: 300px; height: auto;" />
        </div>
    """)

    # Set event handlers
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
    batch_analyze_btn.click(
        fn=process_batch,
        inputs=[batch_files],
        outputs=[
            batch_status,
            batch_zip_btn,
            batch_file_selector,
            batch_results_state,
        ],
        show_progress="minimal",
    )
    batch_file_selector.change(
        fn=on_select_file,
        inputs=[batch_file_selector, batch_results_state],
        outputs=[
            batch_detail_table,
            batch_detail_json,
            batch_detail_msa,
            batch_detail_plot,
            batch_detail_audio,
        ],
    )

if __name__ == "__main__":
    # Download pretrained models if not exist
    download_all(use_mirror=False)
    # Initialize models
    print("Initializing models...")
    initialize_models(
        model_name="SongFormer",
        checkpoint="SongFormer.safetensors",
        config_path="SongFormer.yaml",
    )
    print("Models loaded successfully!")

    # Launch interface (Spaces injects its own server settings; an explicit
    # port would break the platform health check)
    demo.launch()
