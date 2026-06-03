---
title: SongFormer
emoji: "🎵"
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
