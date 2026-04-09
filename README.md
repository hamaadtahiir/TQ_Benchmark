# TurboQuant Benchmark

Jupyter notebook benchmarking Google's **TurboQuant** KV cache compression on Apple Silicon using MLX.

## What is TurboQuant?

TurboQuant is a training-free, data-oblivious vector quantization algorithm from [Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) that compresses the **KV cache** of LLMs to 2-4 bits per element with near-zero quality loss.

**Key distinction**: TurboQuant compresses the KV cache at runtime, NOT model weights. It is complementary to weight quantization (GGUF, GPTQ, AWQ).

- **Paper**: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
- **Algorithm**: Random rotation (PolarQuant) + optimal Lloyd-Max scalar quantization + optional QJL error correction
- **Result**: 4-6x KV cache compression, enabling much longer context windows within the same memory budget

## What This Notebook Covers

| Section | Description |
|---------|-------------|
| **TurboQuant Explained** | How the algorithm works, with visual demonstrations |
| **Model Loading** | Qwen3.5-4B on MLX (hybrid architecture: linear + full attention) |
| **Memory Benchmark** | KV cache size comparison at various sequence lengths |
| **Speed Benchmark** | Token generation throughput (FP16 vs TQ 2/3/4-bit) |
| **Quality Benchmark** | Perplexity on WikiText-2 + side-by-side generation comparison |
| **Combined Dashboard** | 3-panel results summary with conclusions |

## Setup

### Requirements

- Apple Silicon Mac (tested on M4 Max, 48 GB unified memory)
- Python 3.11+
- ~2.5 GB disk space for the Qwen3.5-4B-4bit model (downloaded on first run)

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook turboquant_benchmark.ipynb
```

Or open in VS Code / JupyterLab and run all cells top-to-bottom.

## Configurations Benchmarked

| Config | KV Cache Bits | Compression vs FP16 |
|--------|:------------:|:-------------------:|
| FP16 Baseline | 16 | 1.0x |
| TurboQuant 4-bit | 4 | ~4.0x |
| TurboQuant 3-bit | 3 | ~5.3x |
| TurboQuant 2-bit | 2 | ~8.0x |

## LM Studio Status

LM Studio does not support TurboQuant yet ([feature request #1719](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1719)). This notebook uses Python MLX (`mlx-lm` + `mlx-optiq`) as the implementation path. When LM Studio adds support, the flags will be:

```
--cache-type-k turbo3 --cache-type-v turbo3
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| [mlx-lm](https://github.com/ml-explore/mlx-lm) | Model loading & generation on Apple Silicon |
| [mlx-optiq](https://pypi.org/project/mlx-optiq/) | TurboQuant KV cache implementation for MLX |
| [datasets](https://github.com/huggingface/datasets) | WikiText-2 for perplexity evaluation |
