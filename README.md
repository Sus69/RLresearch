# Hybrid Transformer Research

Research workspace for a hybrid Transformer architecture.

## Environment

Verified on 2026-07 (see `requirements.txt` for the exact pinned set).

| Component | Verified value |
|---|---|
| OS | Windows 11 |
| Editor | VS Code 1.147.0 |
| Python | 3.13.14 (venv in `.venv/`) |
| Deep learning | PyTorch 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19) |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM, driver 610.78, compute capability 8.6 |
| Model framework | Hugging Face Transformers 5.24.0 |
| Fine-tuning | PEFT (deliberately not installed yet) |
| Tracking | TensorBoard 2.21.0 |
| Version control | Git 2.56.0 |

`requirements.txt` pins every direct dependency to the version recorded above, so
`pip install -r requirements.txt` in a matching environment installs nothing.

Note: Python 3.13 is used rather than 3.11 because no 3.11 interpreter is available
on this machine; PyTorch ships cp313 CUDA wheels, so this is fully supported.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Do not install `torch` from PyPI: the Windows wheel there is CPU-only, which is why
`requirements.txt` pins the `+cu128` local version.

## Verify the GPU

```powershell
python test_gpu.py
```

Expected:

```text
PyTorch: 2.11.0+cu128
CUDA build: 12.8
CUDA available: True
GPU: NVIDIA GeForce RTX 2050
VRAM: 4.0 GB
GPU computation successful!
```

## First inference run (Qwen3-0.6B-Base)

```powershell
python qwen3_infer.py
```

The model is cached under `.hf-cache/` (gitignored) via `HF_HOME`.

Measured on the RTX 2050 at fp16:

| Metric | Value |
|---|---|
| Parameters | 596.0M |
| VRAM after load | 1.11 GB |
| Peak VRAM (64 new tokens) | 1.13 GB |
| Throughput | 3.9 tok/s first call, 12.7-14.1 tok/s steady |

The first prompt is much slower because it includes CUDA kernel warmup.

## Layout

```text
research/     literature review, research gap, methodology, papers
src/backbone/ model backbone
src/heads/    task heads
src/training/ training loops
src/optimization/ optimizers and schedulers
src/evaluation/   metrics and evaluation
configs/      experiment configs
datasets/     local data (not committed)
experiments/  run artifacts (not committed)
results/      result artifacts (not committed)
paper/        paper sources
```

## Notes

4 GB VRAM is the practical constraint: expect quantized loading (4-bit/8-bit) plus
LoRA for fine-tuning rather than full-parameter updates.
