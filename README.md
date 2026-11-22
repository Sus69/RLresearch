# Hybrid Transformer Research

Research workspace for a hybrid Transformer architecture.

## Environment

| Component | Choice |
|---|---|
| OS | Windows 11 |
| Editor | VS Code |
| Python | 3.13 (venv in `.venv/`) |
| Deep learning | PyTorch 2.11.0+cu128 |
| GPU | NVIDIA GeForce RTX 2050, 4 GB VRAM, driver 610.78 |
| Model framework | Hugging Face Transformers |
| Fine-tuning | PEFT (added at the fine-tuning stage) |
| Tracking | TensorBoard |
| Version control | Git |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

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
