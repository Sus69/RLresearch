"""First local inference run with Qwen3-0.6B-Base on the RTX 2050 (4 GB VRAM)."""

import os
import time

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf-cache"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B-Base"

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a transformer, self-attention computes",
]


def vram(label):
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"{label}: allocated {used:.2f} GB, peak {peak:.2f} GB")


def main():
    print("torch        ", torch.__version__)
    print("cuda         ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu          ", torch.cuda.get_device_name(0))

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model = model.to("cuda").eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"model        {MODEL_ID}")
    print(f"parameters   {total / 1e6:.1f}M")
    print(f"dtype        {next(model.parameters()).dtype}")
    vram("after load")

    for prompt in PROMPTS:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        n_in = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                pad_token_id=tok.eos_token_id,
            )
        elapsed = time.time() - start

        n_new = out.shape[1] - n_in
        text = tok.decode(out[0], skip_special_tokens=True)

        print("=" * 70)
        print(text)
        print(f"[{n_new} new tokens in {elapsed:.2f}s -> {n_new / elapsed:.1f} tok/s]")

    vram("after generation")


if __name__ == "__main__":
    main()
