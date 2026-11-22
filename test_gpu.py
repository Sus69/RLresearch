import torch

print("PyTorch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB",
    )

    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation successful!")
else:
    print("CUDA is not available.")
