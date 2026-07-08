import sys

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "torch is not installed correctly. Run 'pip install -r requirements_fixed.txt --user' and retry."
    ) from exc

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("cuda devices:")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"      capability: {torch.cuda.get_device_capability(i)}")
        print(f"      current device: {torch.cuda.current_device()}")

print("python version:", sys.version)
