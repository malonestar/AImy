# AImy/scripts/download_models.py

from huggingface_hub import snapshot_download
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

def download_qwen():
    dest = MODELS_DIR / "qwen2.5-1.5b-it"
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="AXERA-TECH/Qwen2.5-1.5B-Instruct-python",
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_downloads=True,
        allow_patterns=[
            "Qwen2.5-1.5B-Instruct-GPTQ-Int8/**",
            "Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel/**",
        ],
    )

def main():
    print("[MODELS] Downloading Qwen...")
    download_qwen()

    print("[MODELS] Model download complete.")

    if __name__ == "__main__":
        main()