# AImy/scripts/download_models.py
print(">>> download_models.py STARTED <<<", flush=True)

from huggingface_hub import snapshot_download
from pathlib import Path
import sys
import shutil
import subprocess
import gdown 

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

def download_qwen():
    print("[MODELS]: Downloading Qwen2.5-1.5B-IT...")
    dest = MODELS_DIR / "qwen2.5-1.5b-it"
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="AXERA-TECH/Qwen2.5-1.5B-Instruct-python",
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "Qwen2.5-1.5B-Instruct-GPTQ-Int8/**",
            "Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel/**",
        ],
    )
    print("[MODELS]: Qwen download complete.")

def download_audio_model_bundle():
    print("[MODELS] Downloading audio model bundle (SenseVoice + MeloTTS + Vosk)...")

    # Google Drive file ID (models.7z)
    GDRIVE_FILE_ID = "1vncy0l9agCGLPctnY_3CITS9R7rRs_XT"
    archive_path = BASE_DIR / "models.7z"

    # Skip if already present
    if (
        (MODELS_DIR / "sensevoice").exists()
        and (MODELS_DIR / "melotts").exists()
        and (MODELS_DIR / "vosk").exists()
    ):
        print("[MODELS] Audio models already present, skipping.")
        return

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    try:
        gdown.download(
            url,
            str(archive_path),
            quiet=False,
            fuzzy=True,   # important for large Drive files
        )
    except Exception as e:
        print("[ERROR] Failed to download audio model bundle.")
        print("Ensure the Google Drive file is shared as:")
        print("  Anyone with the link → Viewer")
        raise

    print("[MODELS] Extracting models.7z...")
    subprocess.run(
        ["7z", "x", str(archive_path), f"-o{BASE_DIR}"],
        check=True
    )

    archive_path.unlink(missing_ok=True)
    print("[MODELS] Audio model bundle installed successfully.")

def download_bert_base_uncased():
    print("[MODELS] Downloading BERT base uncased tokenizer...")

    # Google Drive file ID (bert-base-uncased.7z or .zip)
    GDRIVE_FILE_ID = "1zdu7vlBMglUf4Ip_6uuAz4EBCaG9PkMc"
    archive_path = BASE_DIR / "bert-base-uncased.7z"
    bert_dest = BASE_DIR / "bert-base-uncased"

    # Skip if already present
    if bert_dest.exists():
        print("[MODELS] bert-base-uncased already present, skipping.")
        return

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    try:
        gdown.download(
            url,
            str(archive_path),
            quiet=False,
            fuzzy=True,   # required for large Drive files
        )
    except Exception:
        print("[ERROR] Failed to download bert-base-uncased.")
        print("Ensure the Google Drive file is shared as:")
        print("  Anyone with the link → Viewer")
        raise

    print("[MODELS] Extracting bert-base-uncased...")
    subprocess.run(
        ["7z", "x", str(archive_path), f"-o{BASE_DIR}"],
        check=True
    )

    archive_path.unlink(missing_ok=True)
    print("[MODELS] BERT base uncased installed successfully.")

def download_yolo():
    print("[MODELS] Downloading YOLO11x...")
    dest = MODELS_DIR / "yolo11x"
    dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="AXERA-TECH/YOLO11",
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "ax650/yolo11x.axmodel",
        ],
    )
    src = dest / "ax650" / "yolo11x.axmodel"
    dst = dest / "yolo11x.axmodel"

    if not src.exists():
        raise FileNotFoundError(f"Expected file not found: {src}")
    
    if not dst.exists():
        shutil.move(src, dst)

    shutil.rmtree(dest/"ax650", ignore_errors=True)
    print("[MODELS] YOLO11x download complete)")


def main():
    download_qwen()
    download_audio_model_bundle()
    download_bert_base_uncased()
    download_yolo()    

if __name__ == "__main__":
    main()