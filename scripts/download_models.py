# AImy/scripts/download_models.py
print(">>> download_models.py STARTED <<<", flush=True)

from huggingface_hub import snapshot_download
from pathlib import Path
import sys
import shutil

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

def download_sensevoice():
    print("[MODELS] Downloading SenseVoice...")
    dest = MODELS_DIR / "sensevoice"
    models_dest = dest / "models"

    dest.mkdir(parents=True, exist_ok=True)
    models_dest.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id="AXERA-TECH/SenseVoice",
        local_dir=dest,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "sensevoice_ax650/sensevoice.axmodel",
            "sensevoice_ax650/chn_jpn_yue_eng_ko_spectok.bpe.model",
            "sensevoice_ax650/am.mvn",
        ],
    )

    # ---- flatten files ----
    ax_src = dest / "sensevoice_ax650" / "sensevoice.axmodel"
    bpe_src = dest / "sensevoice_ax650" / "chn_jpn_yue_eng_ko_spectok.bpe.model"
    mvn_src = dest / "sensevoice_ax650" / "am.mvn"

    ax_dst = models_dest / "sensevoice.axmodel"
    bpe_dst = models_dest / "chn_jpn_yue_eng_ko_spectok.bpe.model"
    mvn_dst = dest / "am.mvn"

    for src in [ax_src, bpe_src, mvn_src]:
        if not src.exists():
            raise FileNotFoundError(f"Expected file not found: {src}")

    if not ax_dst.exists():
        shutil.move(ax_src, ax_dst)
    if not bpe_dst.exists():
        shutil.move(bpe_src, bpe_dst)
    if not mvn_dst.exists():
        shutil.move(mvn_src, mvn_dst)

    shutil.rmtree(dest / "sensevoice_ax650", ignore_errors=True)

    print("[MODELS] SenseVoice download complete.")

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
    download_yolo()
    download_sensevoice()
    

if __name__ == "__main__":
    main()