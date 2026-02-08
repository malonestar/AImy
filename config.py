# AImy/config.py
from pathlib import Path

# Base project directory
THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR / "models"
RESOURCES_DIR = THIS_DIR / "resources"

# =====================================================
# Vision - YOLO11x
# =====================================================
YOLO11X_DIR = MODELS_DIR / "yolo11x"
YOLO11X_MODEL_PATH = YOLO11X_DIR / "yolo11x.axmodel"
YOLO11X_LABELS_PATH = YOLO11X_DIR / "coco.txt"

# Camera capture defaults
CAM_CAPTURE_WIDTH = 1280
CAM_CAPTURE_HEIGHT = 720

# =====================================================
# Wake Word Detection - Porcupine (picovoice) or Vosk
# =====================================================
'''
Porcupine requires an api key. The wake word model is run locally,
but picovoice requires authentication at boot. 
Picovoice is a better option for performance, but is a more limited 
license.  Vosk is the alternative for a purely local option.
'''
WAKEWORD_ENGINE = "vosk" # options: porupine, vosk

PORCUPINE_ACCESS_KEY = "your-picovoice-api-key"
PORCUPINE_DIR = MODELS_DIR / "porcupine"
PORCUPINE_KEYWORD_PATH = PORCUPINE_DIR / "hey-amy_en_raspberry-pi_v4_0_0.ppn"
PORCUPINE_SENSITIVITY = 0.6

VOSK_MODEL_PATH = MODELS_DIR / "Vosk"
VOSK_WAKEWORD = "hey amy"

# =====================================================
# ASR - SenseVoice
# =====================================================
SENSEVOICE_DIR = MODELS_DIR / "sensevoice"
SENSEVOICE_MODEL_PATH = SENSEVOICE_DIR / "models" / "sensevoice.axmodel"
SENSEVOICE_BPE_PATH   = SENSEVOICE_DIR / "models" / "chn_jpn_yue_eng_ko_spectok.bpe.model"

# =====================================================
# LLM - Qwen2.5-1.5B-Instruct-Int8
# =====================================================
QWEN_DIR = MODELS_DIR / "qwen2.5-1.5b-it"
QWEN_HF_PATH = QWEN_DIR / "Qwen2.5-1.5B-Instruct-GPTQ-Int8"
QWEN_AX_PATH = QWEN_DIR / "Qwen2.5-1.5B-Instruct-GPTQ-Int8_axmodel"

LLM_SYSTEM_PROMPT = (
    "You are 'ay mee', an efficient assistant running on a Raspberry Pi 5 with an Axera 8850 accelerator. "
    "Answer clearly and concisely. "
    "When doing calculations, do not share reasoning steps in the final answer."
)

# Sampling controls
LLM_TEMPERATURE = 0.6     # higher = more creative, lower = more precise
LLM_TOPP        = 0.9     # nucleus sampling
LLM_TOPK        = 1       # keep 1 unless you explicitly want diversity

# =====================================================
# TTS - MeloTTS
# =====================================================
# Greeting (wav file generated with MeloTTS, played upon detection)
GREETING_WAV = RESOURCES_DIR / "greeting.wav"

MELO_DIR = MODELS_DIR / "melotts" / "python"
MELO_MODELS_DIR = MELO_DIR / "models"

# Language selection: "EN" (default demo)
MELO_LANG = "EN"  # choices: EN, ZH_MIX_EN, JP, ZH, KR, ES, FR

# Encoder/decoder model paths (choose zh variant if ZH in language)
_lang_code = "zh" if "ZH" in MELO_LANG else MELO_LANG.lower()
MELO_ENCODER = MELO_MODELS_DIR / f"encoder-{_lang_code}.onnx"
MELO_DECODER = MELO_MODELS_DIR / f"decoder-{_lang_code}.axmodel"

# Conditioning vector file lives one level above python/
MELO_GVEC = MELO_DIR.parent / f"g-{_lang_code}.bin"

# Synth defaults
MELO_SAMPLE_RATE = 44100
MELO_DEC_LEN     = 128
MELO_SPEED       = 1.2
MELO_TMP_OUTDIR  = THIS_DIR / "output"

# Utility: ensure output directory exists
MELO_TMP_OUTDIR.mkdir(exist_ok=True)

# =====================================================
# Discord 
# =====================================================
'''
Currently the only Discord functionality is within the object detection 
in the camera feed.  If a ROI is defined, and a person is detected in 
the ROI for the defined amount of time, then a notification message
and image will be sent via the webhook.
'''
DISCORD_ENABLED = False
DISCORD_WEBHOOK_URL = "YOUR_WEBHOOK_URL_HERE"


