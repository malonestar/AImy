# AImy/main.py
from loguru import logger
import threading, os, signal
import subprocess
import sys
import requests
import time
from pathlib import Path
from core.events import EventBus
from core.controller import StateController
from core.axcl_executor import AxclExecutor
from services.audio_out import AudioOut
from adapters.asr_sensevoice import SenseVoiceAdapter
from services.asr_sensevoice_service import ASRService
from adapters.llm_qwen import QwenAdapter
from services.llm_service import LLMService
from adapters.tts_melotts import MeloTTSAdapter
from services.tts_service import TTSService
from core.states import AssistantState
from adapters.axera_utils import Detector
from services.yolo11x_trigger_service import run_yolo11x_trigger_loop
from services.vision.frame_broadcast import get_jpeg_frame
from core.event_names import (
    VISION_INFER_RESUMED,
    VISION_ROI_DETECT_MODE_ON,
    SPEECH_PLAYED,
    GREETING_STARTED,
    GREETING_DONE,
    STATE_CHANGED,
    REQUEST_LLM,
    CHAT_USER_MESSAGE,
    WAKEWORD_DETECTED,
    CHAT_ASSISTANT_MESSAGE,
    ERROR
)
import config
from flask import Flask, request, jsonify, Response

#------------- UI API BACKEND ROUTING --------------------

def ui_post(path, payload):
    try:
        requests.post(f"http://127.0.0.1:5000{path}", json=payload, timeout=0.2)
    except Exception:
        pass

bus = None
api = Flask("AImyAPI")

@api.route("/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    text = data["text"]
    source = data.get("source", "text")

    bus.publish(CHAT_USER_MESSAGE, {"text": text})
    bus.publish(REQUEST_LLM, {"text": text, "source": source})

    return jsonify({"ok": True})

@api.route("/video_feed")
def video_feed():
    def gen():
        while True:
            frame = get_jpeg_frame()
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
                time.sleep(0.03)
            else:
                # paused / no frames yet
                time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@api.route("/vision/roi/edit", methods=["POST"])
def api_roi_edit():
    bus.publish(VISION_ROI_DETECT_MODE_ON, None)
    return jsonify({"ok": True})

@api.route("/wake", methods=["POST"])
def api_wake():
    bus.publish(WAKEWORD_DETECTED, {"source": "ui"})
    return jsonify({"ok": True})

@api.route("/shutdown", methods=["POST"])
def api_shutdown():
    def delayed_exit():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=delayed_exit, daemon=True).start()
    return jsonify({"ok": True})

def run_api():
    api.run(host="127.0.0.1", port=7000, debug=False)

threading.Thread(target=run_api, daemon=True).start()

# -------------------- UI FUNCTIONS ------------------

ui_process = None

def launch_ui():
    global ui_process

    ui_app = Path(__file__).parent / "ui" / "app.py"
    if not ui_app.exists():
        logger.warning("[UI] app.py not found, skipping UI launch")
        return

    ui_process = subprocess.Popen(
        [sys.executable, "-m", "ui.app", "--dev"],
        cwd=Path(__file__).parent,
        start_new_session=True
    )

    logger.info("[UI] Dashboard launched")

def wait_for_ui(timeout=10.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://127.0.0.1:5000/health", timeout=0.5)
            if r.status_code == 200:
                logger.info("[UI] UI is ready")
                return True
        except Exception:
            pass
        time.sleep(0.1)

    logger.warning("[UI] UI did not become ready in time")
    return False

#----------------- MAIN LOOP -----------------
def main():
    global bus
    logger.add("assistant.log", rotation="1 MB", level="INFO")

    bus = EventBus()
    controller = StateController(bus)
    
    #----------- LAUNCH UI ---------------
    launch_ui()
    wait_for_ui()

    logger.info("[SYSTEM] UI ready, loading models…")
    bus.subscribe(STATE_CHANGED, lambda evt: ui_post("/push_state", evt.payload))

    #----------- CHAT WINDOW --------------
    bus.subscribe(
        CHAT_USER_MESSAGE,
        lambda evt: ui_post(
            "/push_chat",
            {
                "role": "user",
                "text": evt.payload.get("text", "")
            }
        )
    )

    bus.subscribe(
        CHAT_ASSISTANT_MESSAGE,
        lambda evt: ui_post(
            "/push_chat",
            {
                "role": "assistant",
                "text": evt.payload.get("text", "")
            }
        )
    )

    #----------- LAUNCH EXECUTOR ----------
    executor = AxclExecutor()
    executor.start()

    audio = AudioOut()

    # ---------- PRELOAD MODELS ----------
    logger.info("[MODELS] Loading models...")

    llm = QwenAdapter(config.QWEN_HF_PATH, config.QWEN_AX_PATH)
    llm.init_model()
    logger.info("[LLM] Qwen initialized.")
    llm_service = LLMService(bus, executor, llm)

    asr = SenseVoiceAdapter()
    asr.init_asr()
    logger.info("[ASR] SenseVoice initialized.")
    asr_service = ASRService(bus, executor, asr)

    tts = MeloTTSAdapter()
    tts.init_tts()
    logger.info("[TTS] MeloTTS initialized.")
    tts_service = TTSService(bus, executor, tts, audio)

    # Boot state
    controller.set_state(AssistantState.LOOKING)

    if config.WAKEWORD_ENGINE == 'porcupine':
        from services.wakeword_porcupine_service import WakeWordService

        wakeword = WakeWordService(
            bus=bus,
            controller=controller,
            keyword_path=config.PORCUPINE_KEYWORD_PATH,
            sensitivity=config.PORCUPINE_SENSITIVITY,
            access_key=config.PORCUPINE_ACCESS_KEY,
        )
        logger.info(
            f"[WAKEWORD] keyword={config.PORCUPINE_KEYWORD_PATH.name} "
            f"sensitivity={config.PORCUPINE_SENSITIVITY}"
        )

    elif config.WAKEWORD_ENGINE == 'vosk':
        from services.wakeword_vosk_service import WakeWordService

        wakeword = WakeWordService(
            bus=bus,
            controller=controller,
        )
        logger.info(
            "[WAKEWORD] Vosk loaded successfully.  "
            f"Wake word: {config.VOSK_WAKEWORD}"
        )

    else:
        raise ValueError(
            f"Unknown WAKEWORD_ENGINE: {config.WAKEWORD_ENGINE}"
        )
    wakeword.start()

    # ---------- VISION LOOP ----------

    logger.info("[VISION] Initializing detector...")

    detector = Detector(
        model_path=str(config.YOLO11X_MODEL_PATH),
        labels_path=str(config.YOLO11X_LABELS_PATH)
    )

    def vision_runner():
        try:
            run_yolo11x_trigger_loop(
                detector=detector,
                cap_width=config.CAM_CAPTURE_WIDTH,
                cap_height=config.CAM_CAPTURE_HEIGHT,
                bus=bus,  
            )
        except Exception as e:
            logger.exception(f"[VISION THREAD CRASHED] {e}")

    threading.Thread(
        target=vision_runner,
        daemon=True
    ).start()

    logger.info("[VISION] Vision thread started")

    # ---------- Greeting playback (vision trigger) ----------
    def on_greeting_started(evt):
        def play_greeting_then_signal():
            greeting_path = config.GREETING_WAV
            if greeting_path.exists():
                logger.info(f"[AUDIO] Playing greeting clip: {greeting_path}")
                audio.play_wav(str(greeting_path))
            else:
                logger.warning(f"[AUDIO] Missing greeting clip: {greeting_path}")

            bus.publish(GREETING_DONE, None)

        threading.Thread(target=play_greeting_then_signal, daemon=True).start()

    bus.subscribe(GREETING_STARTED, on_greeting_started)

    # ---------- After speech finishes, resume vision inference ----------
    def on_speech_played(evt):
        bus.publish(VISION_INFER_RESUMED, None)
        if config.WAKEWORD_ENGINE == "vosk":
            logger.info("[WAKEWORD] Restarting Vosk listener")
            wakeword.start()

    bus.subscribe(SPEECH_PLAYED, on_speech_played)

    # ---------- Keep main alive ----------
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[SHUTDOWN] Keyboard interrupt received.")
    except Exception as e:
        logger.exception(f"[ERROR] main loop failed: {e}")
        bus.publish(ERROR, {"err": repr(e)})
    finally:
        try:
            tts.shutdown()
        except Exception:
            pass
        try:
            llm.shutdown()
        except Exception:
            pass
        try:
            wakeword.stop()
        except Exception:
            pass
        try:
            if ui_process:
                logger.info("[UI] Shutting down dashboard")
                ui_process.terminate()
                ui_process.wait(timeout=2)
        except Exception:
            pass
        executor.stop()
        logger.info("[EXIT] AImy module shut down cleanly.")

if __name__ == "__main__":
    main()
