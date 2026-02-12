# AImy/services/wakeword_vosk_service.py
import threading
import time
import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from loguru import logger

from core.event_names import WAKEWORD_DETECTED
from core.states import AssistantState
from config import VOSK_MODEL_PATH, VOSK_WAKEWORD

MIC_SAMPLE_RATE = 44100
MIC_CHANNELS = 1

_vosk_model_cache = None

class WakeWordService:
    def __init__(self, bus, controller):
        self.bus = bus
        self.controller = controller

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[WAKEWORD] Vosk started")

    def stop(self):
        self._running = False

    def _load_model(self):
        global _vosk_model_cache
        if _vosk_model_cache is None:
            logger.info(f"[WAKEWORD] Loading Vosk model from {VOSK_MODEL_PATH}")
            _vosk_model_cache = Model(str(VOSK_MODEL_PATH))
        return _vosk_model_cache

    def _run(self):
        model = self._load_model()
        recognizer = KaldiRecognizer(model, MIC_SAMPLE_RATE)

        audio_q = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"[WAKEWORD] Audio status: {status}")
            audio_q.put(bytes(indata))

        try:
            with sd.RawInputStream(
                samplerate=MIC_SAMPLE_RATE,
                blocksize=8000,
                dtype="int16",
                channels=MIC_CHANNELS,
                callback=callback,
            ):
                while self._running:
                    # Only listen while LOOKING
                    if self.controller.get_state() != AssistantState.LOOKING:
                        time.sleep(0.05)
                        continue

                    data = audio_q.get()

                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "").lower()

                        if not text:
                            continue

                        logger.debug(f"[WAKEWORD] Heard: {text}")

                        if VOSK_WAKEWORD in text:
                            logger.info(f"[WAKEWORD] Detected: {VOSK_WAKEWORD}")
                            self._running = False # stop mic upon wakeword detected
                            self.bus.publish(WAKEWORD_DETECTED, None)
                            break # exit loop to close mic stream

        except Exception as e:
            logger.exception(f"[WAKEWORD] Vosk error: {e}")
