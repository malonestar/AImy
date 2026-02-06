# AImy/services/wakeword_porcupine_service.py
import threading
import pvporcupine
import pyaudio
import struct
import time
from loguru import logger
from core.event_names import WAKEWORD_DETECTED
from core.states import AssistantState

class WakeWordService:
    def __init__(self, bus, controller, keyword_path, sensitivity=0.6, access_key=None):
        self.bus = bus
        self.controller = controller
        self.keyword_path = keyword_path
        self.sensitivity = sensitivity
        self.access_key = access_key

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[WAKEWORD] Porcupine started")

    def stop(self):
        self._running = False

    def _run(self):
        porcupine = pvporcupine.create(
            access_key=self.access_key,
            keyword_paths=[str(self.keyword_path)],
            sensitivities=[self.sensitivity],
        )

        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )

        try:
            while self._running:
                # Only listen while LOOKING
                if self.controller.get_state() != AssistantState.LOOKING:
                    time.sleep(0.05)
                    continue

                pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from(
                    "h" * porcupine.frame_length, pcm
                )

                result = porcupine.process(pcm)
                if result >= 0:
                    logger.info("[WAKEWORD] Detected: hey amy")
                    self.bus.publish(WAKEWORD_DETECTED, None)
        finally:
            stream.close()
            pa.terminate()
            porcupine.delete()
