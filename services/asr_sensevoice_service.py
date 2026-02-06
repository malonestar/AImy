# AImy/services/asr_sensevoice_service.py
import queue, time
import numpy as np
import threading
import sounddevice as sd
from loguru import logger
from core.event_names import REQUEST_LISTEN, USER_TEXT_READY, CHAT_USER_MESSAGE
from ui.mic_level import publish_mic_level


import config  

SAMPLE_RATE   = 16000
BLOCK_MS      = 30
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

# basic VAD thresholds
RMS_START     = 0.015   # start talking
RMS_END       = 0.010   # stay talking
MIN_SPEECH_MS = 250
END_SIL_MS    = 450
MAX_UTTER_MS  = 10000  # 10s

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x, dtype=np.float32)) + 1e-12)


class SenseVoiceMicListener:
    """
    One-shot ASR:
      - waits for speech
      - captures one utterance
      - publishes USER_TEXT_READY
      - exits cleanly
    """

    def __init__(self, bus, executor, asr_adapter):
        self.bus = bus
        self.executor = executor
        self.asr = asr_adapter

        self.q = queue.Queue()
        self._reset_state()
        self._stopped = False

    # ---------- state ----------
    def _reset_state(self):
        self.speaking = False
        self.speech_buf = []
        self.above_ms = 0
        self.silence_ms = 0
        self.utter_ms = 0

    # ---------- audio callback ----------
    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"[ASR] callback status: {status}")
        mono = indata.mean(axis=1).astype(np.float32, copy=False)
        try:
            self.q.put_nowait(mono)
        except queue.Full:
            logger.warning("[ASR] input queue full, dropping block")

    # ---------- mic open ----------
    def _open_mic(self):
        return sd.InputStream(
            channels=1,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SAMPLES,
            dtype="float32",
            callback=self._audio_cb,
        )

    # ---------- VAD ----------
    def _is_speech(self, energy: float) -> bool:
        threshold = RMS_END if self.speaking else RMS_START
        return energy > threshold

    def _process_block(self, block: np.ndarray):
        self.speech_buf.append(block)

        energy = rms(block)

        # --- mic level scaling for UI ---
        # Map ~0.01–0.06 RMS → 0.0–1.0
        ui_level = min(1.0, max(0.0, (energy - 0.01) * 20))

        publish_mic_level(ui_level)

        if self._is_speech(energy):
            self.above_ms += BLOCK_MS
            self.silence_ms = 0
        else:
            self.silence_ms += BLOCK_MS
            self.above_ms = max(0, self.above_ms - BLOCK_MS // 2)

        if not self.speaking and self.above_ms >= MIN_SPEECH_MS:
            self.speaking = True
            self.utter_ms = 0
            logger.debug("[ASR] speech start")

        if self.speaking:
            self.utter_ms += BLOCK_MS
            return self._check_commit()

        return None

    def _check_commit(self):
        if self.silence_ms >= END_SIL_MS:
            return "silence"
        if self.utter_ms >= MAX_UTTER_MS:
            return "maxlen"
        return None

    # ---------- commit ----------
    def _commit_and_stop(self, reason: str):
        audio_full = np.concatenate(self.speech_buf, axis=0)
        logger.info(f"[ASR] committing {len(audio_full)} samples (reason={reason})")

        self._reset_state()

        def task():
            return self.asr.infer_audio(audio_full)

        def cb(text):
            text = (text or "").strip()
            if text:
                logger.info(f"[ASR] final text ({reason}): {text!r}")
                #self.bus.publish(USER_TEXT_READY, {"text": text})
                # 1️ Publish user message to chat
                self.bus.publish(
                    CHAT_USER_MESSAGE,
                    {"text": text}
                )

                # 2️ Continue normal pipeline
                self.bus.publish(
                    USER_TEXT_READY,
                    {
                        "text": text,
                        "source": "voice",
                    }
                )
            else:
                logger.info(f"[ASR] empty transcript ({reason}), ignoring")
            self._stopped = True

        self.executor.submit("ASR:SenseVoice:infer", task, cb)

    # ---------- fallback ----------
    def _fallback_no_mic(self, err: Exception):
        logger.error(f"[ASR] Mic failed to open: {err!r}")
        self._stopped = True

    # ---------- main loop ----------
    def listen_once(self):
        logger.info("[ASR] Listening once… (opening mic)")
        start_ts = time.time()

        try:
            stream = self._open_mic()
        except Exception as e:
            self._fallback_no_mic(e)
            return

        with stream:
            while not self._stopped:
                try:
                    block = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue

                reason = self._process_block(block)
                if reason:
                    self._commit_and_stop(reason)
                    break

                # hard timeout
                if (time.time() - start_ts) * 1000 > (MAX_UTTER_MS + 4000):
                    logger.warning("[ASR] hard timeout")
                    if self.speech_buf:
                        self._commit_and_stop("hard_timeout")
                    self._stopped = True

class ASRService:
    """
    Event-driven ASR service.
    REQUEST_LISTEN → SenseVoiceMicListener.listen_once()
    """

    def __init__(self, bus, executor, asr_adapter):
        self.bus = bus
        self.executor = executor
        self.asr = asr_adapter
        self._listening = False

        bus.subscribe(REQUEST_LISTEN, self._on_request_listen)

    def _on_request_listen(self, evt):
        if self._listening:
            logger.debug("[ASR] listen already active, ignoring REQUEST_LISTEN")
            return

        self._listening = True
        logger.info("[ASR] REQUEST_LISTEN received")

        def run():
            try:
                listener = SenseVoiceMicListener(self.bus, self.executor, self.asr)
                listener.listen_once()
            finally:
                self._listening = False
                logger.debug("[ASR] listen cycle finished")

        threading.Thread(target=run, daemon=True).start()