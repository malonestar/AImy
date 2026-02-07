import sounddevice as sd
import json
import queue
from loguru import logger
from vosk import Model, KaldiRecognizer
from config import VOSK_MODEL_PATH, VOSK_WAKEWORD

MIC_SAMPLE_RATE = 44100
MIC_CHANNELS = 1
vosk_model_cache = None

def listen_for_wake_word():
    global vosk_model_cache
    logger.info(
        "[WAKEWORD] Listening for wake word... "
        f"wakeword = {VOSK_WAKEWORD}"
        )
    if vosk_model_cache is None:
            vosk_model_cache = Model(VOSK_MODEL_PATH)
    model = vosk_model_cache
    rec = KaldiRecognizer (model, MIC_SAMPLE_RATE)
    q = queue.Queue()

    def callback(indata, frames, time, status):
          q.put(bytes(indata))

    with sd.RawInputStream(samplerate=MIC_SAMPLE_RATE, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
          while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                      result = json.loads(rec.Result())
                      text = result.get("text", "").lower()
                      logger.info(f"[WAKEWORD] wake word detected: {text}")
                      if VOSK_WAKEWORD in text:
                            return
    

