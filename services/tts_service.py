# AImy/services/tts_service.py
from loguru import logger
from core.event_names import REQUEST_SPEAK, SPEECH_PLAYED
from services.speech.tts_normalization import normalize_for_tts

class TTSService:
    """
    Super simple:
      REQUEST_SPEAK -> synth (Melo) -> play_wav -> SPEECH_PLAYED -> resume vision
    """
    def __init__(self, bus, executor, tts_adapter, audio_out):
        self.bus = bus
        self.executor = executor
        self.tts = tts_adapter
        self.audio = audio_out

        bus.subscribe(REQUEST_SPEAK, self._on_request_speak)

    def _on_request_speak(self, evt):
        speech_text = normalize_for_tts(evt.payload["text"])

        def task():
            # Melo returns a wav path
            return self.tts.synth(speech_text)

        def cb(wav_path):
            self.audio.play_wav(wav_path)
            self.bus.publish(SPEECH_PLAYED, None)          

        self.executor.submit("TTS:Melo:synth", task, cb)
