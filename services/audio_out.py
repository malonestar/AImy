# AImy/services/audio_out.py
import subprocess
import shutil
from loguru import logger   

class AudioOut:
    def __init__(self, bus=None):
        self.bus = bus
        self._aplay = shutil.which("aplay")
        self._ffplay = shutil.which("ffplay")

    def play_wav(self, path: str):
        """Play a WAV file synchronously and emit start/end events."""
        if not path:
            logger.warning("[AUDIO] play_wav called with empty path")
            return

        # fire event to pause mic
        if self.bus:
            self.bus.publish("AUDIO_PLAY_START", {"path": path})

        try:
            logger.info(f"[AUDIO] Playing: {path}")
            if self._aplay:
                # quiet mode; blocks until finished
                subprocess.run([self._aplay, "-q", path], check=True)
            elif self._ffplay:
                subprocess.run([self._ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", path], check=True)
            else:
                logger.warning(f"[AUDIO] No player found; skipping playback for {path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"[AUDIO] Playback error: {e}")
        finally:
            # signal end of playback
            if self.bus:
                self.bus.publish("AUDIO_PLAY_END", {"path": path})
            logger.info(f"[AUDIO] Finished: {path}")
