# core/mic_level.py
import queue

mic_level_queue = queue.Queue(maxsize=5)

def publish_mic_level(level: float):
    try:
        mic_level_queue.put_nowait(level)
    except queue.Full:
        pass
