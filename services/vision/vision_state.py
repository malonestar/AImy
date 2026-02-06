# AImy/services/vision/vision_state.py
import threading

# modes
STREAM = "STREAM"
ROI_EDIT = "ROI_EDIT"

_lock = threading.Lock()
mode = STREAM  # default

def set_mode(new_mode: str):
    global mode
    with _lock:
        mode = new_mode

def get_mode() -> str:
    with _lock:
        return mode

def is_streaming() -> bool:
    with _lock:
        return mode == STREAM
