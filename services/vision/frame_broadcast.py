import cv2
import threading
from services.vision import vision_state

_latest_jpeg = None
_lock = threading.Lock()

def publish_frame(frame_bgr):
    global _latest_jpeg
    ok, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return
    with _lock:
        _latest_jpeg = jpg.tobytes()

def get_jpeg_frame():
    #  if not in ROI edit mode, stop sending frames to mjpeg
    if not vision_state.is_streaming():
        return None
    with _lock:
        return _latest_jpeg
