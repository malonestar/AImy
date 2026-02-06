# AImy/services/vision/camera_pipeline.py
import time, threading, queue
from picamera2 import Picamera2

class CameraPipeline:
    """Picamera2 + letterbox preprocess on a background thread.
       Emits tuples: (frame_rgb, input_img, ratio, pad_w, pad_h)."""
    def __init__(self, detector, cap_width: int, cap_height: int):
        self.detector = detector
        self.cap_width = cap_width
        self.cap_height = cap_height
        self.picam2 = None
        self.input_queue: "queue.Queue[tuple]" = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (2304, 1296)},
            lores={"size": (self.cap_width, self.cap_height), "format": "RGB888"}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)
        self.thread = threading.Thread(target=self._worker, daemon=True, name="preprocess-worker")
        self.thread.start()

    def _worker(self):
        print("[INFO] Preprocess worker started.")
        while not self.stop_event.is_set():
            frame_rgb = self.picam2.capture_array("lores")
            input_img, ratio, pad_w, pad_h = self.detector._letterbox(frame_rgb)
            try:
                self.input_queue.put((frame_rgb, input_img, ratio, pad_w, pad_h), block=False)
            except queue.Full:
                continue
        print("[INFO] Preprocess worker stopped.")

    def get(self, timeout=1):
        return self.input_queue.get(timeout=timeout)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
        if self.picam2:
            self.picam2.stop()
