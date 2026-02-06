# AImy/services/vision/presence_trigger.py
import time

class PresenceTrigger:
    """Signals true when TARGET_LABEL center stays inside ROI for >= min_seconds."""
    def __init__(self, target_label="person", min_seconds=5, conf_thresh=0.5):
        self.target_label = target_label
        self.min_seconds = min_seconds
        self.conf_thresh = conf_thresh
        self._start = None
        self._fired = False

    def reset(self):
        self._start = None
        self._fired = False

    def check(self, detections, labels, roi_box):
        """Return (bool should_fire, float seconds_in_roi or 0.0)."""
        now = time.time()
        in_roi = False
        if roi_box:
            for det in detections:
                if labels[det.label] == self.target_label and det.prob > self.conf_thresh:
                    x, y, w, h = det.bbox
                    cx, cy = x + w/2, y + h/2
                    if roi_box[0] < cx < roi_box[2] and roi_box[1] < cy < roi_box[3]:
                        in_roi = True
                        break

        if in_roi:
            if self._start is None:
                self._start = now
            elapsed = now - self._start
            if elapsed >= self.min_seconds and not self._fired:
                self._fired = True
                return True, elapsed
            return False, elapsed
        else:
            self.reset()
            return False, 0.0
