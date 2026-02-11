# AImy/services/vision/roi_manager.py
import cv2

class ROIManager:
    """Handles detect mode, drawing temp box, saving/canceling, final ROI box, and overlay."""
    def __init__(self):
        self.detect_mode = False
        self.roi_defined = False
        self.roi_box = None  # (x1,y1,x2,y2)
        self._drawing = False
        self._start_pt = None
        self._cb_param = {'temp_box': None}

    def mouse_callback(self, event, x, y, flags, param):
        if not self.detect_mode:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._cb_param['temp_box'] = (self._start_pt[0], self._start_pt[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drawing = False
            end_point = (x, y)
            self.roi_box = (
                min(self._start_pt[0], end_point[0]),
                min(self._start_pt[1], end_point[1]),
                max(self._start_pt[0], end_point[0]),
                max(self._start_pt[1], end_point[1]),
            )
            self._cb_param['temp_box'] = None
            print(f"[INFO] ROI box drawn at {self.roi_box}. Press 's' to save or 'c' to cancel.")

    def attach_to_window(self, window_name: str):
        # called after first imshow created the window
        cv2.setMouseCallback(window_name, self.mouse_callback, self._cb_param)

    def enable_detect_mode(self):
        self.detect_mode = True
        self.roi_defined = False
        self.roi_box = None
        print("[INFO] Detect mode enabled. Draw ROI.")

    def save_roi(self):
        if self.roi_box:
            self.detect_mode = False
            self.roi_defined = True
            print(f"[INFO] ROI saved: {self.roi_box}")
            return True
        else:
            self.detect_mode = False
            print("[WARN] No ROI drawn. Exiting detect mode.")
            return False

    def cancel_roi(self):
        self.detect_mode = False
        self.roi_box = None
        self._cb_param['temp_box'] = None
        print("[INFO] ROI cancelled.")

    def overlay(self, frame):
        if self.roi_defined and self.roi_box:
            x1,y1,x2,y2 = self.roi_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        elif self._cb_param.get('temp_box'):
            t = self._cb_param['temp_box']
            cv2.rectangle(frame, (t[0],t[1]), (t[2],t[3]), (0,255,255), 2)
        return frame

    def hud_text(self, frame, paused: bool = False):
        """Draws the on-screen instructions depending on mode and pause state."""
        if self.detect_mode:
            text = "Draw ROI. 's' to save, 'c' to cancel."
            color = (0, 255, 255)
        elif paused:
            text = "Inference paused."
            color = (0, 165, 255)  # orange for clarity
        else:
            text = "Use button to define ROI >> "
            color = (255, 255, 255)

        cv2.putText(frame, text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

