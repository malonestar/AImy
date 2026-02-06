# AImy/services/vision/viewer.py
import os, time, cv2

class Viewer:
    def __init__(self, window_name="LLM 8850 Object Detection",
                 screen_size=(1024, 600), pos=(60, 40), safe_margin=40):
        os.environ.setdefault("XDG_RUNTIME_DIR", f"/tmp/runtime-{os.geteuid()}")
        try:
            os.makedirs(os.environ["XDG_RUNTIME_DIR"], exist_ok=True)
            os.chmod(os.environ["XDG_RUNTIME_DIR"], 0o700)
        except Exception:
            pass

        self.window = window_name
        self.w, self.h = screen_size
        self.x, self.y = pos
        self.safe_margin = safe_margin
        self._placed = False
        self._maxed = False

        # Create a normal, resizable window
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)

    def _place_safe(self):
        # Don’t touch until after first imshow so the WM knows the window exists
        cv2.resizeWindow(self.window, self.w, self.h)
        cv2.moveWindow(self.window, max(self.x, self.safe_margin), max(self.y, self.safe_margin))
        for _ in range(3):
            cv2.waitKey(1)
            time.sleep(0.01)
        self._placed = True

    def toggle_maximize(self, desktop_size=(1024, 600)):
        if not self._maxed:
            W, H = desktop_size
            cv2.resizeWindow(self.window, max(320, W - 2*self.safe_margin), max(240, H - 2*self.safe_margin))
            cv2.moveWindow(self.window, self.safe_margin, self.safe_margin)
            self._maxed = True
        else:
            cv2.resizeWindow(self.window, self.w, self.h)
            cv2.moveWindow(self.window, max(self.x, self.safe_margin), max(self.y, self.safe_margin))
            self._maxed = False
        cv2.waitKey(1)

    def show(self, frame):
        cv2.imshow(self.window, frame)
        if not self._placed:
            self._place_safe()

    def wait_key(self, delay=1):
        k = cv2.waitKey(delay) & 0xFF
        if k in (ord('M'), ord('m')):
            # Adjust desktop_size to your layout if needed
            self.toggle_maximize(desktop_size=(1024, 600))
        return k

    def close(self):
        try:
            cv2.destroyWindow(self.window)
        except Exception:
            pass
        cv2.waitKey(1)
