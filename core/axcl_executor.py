import queue, threading
from typing import Callable, Any
from loguru import logger

class AxclExecutor:
    """Single-threaded worker that serializes all accelerator tasks."""
    def __init__(self, name="axcl"):
        self._q: "queue.Queue[tuple[str, Callable[[], Any], Callable[[Any], None] | None]]" = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name=f"{name}-worker")
        self._running = False

    def start(self):
        self._running = True
        self._worker.start()
        logger.info("AxclExecutor started")

    def stop(self):
        self._running = False
        self._q.put(("__STOP__", lambda: None, None))
        self._worker.join()

    def submit(self, tag: str, fn: Callable[[], Any], callback: Callable[[Any], None] | None = None):
        """Enqueue a function that will use the accelerator. Returns immediately."""
        self._q.put((tag, fn, callback))

    def _run(self):
        while self._running:
            tag, fn, cb = self._q.get()
            if tag == "__STOP__":
                break
            try:
                logger.debug(f"[AXCL] → {tag}")
                result = fn()  # Do the serialized accelerator work
                if cb:
                    cb(result)
            except Exception as e:
                logger.exception(f"[AXCL] Task {tag} failed: {e}")
