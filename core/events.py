from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from threading import RLock

@dataclass
class Event:
    type: str
    payload: Dict[str, Any] | None = None

class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Event], None]]] = {}
        self._lock = RLock()

    def subscribe(self, evt_type: str, handler: Callable[[Event], None]):
        with self._lock:
            self._subs.setdefault(evt_type, []).append(handler)

    def publish(self, evt_type: str, payload: Dict[str, Any] | None = None):
        with self._lock:
            handlers = list(self._subs.get(evt_type, []))
        evt = Event(evt_type, payload)
        for h in handlers:
            h(evt)

    def unsubscribe(self, evt_type, handler):
        with self._lock:
            handlers = self._subs.get(evt_type)
            if not handlers:
                return
            if handler in handlers:
                handlers.remove(handler)
            if not handlers:
                del self._subs[evt_type]
