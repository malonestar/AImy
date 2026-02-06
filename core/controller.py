# AImy/core/controller.py
from threading import RLock
from loguru import logger
import requests

from .states import AssistantState
from .events import Event, EventBus
from .event_names import (
    VISION_PERSON_PERSISTED,
    VISION_INFER_PAUSED,
    WAKEWORD_DETECTED,
    GREETING_STARTED,
    GREETING_DONE,
    REQUEST_LISTEN,
    USER_TEXT_READY,
    REQUEST_LLM,
    REQUEST_SPEAK,
    SPEECH_PLAYED,
    ERROR,
)

class StateController:
    """
    Owns ALL state transitions.
    Other modules should publish events; controller decides state changes.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._state = AssistantState.ASLEEP
        self._lock = RLock()

        bus.subscribe(VISION_PERSON_PERSISTED, self._on_person_persisted)
        bus.subscribe(WAKEWORD_DETECTED, self._on_wakeword)
        bus.subscribe(GREETING_DONE, self._on_greeting_done)
        bus.subscribe(USER_TEXT_READY, self._on_user_text_ready)
        bus.subscribe(REQUEST_SPEAK, self._on_request_speak)
        bus.subscribe(SPEECH_PLAYED, self._on_speech_played)
        bus.subscribe(ERROR, self._on_error)

    def get_state(self) -> AssistantState:
        with self._lock:
            return self._state

    def set_state(self, new: AssistantState):
        with self._lock:
            old = self._state
            if old == new:
                return
            self._state = new

        logger.info(f"[STATE] {old.name} → {new.name}")
        # Publish internally
        self.bus.publish("state_changed", {"state": new.name})
        # 🔥 Push to UI (non-blocking)
        try:
            requests.post(
                "http://127.0.0.1:5000/push_state",
                json={"state": new.name},
                timeout=0.1,
            )
        except Exception:
            pass  # UI may not be running

    # ---------- handlers ----------

    def _on_person_persisted(self, evt: Event):
        if self.get_state() in (AssistantState.ASLEEP, AssistantState.IDLE, AssistantState.LOOKING):
            self.set_state(AssistantState.GREETING)
            self.bus.publish(GREETING_STARTED, None)
        
    def _on_wakeword(self, evt):
        if self.get_state() == AssistantState.LOOKING:
            self.bus.publish(VISION_INFER_PAUSED, None)
            self.set_state(AssistantState.GREETING)
            self.bus.publish(GREETING_STARTED, None)
    
    def _on_greeting_done(self, evt: Event):
        # Greeting finished -> listen once
        if self.get_state() == AssistantState.GREETING:
            self.set_state(AssistantState.LISTENING)
            self.bus.publish(REQUEST_LISTEN, None)

    def _on_user_text_ready(self, evt: Event):
        # Transcript arrived -> think -> request LLM
        if self.get_state() in (AssistantState.GREETING, AssistantState.LISTENING):
            text = (evt.payload or {}).get("text", "")
            text = (text or "").strip()
            if not text:
                logger.info("[CTRL] empty USER_TEXT_READY text, ignoring")
                return

            self.set_state(AssistantState.THINKING)
            #self.bus.publish(REQUEST_LLM, {"text": text})
            self.bus.publish(
                REQUEST_LLM,
                {
                    "text": text,
                    "source": (evt.payload or {}).get("source", "voice"),
                }
            )

    def _on_request_speak(self, evt: Event):
        if self.get_state() == AssistantState.THINKING:
            self.set_state(AssistantState.SPEAKING)

    def _on_speech_played(self, evt: Event):
        if self.get_state() == AssistantState.SPEAKING:
            self.set_state(AssistantState.LOOKING)

    def _on_error(self, evt: Event):
        self.set_state(AssistantState.ERROR)
