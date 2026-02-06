from loguru import logger
from core.event_names import (
    REQUEST_LLM,
    REQUEST_SPEAK,
    CHAT_ASSISTANT_MESSAGE,
)

class LLMService:
    """
    Stateless LLM worker:
    - listens for REQUEST_LLM
    - runs inference
    - emits CHAT_ASSISTANT_MESSAGE
    - optionally emits REQUEST_SPEAK (voice only)
    """

    def __init__(self, bus, executor, llm):
        self.bus = bus
        self.executor = executor
        self.llm = llm

        bus.subscribe(REQUEST_LLM, self.on_request_llm)

    def on_request_llm(self, evt):
        payload = evt.payload or {}
        user_text = payload.get("text", "").strip()
        source = payload.get("source", "voice")  # default = voice

        if not user_text:
            logger.info("[LLM] Empty REQUEST_LLM text, skipping")
            return

        def task():
            logger.debug("[LLM] Generating response")
            return self.llm.generate(user_text)

        def cb(answer):
            answer = (answer or "").strip()
            if not answer:
                return

            # 1) Always send to chat
            self.bus.publish(
                CHAT_ASSISTANT_MESSAGE,
                {"text": answer}
            )

            # 2) Speak only if voice input
            source = evt.payload.get("source", "voice")

            if source == "voice":
                logger.debug("[LLM] Voice source → requesting TTS")
                self.bus.publish(REQUEST_SPEAK, {"text": answer})
            else:
                logger.debug("[LLM] Text source → skipping TTS")


        self.executor.submit("LLM:Qwen:oneshot", task, cb)
