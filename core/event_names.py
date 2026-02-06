# AImy/core/event_names.py

# ---------------- Vision events ----------------
VISION_ROI_DETECT_MODE_ON = "VISION_ROI_DETECT_MODE_ON"
VISION_ROI_SAVED          = "VISION_ROI_SAVED"
VISION_ROI_CANCELED       = "VISION_ROI_CANCELED"
VISION_PERSON_PERSISTED   = "VISION_PERSON_PERSISTED"
VISION_INFER_PAUSED       = "VISION_INFER_PAUSED"
VISION_INFER_RESUMED      = "VISION_INFER_RESUMED"

#----------------- UI flow events --------------------------
STATE_CHANGED = "state_changed"

# ---------------- Conversation flow events ----------------
WAKEWORD_DETECTED = "WAKEWORD_DETECTED"

GREETING_STARTED = "GREETING_STARTED"
GREETING_DONE   = "GREETING_DONE"     # greeting audio finished (ready to listen)

CHAT_USER_MESSAGE = "chat.user.message"
CHAT_ASSISTANT_MESSAGE = "chat.assistant.message"

REQUEST_LISTEN  = "REQUEST_LISTEN"    # controller -> ASR: capture one utterance
USER_TEXT_READY = "USER_TEXT_READY"   # ASR -> controller: final transcript ready

REQUEST_LLM     = "REQUEST_LLM"       # controller -> LLM: run inference
REQUEST_SPEAK   = "REQUEST_SPEAK"     # LLM -> TTS: speak this text

SPEECH_PLAYED   = "SPEECH_PLAYED"     # Audio playback completed

ERROR           = "ERROR"
