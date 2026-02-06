# AImy/core/states.py

from enum import Enum, auto

class AssistantState(Enum):
    ASLEEP = auto()
    IDLE = auto()
    LOOKING = auto()
    GREETING = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()
