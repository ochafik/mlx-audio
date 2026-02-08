from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class STTOutput:
    text: str
    segments: List[dict] = None
    language: str = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    total_time: float = 0.0


class StreamingSTTSessionBase(ABC):
    """Abstract base for frame-by-frame STT sessions.

    Distinguishes between:
    - Streaming text output (stream=True in generate()) — all models support this
    - Streaming audio input (feed_audio()) — only certain models support this
    """

    @abstractmethod
    def feed_audio(self, pcm_bytes: bytes) -> List[str]:
        """Feed PCM audio chunk, return new text pieces."""

    @abstractmethod
    def finish(self) -> str:
        """Flush and return final transcript."""

    @abstractmethod
    def reset(self) -> None:
        """Reset session state for new utterance."""
