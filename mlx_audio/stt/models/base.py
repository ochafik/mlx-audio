from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Protocol, TypeVar, runtime_checkable


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


# Generic session type for streaming input
SessionType = TypeVar("SessionType")


@runtime_checkable
class StreamingInputModel(Protocol):
    """Protocol for models that support frame-by-frame streaming audio input.

    Models implementing this protocol can process audio incrementally as it
    arrives, producing text in real-time. This is distinct from streaming
    text output (stream=True in generate()), which all models support.

    Example usage:
        if isinstance(model, StreamingInputModel):
            session = model.create_streaming_session()
            for chunk in audio_chunks:
                text_pieces = model.feed_audio(session, chunk)
                for text in text_pieces:
                    print(text, end="", flush=True)
            final = model.finish_session(session)
            print(final)
    """

    def supports_streaming_input(self) -> bool:
        """Return True if this model supports streaming audio input."""
        ...

    def create_streaming_session(self, **kwargs):
        """Create a new streaming session.

        Returns:
            A session object (dataclass) holding the state for this session.
        """
        ...

    def feed_audio(self, session, pcm_data, **kwargs) -> List[str]:
        """Feed audio chunk to the session.

        Args:
            session: Session object from create_streaming_session()
            pcm_data: Audio data - either:
                - bytes: PCM Int16 at model's sample rate
                - np.ndarray: float32 samples at model's sample rate

        Returns:
            List of text strings produced from this chunk (may be empty).
        """
        ...

    def finish_session(self, session, **kwargs) -> str:
        """Flush any buffered audio and finalize the session.

        Args:
            session: Session object from create_streaming_session()

        Returns:
            Final text (any remaining text not yet returned).
        """
        ...

    def reset_session(self, session) -> None:
        """Reset session state for a new utterance.

        This allows reusing the session object without creating a new one.

        Args:
            session: Session object to reset
        """
        ...


class StreamingSTTSessionBase(ABC):
    """DEPRECATED: Use StreamingInputModel protocol instead.

    This abstract class was designed with a different pattern than the actual
    implementations. Kept for backwards compatibility but will be removed.
    """

    @abstractmethod
    def feed_audio(self, pcm_bytes: bytes) -> List[str]:
        """Feed PCM audio chunk, return new text pieces."""
        pass

    @abstractmethod
    def finish(self) -> str:
        """Flush and return final transcript."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset session state for new utterance."""
        pass
