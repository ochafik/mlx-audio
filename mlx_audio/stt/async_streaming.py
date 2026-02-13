"""Async wrapper for streaming STT models.

Provides async/await interface for StreamingInputModel implementations,
offloading MLX compute to a thread pool to keep the event loop responsive.
"""

import asyncio
from typing import AsyncIterator, List, Optional, Union

from mlx_audio.stt.models.base import StreamingInputModel


class AsyncStreamingSTTSession:
    """Async wrapper for streaming STT sessions.

    Wraps any model implementing StreamingInputModel protocol and provides
    async methods that offload blocking MLX compute to a thread pool.

    Example:
        async with AsyncStreamingSTTSession(model) as session:
            async for text in session.stream_audio(audio_chunks):
                print(text)
    """

    def __init__(self, model: StreamingInputModel, **kwargs):
        """Initialize async session wrapper.

        Args:
            model: Model implementing StreamingInputModel protocol.
            **kwargs: Passed to model.create_streaming_session().
        """
        self._model = model
        self._session = None
        self._kwargs = kwargs
        self._loop = None

    async def __aenter__(self) -> "AsyncStreamingSTTSession":
        """Create the streaming session on context entry."""
        self._loop = asyncio.get_event_loop()
        self._session = await self._loop.run_in_executor(
            None, self._model.create_streaming_session, **self._kwargs
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Finish the session on context exit."""
        if self._session is not None:
            await self.finish()
        return False

    async def feed_audio(self, pcm_data: Union[bytes, "np.ndarray"]) -> List[str]:
        """Feed audio chunk to the session.

        Args:
            pcm_data: Audio data - bytes (int16 PCM) or numpy array (float32).

        Returns:
            List of text deltas produced from this chunk.
        """
        if self._session is None:
            raise RuntimeError("Session not initialized - use async context manager")

        return await self._loop.run_in_executor(
            None, self._model.feed_audio, self._session, pcm_data
        )

    async def finish(self) -> str:
        """Flush remaining audio and finalize the session.

        Returns:
            Final complete transcription text.
        """
        if self._session is None:
            raise RuntimeError("Session not initialized - use async context manager")

        result = await self._loop.run_in_executor(
            None, self._model.finish_session, self._session
        )
        return result

    async def reset(self) -> None:
        """Reset session state for a new utterance."""
        if self._session is None:
            raise RuntimeError("Session not initialized - use async context manager")

        await self._loop.run_in_executor(
            None, self._model.reset_session, self._session
        )

    async def stream_audio(
        self,
        audio_chunks: AsyncIterator[Union[bytes, "np.ndarray"]],
    ) -> AsyncIterator[str]:
        """Stream audio chunks and yield text as it's produced.

        This is a convenience method that handles the feed/finish loop.

        Args:
            audio_chunks: Async iterator of audio data.

        Yields:
            Text strings as they're produced.
        """
        async for chunk in audio_chunks:
            deltas = await self.feed_audio(chunk)
            for delta in deltas:
                yield delta

        # Flush final text
        final = await self.finish()
        if final:
            yield final


async def create_async_session(
    model: StreamingInputModel,
    **kwargs,
) -> AsyncStreamingSTTSession:
    """Create an async streaming STT session.

    This is a convenience function that creates and initializes the session.

    Args:
        model: Model implementing StreamingInputModel protocol.
        **kwargs: Passed to model.create_streaming_session().

    Returns:
        Initialized AsyncStreamingSTTSession.

    Example:
        session = await create_async_session(model)
        text = await session.feed_audio(audio_chunk)
        final = await session.finish()
    """
    session = AsyncStreamingSTTSession(model, **kwargs)
    await session.__aenter__()
    return session
