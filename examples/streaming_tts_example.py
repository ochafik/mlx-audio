#!/usr/bin/env python3
"""
Streaming TTS Example for Qwen3-TTS

This example demonstrates how to use streaming text-to-speech with mlx-audio.
It shows three different use cases:

1. Streaming from simulated LLM output
2. Streaming from pre-written text
3. Streaming with voice cloning

The streaming approach provides:
- Low latency: Audio starts playing as soon as the first sentence is complete
- Seamless transitions: Crossfading ensures no clicks or pops between chunks
- Prosody continuity: ICL conditioning maintains consistent voice and intonation

Usage:
    python examples/streaming_tts_example.py
    python examples/streaming_tts_example.py --voice Aiden --model mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16
"""

import argparse
import time
from typing import Iterator

import mlx.core as mx
import numpy as np


def simulate_llm_stream(text: str, words_per_second: float = 15.0) -> Iterator[str]:
    """Simulate an LLM streaming tokens.

    Args:
        text: Text to stream
        words_per_second: Simulated streaming speed

    Yields:
        Words with appropriate delays
    """
    words = text.split()
    delay = 1.0 / words_per_second

    for i, word in enumerate(words):
        time.sleep(delay)
        yield word + (" " if i < len(words) - 1 else "")


def example_basic_streaming(model, voice: str, language: str):
    """Basic streaming from simulated LLM output."""
    from mlx_audio.tts.models.qwen3_tts.streaming import (
        StreamingConfig,
        StreamingContext,
    )

    print("\n" + "=" * 60)
    print("Example 1: Streaming from LLM output")
    print("=" * 60)

    # Sample text that an LLM might generate
    text = """Hello! I'm so excited to demonstrate streaming text-to-speech.
    This technology allows us to start speaking before the entire response is ready.
    Each sentence is processed as soon as it's complete.
    The audio seamlessly transitions between chunks using crossfading.
    Isn't that amazing?"""

    config = StreamingConfig(
        crossfade_ms=80.0,  # 80ms crossfade for smooth transitions
        context_codes=50,  # ~4 seconds of context
        verbose=True,
    )

    ctx = StreamingContext(
        model=model,
        voice=voice,
        language=language,
        config=config,
    )

    print(f"\nStreaming text: {text[:50]}...")
    print("\nGenerating audio chunks:")

    all_audio = []
    start_time = time.time()

    # Simulate LLM streaming
    for token in simulate_llm_stream(text, words_per_second=20.0):
        for audio_chunk in ctx.add_text(token):
            all_audio.append(audio_chunk)
            duration = audio_chunk.shape[0] / model.sample_rate
            print(f"  -> Received audio chunk: {duration:.2f}s")

    # Finalize
    for audio_chunk in ctx.finalize():
        all_audio.append(audio_chunk)
        duration = audio_chunk.shape[0] / model.sample_rate
        print(f"  -> Final chunk: {duration:.2f}s")

    total_time = time.time() - start_time

    if all_audio:
        full_audio = mx.concatenate(all_audio)
        total_duration = full_audio.shape[0] / model.sample_rate
        print(f"\nTotal audio duration: {total_duration:.2f}s")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Real-time factor: {total_duration / total_time:.2f}x")
        return full_audio

    return None


def example_text_streaming(model, voice: str, language: str):
    """Stream from pre-written text."""
    from mlx_audio.tts.models.qwen3_tts.streaming import StreamingConfig, stream_text

    print("\n" + "=" * 60)
    print("Example 2: Streaming pre-written text")
    print("=" * 60)

    text = """The quick brown fox jumps over the lazy dog.
    This sentence contains every letter of the alphabet!
    Streaming TTS makes long texts more responsive.
    You hear audio while the rest is still being generated."""

    config = StreamingConfig(verbose=True)

    print(f"\nStreaming text: {text[:50]}...")

    all_audio = []
    for audio_chunk in stream_text(
        model=model,
        text=text,
        voice=voice,
        language=language,
        config=config,
    ):
        all_audio.append(audio_chunk)

    if all_audio:
        full_audio = mx.concatenate(all_audio)
        print(f"\nTotal audio duration: {full_audio.shape[0] / model.sample_rate:.2f}s")
        return full_audio

    return None


def example_with_context_reset(model, voice: str, language: str):
    """Show context management across multiple turns."""
    from mlx_audio.tts.models.qwen3_tts.streaming import (
        StreamingConfig,
        StreamingContext,
    )

    print("\n" + "=" * 60)
    print("Example 3: Multi-turn conversation with context")
    print("=" * 60)

    config = StreamingConfig(verbose=True)

    ctx = StreamingContext(
        model=model,
        voice=voice,
        language=language,
        config=config,
    )

    turns = [
        "Hello! How can I help you today?",
        "That's a great question. Let me think about it.",
        "Here's what I found. The answer is quite interesting!",
    ]

    all_audio = []

    for i, turn in enumerate(turns):
        print(f"\n--- Turn {i + 1} ---")
        print(f"Text: {turn}")

        for audio_chunk in ctx.add_text(turn):
            all_audio.append(audio_chunk)

        for audio_chunk in ctx.finalize():
            all_audio.append(audio_chunk)

        # Note: We don't reset between turns to maintain prosody continuity
        # Call ctx.reset() if you want to start fresh

    if all_audio:
        full_audio = mx.concatenate(all_audio)
        print(f"\nTotal conversation audio: {full_audio.shape[0] / model.sample_rate:.2f}s")
        return full_audio

    return None


def save_audio(audio: mx.array, sample_rate: int, filename: str):
    """Save audio to WAV file."""
    try:
        import scipy.io.wavfile as wav

        audio_np = np.array(audio).astype(np.float32)
        # Normalize to [-1, 1]
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

        # Convert to 16-bit PCM
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wav.write(filename, sample_rate, audio_int16)
        print(f"Saved audio to: {filename}")
    except ImportError:
        print("scipy not installed, skipping audio save")


def main():
    parser = argparse.ArgumentParser(description="Streaming TTS Example")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        help="Model to use",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="Aiden",
        help="Voice/speaker to use",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language code",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save generated audio to files",
    )
    args = parser.parse_args()

    print("Loading model...")
    from mlx_audio.tts.utils import load_model

    model = load_model(args.model)
    print(f"Model loaded: {args.model}")
    print(f"Sample rate: {model.sample_rate}Hz")

    # Run examples
    audio1 = example_basic_streaming(model, args.voice, args.language)
    audio2 = example_text_streaming(model, args.voice, args.language)
    audio3 = example_with_context_reset(model, args.voice, args.language)

    # Save if requested
    if args.save:
        if audio1 is not None:
            save_audio(audio1, model.sample_rate, "streaming_example_1.wav")
        if audio2 is not None:
            save_audio(audio2, model.sample_rate, "streaming_example_2.wav")
        if audio3 is not None:
            save_audio(audio3, model.sample_rate, "streaming_example_3.wav")

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
