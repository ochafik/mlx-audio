#!/usr/bin/env python
"""E2E test for streaming STT models."""

import os
import sys
import time
import traceback

# Test audio file
TEST_AUDIO = "examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav"

# Models to test
MODELS = [
    {
        "name": "Moshi STT",
        "model_path": "kyutai/stt-1b-en_fr-mlx",
        "model_type": "moshi",
        "sample_rate": 24000,
    },
    {
        "name": "Voxtral Realtime",
        "model_path": "shreyask/voxtral-mini-4b-realtime-mlx-fp16",
        "model_type": "voxtral_realtime",
        "sample_rate": 16000,
    },
    # VibeVoice and LASR don't have public HF repos yet - tested via unit tests
    # {
    #     "name": "VibeVoice ASR",
    #     "model_path": "...",
    #     "model_type": "vibevoice_asr",
    #     "sample_rate": 24000,
    # },
    # {
    #     "name": "LASR-CTC",
    #     "model_path": "...",
    #     "model_type": "lasr_ctc",
    #     "sample_rate": 16000,
    # },
]


def load_audio(path, sample_rate):
    """Load and resample audio."""
    import numpy as np
    try:
        import sphn
        audio, sr = sphn.read(path, sample_rate=sample_rate)
        return audio[0].astype(np.float32)
    except ImportError:
        # Fallback to miniaudio
        import miniaudio
        decoded = miniaudio.decode_file(path, sample_rate=sample_rate)
        audio = np.frombuffer(decoded.samples, dtype=np.float32)
        return audio


def test_model(model_info):
    """Test a single model."""
    import numpy as np

    name = model_info["name"]
    model_path = model_info["model_path"]
    sample_rate = model_info["sample_rate"]

    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Path: {model_path}")
    print(f"Sample rate: {sample_rate}")
    print(f"{'='*60}")

    try:
        # Import with model-specific path to avoid numba
        import importlib
        model_module = importlib.import_module(
            f"mlx_audio.stt.models.{model_info['model_type']}"
        )

        print("✓ Module imported")

        # Load model
        from mlx_audio.stt.utils import load_model
        print("Loading model (this may take a while)...")
        start = time.time()
        model = load_model(model_path)
        load_time = time.time() - start
        print(f"✓ Model loaded in {load_time:.1f}s")

        # Check streaming support
        supports_streaming = hasattr(model, 'supports_streaming_input') and model.supports_streaming_input()
        print(f"✓ Streaming input support: {supports_streaming}")

        # Load test audio
        print(f"Loading audio: {TEST_AUDIO}")
        audio = load_audio(TEST_AUDIO, sample_rate)
        print(f"✓ Audio loaded: {len(audio)/sample_rate:.2f}s, {len(audio)} samples")

        # Test batch transcription
        print("\n--- Batch Transcription ---")
        start = time.time()
        result = model.generate(audio)
        batch_time = time.time() - start
        print(f"Text: {result.text[:200]}..." if len(result.text) > 200 else f"Text: {result.text}")
        print(f"Time: {batch_time:.2f}s")
        print(f"RTF: {batch_time / (len(audio)/sample_rate):.3f}x")

        # Test streaming if supported
        if supports_streaming:
            print("\n--- Streaming Transcription ---")
            start = time.time()
            session = model.create_streaming_session()
            print("✓ Session created")

            # Feed audio in chunks (100ms chunks)
            chunk_size = int(sample_rate * 0.1)  # 100ms
            all_deltas = []
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                # Convert to int16 bytes
                int16_chunk = (chunk * 32768).astype(np.int16)
                deltas = model.feed_audio(session, int16_chunk.tobytes())
                all_deltas.extend(deltas)

            final = model.finish_session(session)
            stream_time = time.time() - start

            print(f"Deltas received: {len(all_deltas)}")
            print(f"Final text: {final[:200]}..." if len(final) > 200 else f"Final text: {final}")
            print(f"Time: {stream_time:.2f}s")
            print(f"RTF: {stream_time / (len(audio)/sample_rate):.3f}x")

            # Test reset
            model.reset_session(session)
            print("✓ Session reset works")

        print(f"\n✓ {name} PASSED")
        return True

    except Exception as e:
        print(f"\n✗ {name} FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    print("Streaming STT E2E Test")
    print("=" * 60)

    test_all = "--all" in sys.argv or os.environ.get("TEST_ALL") == "1"

    results = {}
    for model_info in MODELS:
        if test_all or model_info["model_type"] == "moshi":
            results[model_info["name"]] = test_model(model_info)
        else:
            print(f"\nSkipping {model_info['name']} (use --all to test all)")
            results[model_info["name"]] = None

    print("\n" + "=" * 60)
    print("RESULTS:")
    for name, passed in results.items():
        if passed is None:
            status = "SKIPPED"
        elif passed:
            status = "PASSED"
        else:
            status = "FAILED"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    if "--all" in sys.argv or "TEST_ALL" in os.environ:
        # Test all models
        pass
    main()
