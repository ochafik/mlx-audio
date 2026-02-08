"""Tests for VibeVoice-ASR streaming audio encoding."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.vibevoice_asr.audio_encoder import (
    Block1D,
    SConv1d,
    TokenizerEncoder,
)
from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import StreamingVibeVoiceSession


class TestSConv1dStreaming:
    """Test SConv1d streaming with conv caches."""

    def test_streaming_matches_batch_stride1(self):
        """Streaming output should match batch for stride=1 conv."""
        conv = SConv1d(in_channels=4, out_channels=8, kernel_size=7, stride=1, causal=True)

        # Batch: process full input at once
        x = mx.random.normal((1, 50, 4))
        batch_out = conv(x)

        # Streaming: process in chunks
        chunk_size = 10
        cache = None
        streaming_chunks = []
        for i in range(0, 50, chunk_size):
            chunk = x[:, i : i + chunk_size, :]
            out, cache = conv.forward_streaming(chunk, cache)
            streaming_chunks.append(out)

        streaming_out = mx.concatenate(streaming_chunks, axis=1)

        # Should match
        np.testing.assert_allclose(
            np.array(batch_out), np.array(streaming_out), atol=1e-5
        )

    def test_streaming_matches_batch_stride2(self):
        """Streaming output should match batch for strided conv."""
        conv = SConv1d(in_channels=4, out_channels=8, kernel_size=4, stride=2, causal=True)

        x = mx.random.normal((1, 40, 4))
        batch_out = conv(x)

        # Streaming in chunks of 10 (each producing 5 output frames)
        cache = None
        streaming_chunks = []
        for i in range(0, 40, 10):
            chunk = x[:, i : i + 10, :]
            out, cache = conv.forward_streaming(chunk, cache)
            streaming_chunks.append(out)

        streaming_out = mx.concatenate(streaming_chunks, axis=1)

        np.testing.assert_allclose(
            np.array(batch_out), np.array(streaming_out), atol=1e-5
        )

    def test_depthwise_streaming(self):
        """Streaming should work for depthwise (groups=channels) conv."""
        dim = 8
        conv = SConv1d(
            in_channels=dim, out_channels=dim, kernel_size=7,
            stride=1, groups=dim, causal=True,
        )

        x = mx.random.normal((1, 30, dim))
        batch_out = conv(x)

        cache = None
        streaming_chunks = []
        for i in range(0, 30, 10):
            chunk = x[:, i : i + 10, :]
            out, cache = conv.forward_streaming(chunk, cache)
            streaming_chunks.append(out)

        streaming_out = mx.concatenate(streaming_chunks, axis=1)

        np.testing.assert_allclose(
            np.array(batch_out), np.array(streaming_out), atol=1e-5
        )

    def test_non_causal_raises(self):
        """Streaming should raise for non-causal conv."""
        conv = SConv1d(in_channels=4, out_channels=4, kernel_size=3, causal=False)
        x = mx.random.normal((1, 10, 4))
        with pytest.raises(NotImplementedError):
            conv.forward_streaming(x)


class TestBlock1DStreaming:
    """Test Block1D streaming."""

    def test_streaming_produces_output(self):
        """Block1D streaming should produce same-length output."""
        dim = 16
        block = Block1D(dim=dim, kernel_size=7, causal=True)

        x = mx.random.normal((1, 20, dim))
        out, cache = block.forward_streaming(x)

        assert out.shape == x.shape
        assert cache is not None

    def test_streaming_matches_batch(self):
        """Block1D streaming should match batch processing."""
        dim = 16
        block = Block1D(dim=dim, kernel_size=7, causal=True)

        x = mx.random.normal((1, 30, dim))
        batch_out = block._forward_block(x)

        cache = None
        streaming_chunks = []
        for i in range(0, 30, 10):
            chunk = x[:, i : i + 10, :]
            out, cache = block.forward_streaming(chunk, cache)
            streaming_chunks.append(out)

        streaming_out = mx.concatenate(streaming_chunks, axis=1)

        np.testing.assert_allclose(
            np.array(batch_out), np.array(streaming_out), atol=1e-4
        )


class TestTokenizerEncoderStreaming:
    """Test TokenizerEncoder streaming with small config."""

    @pytest.fixture
    def small_encoder(self):
        """Create a small encoder for testing."""
        return TokenizerEncoder(
            channels=1,
            vae_dim=8,
            n_filters=4,
            ratios=[2, 2],
            depths=[2, 2, 2],
            causal=True,
        )

    def test_num_streaming_caches(self, small_encoder):
        """Cache count should match expected layers."""
        # stages: 3 (depths has 3 entries)
        # downsample_layers: 3 (stem + 2 downsamples)
        # blocks: 2+2+2 = 6
        # head: 1
        # total: 3 + 6 + 1 = 10
        assert small_encoder.num_streaming_caches() == 10

    def test_streaming_produces_output(self, small_encoder):
        """Streaming should produce output for sufficient input."""
        # Total downsample = 2*2 = 4, so need at least 4 samples per token
        x = mx.random.normal((1, 100, 1))
        out, caches = small_encoder.forward_streaming(x)

        assert out.shape[0] == 1
        assert out.shape[2] == 8  # vae_dim
        assert out.shape[1] > 0
        assert len(caches) == small_encoder.num_streaming_caches()

    def test_streaming_matches_batch(self, small_encoder):
        """Streaming chunks should match batch processing."""
        x = mx.random.normal((1, 200, 1))
        batch_out = small_encoder(x)

        # Stream in two halves
        caches = None
        chunks = []
        for start in [0, 100]:
            chunk = x[:, start : start + 100, :]
            out, caches = small_encoder.forward_streaming(chunk, caches)
            chunks.append(out)

        streaming_out = mx.concatenate(chunks, axis=1)

        # Should match (may differ slightly due to eval boundaries)
        np.testing.assert_allclose(
            np.array(batch_out), np.array(streaming_out), atol=1e-3
        )


class TestStreamingVibeVoiceSession:
    """Test session creation."""

    def test_session_creation(self):
        session = StreamingVibeVoiceSession(
            acoustic_caches=None,
            semantic_caches=None,
        )
        assert not session.finished
        assert len(session.acoustic_features) == 0
        assert len(session.semantic_features) == 0
        assert session.total_audio_samples == 0
