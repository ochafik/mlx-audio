"""Tests for LASR-CTC chunk-based streaming."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.models.lasr_ctc.config import LasrEncoderConfig, ModelConfig
from mlx_audio.stt.models.lasr_ctc.lasr import LasrForCTC, StreamingLasrSession


class TestStreamingLasrSession:
    """Test session creation and basic properties."""

    def test_session_creation(self):
        session = StreamingLasrSession()
        assert session.chunk_samples == 32000
        assert session.overlap_samples == 8000
        assert not session.finished
        assert len(session.audio_buffer) == 0
        assert len(session.prev_chunk_tokens) == 0

    def test_custom_session(self):
        session = StreamingLasrSession(
            chunk_samples=48000,
            overlap_samples=16000,
        )
        assert session.chunk_samples == 48000
        assert session.overlap_samples == 16000


class TestLasrCTCStreaming:
    """Test chunk-based streaming with a small model."""

    @pytest.fixture
    def small_model(self):
        config = ModelConfig(
            vocab_size=100,
            encoder_config=LasrEncoderConfig(
                hidden_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                intermediate_size=128,
                num_mel_bins=80,
                subsampling_conv_channels=32,
                conv_kernel_size=8,
            ),
        )
        return LasrForCTC(config)

    def test_create_session(self, small_model):
        session = small_model.create_streaming_session(
            chunk_duration_s=1.0,
            overlap_duration_s=0.25,
        )
        assert session.chunk_samples == 16000
        assert session.overlap_samples == 4000

    def test_feed_audio_buffers(self, small_model):
        """Small audio should be buffered, not processed."""
        session = small_model.create_streaming_session(chunk_duration_s=2.0)
        audio = np.random.randn(8000).astype(np.float32)  # 0.5s < 2s chunk
        results = small_model.feed_audio(session, audio)
        assert results == []
        assert len(session.audio_buffer) == 8000

    def test_feed_audio_produces_tokens(self, small_model):
        """Enough audio should produce CTC tokens."""
        session = small_model.create_streaming_session(
            chunk_duration_s=1.0,
            overlap_duration_s=0.25,
        )
        # Feed 2 seconds of audio (should produce at least 1 chunk)
        audio = np.random.randn(32000).astype(np.float32)
        results = small_model.feed_audio(session, audio)

        # Should have processed at least one chunk
        assert len(results) >= 1
        # Each result is a list of token IDs
        for r in results:
            assert isinstance(r, list)

    def test_finish_session(self, small_model):
        """finish_session should process remaining buffer."""
        session = small_model.create_streaming_session(chunk_duration_s=2.0)
        # Feed less than one chunk
        audio = np.random.randn(24000).astype(np.float32)  # 1.5s
        small_model.feed_audio(session, audio)

        # Finish should process the remaining audio
        results = small_model.finish_session(session)
        assert session.finished

    def test_feed_after_finish(self, small_model):
        """Feeding after finish should return empty."""
        session = small_model.create_streaming_session()
        session.finished = True
        results = small_model.feed_audio(session, np.zeros(1000, dtype=np.float32))
        assert results == []

    def test_pcm_bytes_input(self, small_model):
        """Should accept PCM Int16 bytes."""
        session = small_model.create_streaming_session(
            chunk_duration_s=0.5, overlap_duration_s=0.1
        )
        audio = (np.random.randn(16000) * 32767).astype(np.int16).tobytes()
        results = small_model.feed_audio(session, audio)
        # Should have buffered or processed
        assert isinstance(results, list)

    def test_mel_features(self):
        """Static mel computation should produce reasonable output."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, 16000, dtype=np.float32))
        mel = LasrForCTC._compute_mel_features(audio)

        assert mel is not None
        assert mel.shape[1] == 128  # n_mels
        assert mel.shape[0] > 0  # some frames
