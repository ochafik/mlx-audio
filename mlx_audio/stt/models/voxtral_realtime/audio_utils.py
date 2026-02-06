"""Audio processing utilities for Voxtral-Realtime.

Computes log-mel spectrograms compatible with the Mistral Whisper-style encoder.
"""

import mlx.core as mx
import numpy as np
from typing import Optional

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 128
GLOBAL_LOG_MEL_MAX = 1.5


def _hanning(size: int) -> mx.array:
    window = np.hanning(size + 1)[:-1].astype(np.float32)
    return mx.array(window)


def _mel_filter_bank_slaney(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
) -> mx.array:
    """Create Slaney-style mel filter bank (matching Whisper)."""
    if fmax is None:
        fmax = sr / 2.0

    def hz_to_mel(freq):
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = 27.0 / np.log(6.4)
        freq = np.asarray(freq, dtype=np.float64)
        mels = 3.0 * freq / 200.0
        log_region = freq >= min_log_hz
        mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hz) * logstep
        return mels

    def mel_to_hz(mels):
        min_log_hz = 1000.0
        min_log_mel = 15.0
        logstep = np.log(6.4) / 27.0
        mels = np.asarray(mels, dtype=np.float64)
        freq = 200.0 * mels / 3.0
        log_region = mels >= min_log_mel
        freq[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
        return freq

    mel_min = hz_to_mel(np.array([fmin]))[0]
    mel_max = hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]

        rising = np.maximum(0, np.minimum(1, (fft_freqs - left) / max(center - left, 1e-10)))
        falling = np.maximum(0, np.minimum(1, (right - fft_freqs) / max(right - center, 1e-10)))
        filters[i] = rising * falling

        enorm = 2.0 / max(hz_points[i + 2] - hz_points[i], 1e-10)
        filters[i] *= enorm

    return mx.array(filters)


_cached_filters = None
_cached_window = None


def _get_mel_filters() -> mx.array:
    global _cached_filters
    if _cached_filters is None:
        _cached_filters = _mel_filter_bank_slaney(fmax=8000)
    return _cached_filters


def _get_window() -> mx.array:
    global _cached_window
    if _cached_window is None:
        _cached_window = _hanning(N_FFT)
    return _cached_window


def _stft(x: mx.array, window: mx.array) -> mx.array:
    """Compute STFT using MLX."""
    nperseg = N_FFT
    noverlap = nperseg - HOP_LENGTH

    # Reflection padding
    padding = nperseg // 2
    prefix = x[padding:0:-1]
    suffix = x[-2:-padding - 2:-1]
    x = mx.concatenate([prefix, x, suffix])

    n_frames = (x.size - nperseg) // HOP_LENGTH + 1
    shape = [n_frames, nperseg]
    strides = [HOP_LENGTH, 1]
    x_strided = mx.as_strided(x, shape=shape, strides=strides)
    x_windowed = x_strided * window
    return mx.fft.rfft(x_windowed)


def log_mel_spectrogram(
    audio: mx.array,
    global_max: Optional[float] = GLOBAL_LOG_MEL_MAX,
) -> mx.array:
    """Compute log-mel spectrogram.

    Args:
        audio: 1D audio waveform (16kHz)
        global_max: Global max for normalization. Uses GLOBAL_LOG_MEL_MAX by default.

    Returns:
        Log-mel spectrogram of shape [n_mels, n_frames] transposed to [n_frames, n_mels]
    """
    window = _get_window()
    filters = _get_mel_filters()

    freqs = _stft(audio, window)
    freqs = freqs[:-1, :]
    magnitudes = mx.abs(freqs) ** 2

    mel_spec = magnitudes @ filters.T
    log_spec = mx.log10(mx.maximum(mel_spec, 1e-10))

    if global_max is not None:
        log_max = mx.array(global_max)
    else:
        log_max = mx.max(log_spec)

    log_spec = mx.maximum(log_spec, log_max - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # Return as [n_frames, n_mels] for MLX Conv1d convention
    return log_spec
