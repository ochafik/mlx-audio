import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput

from .config import LasrEncoderConfig, ModelConfig


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> Tuple[mx.array, mx.array]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class LasrEncoderRotaryEmbedding(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        self.base = config.rope_theta

    def __call__(self, x: mx.array, offset: int = 0) -> Tuple[mx.array, mx.array]:
        # x shape: [batch, seq_len, num_heads, head_dim] or [seq_len, head_dim] depending on usage
        # We need seq_len
        seq_len = x.shape[1]

        # Create position indices
        indices = mx.arange(offset, offset + seq_len, dtype=mx.float32)

        # Compute inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )

        # Compute angles
        # indices: [seq_len], inv_freq: [dim/2]
        # output: [seq_len, dim/2]
        args = indices[:, None] * inv_freq[None, :]

        # Repeat for cos/sin to match dim
        # [seq_len, dim]
        args = mx.concatenate([args, args], axis=-1)

        cos = mx.cos(args)
        sin = mx.sin(args)

        # Reshape to broadcast: [1, seq_len, 1, dim] to match [batch, seq_len, num_heads, head_dim]
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        return cos, sin


class LasrEncoderSubsampling(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.dense_0 = nn.Linear(config.num_mel_bins, config.hidden_size)
        self.conv_0 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.conv_1 = nn.Conv1d(
            config.hidden_size,
            config.subsampling_conv_channels,
            kernel_size=config.subsampling_conv_kernel_size,
            stride=config.subsampling_conv_stride,
        )
        self.dense_1 = nn.Linear(config.subsampling_conv_channels, config.hidden_size)
        self.act_fn = nn.ReLU()

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = self.act_fn(self.dense_0(input_features))

        hidden_states = self.act_fn(self.conv_0(hidden_states))
        hidden_states = self.act_fn(self.conv_1(hidden_states))
        return self.dense_1(hidden_states)


class LasrEncoderAttention(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads

        # Handle GQA/MQA if configured, but default config suggests standard MHA
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Ensure cos/sin broadcast correctly
            # cos shape [1, L, 1, D]
            q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        q = q.transpose(0, 2, 1, 3)  # [B, n_heads, L, D]
        k = k.transpose(0, 2, 1, 3)  # [B, n_kv_heads, L, D]
        v = v.transpose(0, 2, 1, 3)  # [B, n_kv_heads, L, D]

        if self.num_key_value_groups > 1:
            k = mx.repeat(k, self.num_key_value_groups, axis=1)
            v = mx.repeat(v, self.num_key_value_groups, axis=1)

        # Attention
        w = (q @ k.transpose(0, 1, 3, 2)) * self.scaling
        if mask is not None:
            # mask expected shape broadcastable to [B, n_heads, L, L]
            w = w + mask

        w = mx.softmax(w, axis=-1)
        o = w @ v

        o = o.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(o)


class LasrEncoderConvolutionModule(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        channels = config.hidden_size
        kernel_size = config.conv_kernel_size

        # Activation
        self.activation = (
            nn.SiLU() if config.hidden_act == "silu" else nn.ReLU()
        )  # Simplification

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.convolution_bias,
        )

        # Depthwise conv
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=config.convolution_bias,
        )
        self.kernel_size = kernel_size

        self.norm = nn.BatchNorm(
            config.hidden_size, momentum=config.batch_norm_momentum
        )

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=config.convolution_bias,
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Input (B, L, C)

        # Pointwise 1
        hidden_states = self.pointwise_conv1(hidden_states)

        # GLU: split last dim
        act_size = hidden_states.shape[-1] // 2
        hidden_states = hidden_states[..., :act_size] * mx.sigmoid(
            hidden_states[..., act_size:]
        )

        # Depthwise
        # Manual asymmetric padding for "same" convolution
        # Left: (K-1)//2, Right: K-1 - Left
        pad_left = (self.kernel_size - 1) // 2
        pad_right = self.kernel_size - 1 - pad_left

        # MLX pad expects list of (low, high) for each dim
        # Input (N, L, C). We pad dim 1.
        hidden_states = mx.pad(hidden_states, ((0, 0), (pad_left, pad_right), (0, 0)))

        hidden_states = self.depthwise_conv(hidden_states)

        hidden_states = self.norm(hidden_states)

        hidden_states = self.activation(hidden_states)
        hidden_states = self.pointwise_conv2(hidden_states)

        return hidden_states


class LasrEncoderFeedForward(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.attention_bias
        )
        self.activation = nn.SiLU() if config.hidden_act == "silu" else nn.ReLU()
        self.linear2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.attention_bias
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.activation(self.linear1(hidden_states))
        hidden_states = self.linear2(hidden_states)
        return hidden_states


class LasrEncoderBlock(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.feed_forward1 = LasrEncoderFeedForward(config)
        self.self_attn = LasrEncoderAttention(config)
        self.conv = LasrEncoderConvolutionModule(config)
        self.feed_forward2 = LasrEncoderFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.norm_self_att = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_conv = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_feed_forward2 = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.norm_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.feed_forward_residual_weights = config.feed_forward_residual_weights
        self.conv_residual_weights = config.conv_residual_weights

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # FF1
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual
            + self.feed_forward_residual_weights[1] * hidden_states
        )

        # Self Attn
        normalized_hidden_states = self.norm_self_att(hidden_states)
        attn_output = self.self_attn(
            normalized_hidden_states, position_embeddings=position_embeddings, mask=mask
        )
        hidden_states = hidden_states + attn_output

        # Conv
        conv_output = self.conv(self.norm_conv(hidden_states))
        hidden_states = (
            self.conv_residual_weights[0] * hidden_states
            + self.conv_residual_weights[1] * conv_output
        )

        # FF2
        residual = hidden_states
        hidden_states = self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = (
            self.feed_forward_residual_weights[0] * residual
            + self.feed_forward_residual_weights[1] * hidden_states
        )

        return self.norm_out(hidden_states)


class LasrEncoder(nn.Module):
    def __init__(self, config: LasrEncoderConfig):
        super().__init__()
        self.config = config
        self.subsampler = LasrEncoderSubsampling(config)
        self.rotary_emb = LasrEncoderRotaryEmbedding(config)
        self.layers = [
            LasrEncoderBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.out_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, input_features: mx.array, mask: Optional[mx.array] = None
    ) -> mx.array:
        hidden_states = self.subsampler(input_features)

        # Positional Embeddings
        cos, sin = self.rotary_emb(hidden_states)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, position_embeddings=(cos, sin), mask=mask
            )

        return self.out_norm(hidden_states)


class LasrForCTC(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = LasrEncoder(config.encoder_config)
        self.ctc_head = nn.Linear(config.encoder_config.hidden_size, config.vocab_size)

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = self.encoder(input_features)
        logits = self.ctc_head(hidden_states)
        return logits

    def decode(self, input_features: mx.array) -> STTOutput:
        logits = self(input_features)
        logprobs = nn.log_softmax(logits, axis=-1)

        # Greedy decode
        tokens = mx.argmax(logprobs, axis=-1)

        # Decode tokens to text (Requires tokenizer)
        return STTOutput(text="", tokens=tokens)

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """
        Sanitize weights from PyTorch/HF format to MLX format.
        """
        new_weights = {}
        for k, v in weights.items():
            if "rotary_emb.inv_freq" in k:
                continue

            # Handle Conv1d weights: (out, in, kernel) -> (out, kernel, in)
            if "conv" in k and "weight" in k and v.ndim == 3:
                v = mx.transpose(v, (0, 2, 1))

            # Handle CTC head (Conv1d 1x1 in HF -> Linear in MLX)
            if "ctc_head.weight" in k and v.ndim == 3:
                v = mx.squeeze(v, axis=-1)

            new_weights[k] = v

        return new_weights

    # --- Chunk-based streaming API ---
    #
    # LASR-CTC uses bidirectional attention and symmetric convolutions,
    # so true frame-by-frame streaming is not possible. Instead, we
    # process audio in overlapping chunks, CTC-decode each chunk, and
    # stitch the results together.

    def supports_streaming_input(self) -> bool:
        return True

    def create_streaming_session(
        self,
        chunk_duration_s: float = 2.0,
        overlap_duration_s: float = 0.5,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400,
    ) -> "StreamingLasrSession":
        """Create a chunk-based streaming session.

        Args:
            chunk_duration_s: Duration of each processing chunk in seconds
            overlap_duration_s: Overlap between consecutive chunks in seconds
            sample_rate: Audio sample rate (for mel computation)
            hop_length: STFT hop length
            n_fft: STFT window size
        """
        return StreamingLasrSession(
            chunk_samples=int(chunk_duration_s * sample_rate),
            overlap_samples=int(overlap_duration_s * sample_rate),
            sample_rate=sample_rate,
            hop_length=hop_length,
            n_fft=n_fft,
        )

    def feed_audio(
        self,
        session: "StreamingLasrSession",
        pcm_16k: Union[bytes, np.ndarray],
    ) -> list:
        """Feed audio chunk and return new CTC-decoded token sequences.

        Buffers audio until a full chunk is available, then processes it
        through the encoder and CTC-decodes.

        Args:
            session: Active streaming session
            pcm_16k: PCM audio at 16kHz (bytes=Int16, ndarray=float32)

        Returns:
            List of new token ID sequences (one per completed chunk).
            Each is a list of int token IDs after CTC dedup + blank removal.
        """
        if session.finished:
            return []

        if isinstance(pcm_16k, bytes):
            samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.asarray(pcm_16k, dtype=np.float32)

        if len(samples) == 0:
            return []

        session.audio_buffer = np.concatenate([session.audio_buffer, samples])
        results = []

        stride = session.chunk_samples - session.overlap_samples
        if stride <= 0:
            stride = session.chunk_samples  # Fallback: no overlap

        while len(session.audio_buffer) >= session.chunk_samples:
            chunk = session.audio_buffer[:session.chunk_samples]

            tokens = self._process_chunk(chunk, session)
            if tokens:
                results.append(tokens)

            # Advance by stride (keep overlap)
            session.audio_buffer = session.audio_buffer[stride:]

        return results

    def finish_session(self, session: "StreamingLasrSession") -> list:
        """Process any remaining audio in the buffer.

        Returns:
            List of token ID sequences from the final chunk (may be empty).
        """
        if session.finished:
            return []

        session.finished = True
        results = []

        # Process remaining audio if there's enough for mel computation
        if len(session.audio_buffer) > session.n_fft:
            tokens = self._process_chunk(session.audio_buffer, session)
            if tokens:
                results.append(tokens)

        return results

    def _process_chunk(
        self, audio_chunk: np.ndarray, session: "StreamingLasrSession"
    ) -> list:
        """Process a single audio chunk through mel → encoder → CTC decode.

        Returns list of non-blank, deduplicated token IDs.
        """
        # Compute mel spectrogram for this chunk
        mel = self._compute_mel_features(
            audio_chunk, session.sample_rate, session.hop_length, session.n_fft,
            n_mels=self.config.encoder_config.num_mel_bins,
        )
        if mel is None or mel.shape[1] == 0:
            return []

        # mel: [1, frames, mel_bins]
        mel_mx = mx.array(mel[None, :, :], dtype=mx.float32)

        # Forward through encoder + CTC head
        logits = self(mel_mx)  # [1, frames', vocab_size]
        mx.eval(logits)

        # Greedy CTC decode
        tokens = mx.argmax(logits[0], axis=-1)  # [frames']
        tokens = np.array(tokens).tolist()

        # CTC collapse: remove consecutive duplicates, then remove blanks
        deduped = []
        prev = None
        for t in tokens:
            if t != prev:
                deduped.append(t)
            prev = t

        # Remove blank token (typically 0)
        blank_id = self.config.pad_token_id
        deduped = [t for t in deduped if t != blank_id]

        # Handle overlap deduplication with previous chunk
        if session.prev_chunk_tokens and deduped:
            # Remove prefix tokens that match the tail of the previous chunk
            # (these come from the overlap region)
            overlap_frames = session.overlap_samples // session.hop_length // 4  # ~4x subsample
            tail = session.prev_chunk_tokens[-overlap_frames:] if overlap_frames > 0 else []
            if tail:
                # Find and remove overlapping prefix
                max_match = min(len(tail), len(deduped))
                match_len = 0
                for i in range(1, max_match + 1):
                    if tail[-i:] == deduped[:i]:
                        match_len = i
                if match_len > 0:
                    deduped = deduped[match_len:]

        session.prev_chunk_tokens = deduped if deduped else session.prev_chunk_tokens

        return deduped

    @staticmethod
    def _compute_mel_features(
        audio: np.ndarray,
        sample_rate: int = 16000,
        hop_length: int = 160,
        n_fft: int = 400,
        n_mels: int = 128,
    ) -> Optional[np.ndarray]:
        """Compute log-mel spectrogram features for a chunk.

        Returns mel features as [frames, n_mels] numpy array, or None.
        """
        if len(audio) < n_fft:
            return None

        # Simple mel computation using numpy
        # Window
        window = np.hanning(n_fft + 1)[:-1].astype(np.float32)

        # Pad for STFT
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        if n_frames <= 0:
            return None

        # Frame extraction
        indices = np.arange(n_fft)[None, :] + (np.arange(n_frames) * hop_length)[:, None]
        if indices.max() >= len(audio):
            n_frames = (len(audio) - n_fft) // hop_length + 1
            indices = np.arange(n_fft)[None, :] + (np.arange(n_frames) * hop_length)[:, None]

        frames = audio[indices] * window[None, :]

        # FFT
        spectrum = np.fft.rfft(frames, n=n_fft, axis=-1)
        magnitudes = np.abs(spectrum) ** 2

        # Mel filter bank (simple triangular)
        n_freq = 1 + n_fft // 2
        mel_low = 0.0
        mel_high = 2595.0 * np.log10(1.0 + (sample_rate / 2.0) / 700.0)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

        fb = np.zeros((n_freq, n_mels), dtype=np.float32)
        for m in range(n_mels):
            f_left = bins[m]
            f_center = bins[m + 1]
            f_right = bins[m + 2]
            for k in range(f_left, f_center):
                if f_center > f_left:
                    fb[k, m] = (k - f_left) / (f_center - f_left)
            for k in range(f_center, f_right):
                if f_right > f_center:
                    fb[k, m] = (f_right - k) / (f_right - f_center)

        mel_spec = magnitudes @ fb  # [frames, n_mels]

        # Log scale
        mel_spec = np.log(np.maximum(mel_spec, 1e-10))

        return mel_spec


@dataclass
class StreamingLasrSession:
    """Session for chunk-based streaming CTC decoding.

    LASR-CTC uses bidirectional attention, so true frame-by-frame
    streaming is not possible. Instead, audio is processed in overlapping
    chunks. Each chunk is independently encoded and CTC-decoded, with
    overlap deduplication to stitch results.
    """

    chunk_samples: int = 32000  # 2s at 16kHz
    overlap_samples: int = 8000  # 0.5s overlap
    sample_rate: int = 16000
    hop_length: int = 160
    n_fft: int = 400
    audio_buffer: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    prev_chunk_tokens: list = field(default_factory=list)
    finished: bool = False
