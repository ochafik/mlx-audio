"""Causal audio encoder for Voxtral Realtime.

32-layer causal transformer with:
- Causal conv1d stem (128 -> 1280, stride 1; 1280 -> 1280, stride 2)
- Interleaved RoPE (theta=1M)
- Sliding window attention (750)
- SwiGLU FFN
- Selective biases (wq/wv/wo yes, wk no; w2 only in FFN)
- 4x downsample + adapter MLP

Optimizations:
- RoPE frequencies computed once and shared across all 32 layers
- Attention mask computed once and shared across all 32 layers
- Interleave via stack+reshape instead of indexed assignment
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .config import EncoderConfig


def _interleaved_rope(x, cos, sin, n_heads, head_dim):
    """Apply interleaved (GPT-J style) RoPE.

    Rotates consecutive pairs: (x[0], x[1]), (x[2], x[3]), ...
    x: [seq, n_heads * head_dim]
    cos, sin: [seq, head_dim // 2]
    """
    seq_len = x.shape[0]
    x = x.reshape(seq_len, n_heads, head_dim)
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
    cos = cos[:, None, :]  # [seq, 1, hd/2]
    sin = sin[:, None, :]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    # Interleave back via stack (avoids indexed assignment on zeros)
    out = mx.stack([o1, o2], axis=-1).reshape(seq_len, n_heads, head_dim)
    return out.reshape(seq_len, n_heads * head_dim)


def _compute_rope_freqs(positions, head_dim, theta):
    """Compute cos/sin frequencies for RoPE.

    positions: [seq_len] int array
    Returns: (cos, sin) each [seq_len, head_dim // 2]
    """
    half_dim = head_dim // 2
    freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    angles = positions[:, None].astype(mx.float32) * freqs[None, :]
    return mx.cos(angles), mx.sin(angles)


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left-only padding."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - stride  # left-only padding
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True
        )

    def __call__(self, x):
        # x: [batch, seq, channels] (MLX conv1d expects NLC)
        # Left-pad only
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        return self.conv(x)


class EncoderAttention(nn.Module):
    """Multi-head attention for encoder with selective biases."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.rope_theta = config.rope_theta
        attn_dim = config.n_heads * config.head_dim

        # Selective biases: wq, wv, wo have bias; wk does NOT
        self.wq = nn.Linear(config.dim, attn_dim, bias=True)
        self.wk = nn.Linear(config.dim, attn_dim, bias=False)
        self.wv = nn.Linear(config.dim, attn_dim, bias=True)
        self.wo = nn.Linear(attn_dim, config.dim, bias=True)

    def __call__(self, x, rope_cos, rope_sin, mask):
        """
        Args:
            x: [seq, dim]
            rope_cos, rope_sin: precomputed [seq, head_dim // 2]
            mask: precomputed [1, 1, seq, seq] additive mask
        """
        seq_len = x.shape[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # RoPE (using pre-computed frequencies)
        q = _interleaved_rope(q, rope_cos, rope_sin, self.n_heads, self.head_dim)
        k = _interleaved_rope(k, rope_cos, rope_sin, self.n_heads, self.head_dim)

        # Reshape for attention: [1, n_heads, seq, head_dim]
        q = q.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(1, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )

        # Reshape back: [1, n_heads, seq, head_dim] -> [seq, n_heads * head_dim]
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, self.n_heads * self.head_dim)
        return self.wo(attn_out)


class EncoderLayer(nn.Module):
    """Single encoder transformer layer."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)
        self.attention = EncoderAttention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # SwiGLU FFN: w1=gate (no bias), w3=up (no bias), w2=down (bias)
        self.feed_forward_w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.feed_forward_w2 = nn.Linear(config.hidden_dim, config.dim, bias=True)

    def __call__(self, x, rope_cos, rope_sin, mask):
        # Attention
        h = self.attention_norm(x)
        h = self.attention(h, rope_cos, rope_sin, mask)
        x = x + h

        # SwiGLU FFN
        h = self.ffn_norm(x)
        gate = nn.silu(self.feed_forward_w1(h))
        up = self.feed_forward_w3(h)
        x = x + self.feed_forward_w2(gate * up)

        return x


class AudioEncoder(nn.Module):
    """Full causal audio encoder: conv stem + transformer + downsample + adapter."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config

        # Conv stem
        self.conv_layers_0_conv = CausalConv1d(128, config.dim, kernel_size=3, stride=1)
        self.conv_layers_1_conv = CausalConv1d(config.dim, config.dim, kernel_size=3, stride=2)

        # Transformer layers
        self.transformer_layers = [EncoderLayer(config) for _ in range(config.n_layers)]
        self.transformer_norm = nn.RMSNorm(config.dim, eps=config.norm_eps)

        # Adapter MLP
        adapter_input_dim = config.dim * config.downsample_factor  # 5120
        decoder_dim = 3072
        self.audio_language_projection_0 = nn.Linear(
            adapter_input_dim, decoder_dim, bias=False
        )
        self.audio_language_projection_2 = nn.Linear(
            decoder_dim, decoder_dim, bias=False
        )

    def __call__(self, mel):
        """
        Args:
            mel: [mel_bins, frames] log-mel spectrogram

        Returns:
            mx.array: [seq/4, decoder_dim] adapter output
        """
        # mel is [128, frames], transpose to [frames, 128] for conv (NLC format)
        x = mel.T  # [frames, 128]
        x = x[None, :, :]  # [1, frames, 128]

        # Conv stem
        x = nn.gelu(self.conv_layers_0_conv(x))  # [1, frames, 1280]
        x = nn.gelu(self.conv_layers_1_conv(x))  # [1, frames/2, 1280]

        x = x.squeeze(0)  # [seq, 1280]
        seq_len = x.shape[0]

        # Left-truncate to multiple of downsample_factor
        trunc = seq_len % self.config.downsample_factor
        if trunc > 0:
            x = x[trunc:]
            seq_len = x.shape[0]

        # Pre-compute RoPE frequencies once for all 32 layers
        positions = mx.arange(seq_len)
        rope_cos, rope_sin = _compute_rope_freqs(
            positions, self.config.head_dim, self.config.rope_theta
        )

        # Pre-compute causal sliding window mask once for all 32 layers
        qi = mx.arange(seq_len)[:, None]
        ki = mx.arange(seq_len)[None, :]
        causal = ki <= qi
        window = ki >= (qi - self.config.sliding_window + 1)
        mask = mx.where(causal & window, mx.array(0.0), mx.array(-1e9))

        # Transformer layers (reuse rope + mask)
        for layer in self.transformer_layers:
            x = layer(x, rope_cos, rope_sin, mask)

        # Final norm
        x = self.transformer_norm(x)

        # 4x downsample: [seq, 1280] -> [seq/4, 5120]
        ds_len = seq_len // self.config.downsample_factor
        x = x.reshape(ds_len, self.config.dim * self.config.downsample_factor)

        # Adapter MLP
        x = nn.gelu(self.audio_language_projection_0(x))
        x = self.audio_language_projection_2(x)

        return x  # [seq/4, decoder_dim]

    def init_streaming_state(self) -> "StreamingEncoderState":
        """Create initial streaming state with empty caches."""
        return StreamingEncoderState(
            conv0_cache=None,
            conv1_cache=None,
            kv_caches=[None] * self.config.n_layers,
            downsample_buffer=None,
            position=0,
        )

    def forward_streaming(
        self,
        mel_chunk: mx.array,
        state: "StreamingEncoderState",
    ) -> tuple:
        """Process a small mel chunk incrementally.

        Args:
            mel_chunk: [mel_bins, frames] — small mel chunk
            state: StreamingEncoderState with conv/KV/downsample caches

        Returns:
            (adapter_tokens, updated_state) where adapter_tokens is
            [n_new_tokens, decoder_dim] (may be empty if downsample
            buffer not yet full).
        """
        if mel_chunk.shape[1] == 0:
            return mx.zeros((0, 3072)), state

        # mel is [128, frames], transpose to [frames, 128]
        x = mel_chunk.T  # [frames, 128]
        x = x[None, :, :]  # [1, frames, 128]

        # --- Conv stem with streaming caches ---
        # Conv0: kernel=3, stride=1, padding=2 (kernel-stride)
        x, state.conv0_cache = self._stream_causal_conv(
            self.conv_layers_0_conv, x, state.conv0_cache
        )
        x = nn.gelu(x)

        # Conv1: kernel=3, stride=2, padding=1 (kernel-stride)
        x, state.conv1_cache = self._stream_causal_conv(
            self.conv_layers_1_conv, x, state.conv1_cache
        )
        x = nn.gelu(x)

        x = x.squeeze(0)  # [seq, 1280]
        seq_len = x.shape[0]

        if seq_len == 0:
            return mx.zeros((0, 3072)), state

        # --- Transformer layers with KV cache ---
        positions = mx.arange(state.position, state.position + seq_len)
        rope_cos, rope_sin = _compute_rope_freqs(
            positions, self.config.head_dim, self.config.rope_theta
        )

        new_kv_caches = []
        for i, layer in enumerate(self.transformer_layers):
            x, kv = self._stream_encoder_layer(
                layer, x, rope_cos, rope_sin, state.kv_caches[i]
            )
            new_kv_caches.append(kv)

        state.kv_caches = new_kv_caches
        state.position += seq_len

        # Final norm
        x = self.transformer_norm(x)

        # --- 4x downsample with buffer ---
        if state.downsample_buffer is not None:
            x = mx.concatenate([state.downsample_buffer, x], axis=0)

        total = x.shape[0]
        ds_factor = self.config.downsample_factor
        n_complete = total // ds_factor
        remainder = total % ds_factor

        if n_complete == 0:
            state.downsample_buffer = x
            return mx.zeros((0, 3072)), state

        # Process complete groups
        usable = n_complete * ds_factor
        ds_input = x[:usable].reshape(n_complete, self.config.dim * ds_factor)

        # Save remainder for next call
        state.downsample_buffer = x[usable:] if remainder > 0 else None

        # Adapter MLP (pointwise, no sequence dependency)
        adapter_out = nn.gelu(self.audio_language_projection_0(ds_input))
        adapter_out = self.audio_language_projection_2(adapter_out)

        return adapter_out, state  # [n_new_tokens, decoder_dim]

    @staticmethod
    def _stream_causal_conv(conv_module, x, cache):
        """Run a CausalConv1d incrementally with a left-padding cache.

        Args:
            conv_module: CausalConv1d instance
            x: [1, seq, channels] input
            cache: [1, cache_len, channels] or None

        Returns:
            (output, new_cache)
        """
        pad_size = conv_module.padding
        if cache is not None:
            x = mx.concatenate([cache, x], axis=1)
        elif pad_size > 0:
            # First call: zero-pad left (matches CausalConv1d behavior)
            x = mx.pad(x, [(0, 0), (pad_size, 0), (0, 0)])

        # Save the tail as cache for next call
        if pad_size > 0:
            new_cache = x[:, -pad_size:, :]
        else:
            new_cache = None

        # Run the raw conv (skip CausalConv1d's own padding by calling conv directly)
        out = conv_module.conv(x)
        return out, new_cache

    @staticmethod
    def _stream_encoder_layer(layer, x, rope_cos, rope_sin, kv_cache):
        """Run one encoder layer with KV cache for streaming.

        Args:
            layer: EncoderLayer
            x: [seq, dim] new input
            rope_cos, rope_sin: [seq, head_dim//2] for new positions
            kv_cache: (k_cache, v_cache) or None

        Returns:
            (output, new_kv_cache)
        """
        attn = layer.attention
        seq_len = x.shape[0]

        # Attention with KV cache
        h = layer.attention_norm(x)
        q = attn.wq(h)
        k = attn.wk(h)
        v = attn.wv(h)

        # RoPE on new positions only
        q = _interleaved_rope(q, rope_cos, rope_sin, attn.n_heads, attn.head_dim)
        k = _interleaved_rope(k, rope_cos, rope_sin, attn.n_heads, attn.head_dim)

        # Append to KV cache
        if kv_cache is not None:
            k_old, v_old = kv_cache
            k_full = mx.concatenate([k_old, k], axis=0)
            v_full = mx.concatenate([v_old, v], axis=0)
        else:
            k_full = k
            v_full = v

        kv_len = k_full.shape[0]

        # Trim to sliding window
        if kv_len > attn.sliding_window:
            trim = kv_len - attn.sliding_window
            k_full = k_full[trim:]
            v_full = v_full[trim:]
            kv_len = attn.sliding_window

        # Save cache (2D, before reshape)
        new_kv = (k_full, v_full)

        # Reshape for attention
        q4 = q.reshape(1, seq_len, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        k4 = k_full.reshape(1, kv_len, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
        v4 = v_full.reshape(1, kv_len, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(attn.head_dim)

        # Mask: causal sliding window for new queries against full KV
        if seq_len == 1 and kv_len <= attn.sliding_window:
            mask = None
        else:
            # q positions are the last seq_len positions in the sequence
            total_seen = kv_len  # total KV positions after trim
            q_offset = total_seen - seq_len
            qi = mx.arange(q_offset, total_seen)[:, None]
            ki = mx.arange(total_seen)[None, :]
            causal = ki <= qi
            window = ki >= (qi - attn.sliding_window + 1)
            mask = mx.where(causal & window, mx.array(0.0), mx.array(-1e9))

        attn_out = mx.fast.scaled_dot_product_attention(
            q4, k4, v4, scale=scale, mask=mask
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(seq_len, attn.n_heads * attn.head_dim)
        h = attn.wo(attn_out)
        x = x + h

        # SwiGLU FFN
        h = layer.ffn_norm(x)
        gate = nn.silu(layer.feed_forward_w1(h))
        up = layer.feed_forward_w3(h)
        x = x + layer.feed_forward_w2(gate * up)

        return x, new_kv


@dataclass
class StreamingEncoderState:
    """Persistent state for incremental audio encoding."""

    conv0_cache: Optional[mx.array]
    conv1_cache: Optional[mx.array]
    kv_caches: list  # Per-layer (k, v) tuples or None
    downsample_buffer: Optional[mx.array]
    position: int
