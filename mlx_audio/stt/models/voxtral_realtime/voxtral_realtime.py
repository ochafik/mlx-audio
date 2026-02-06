import math
import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from tqdm import tqdm

from mlx_audio.stt.generate import wired_limit
from mlx_audio.stt.utils import get_model_path

from ..base import STTOutput
from .config import AudioConfig, ModelConfig, TextConfig


# =============================================================================
# Encoder Components
# =============================================================================


class CausalConv1d(nn.Module):
    """Conv1d with left-only (causal) padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding_total = kernel_size - stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, C] (MLX convention)
        if self.padding_total > 0:
            x = mx.pad(x, [(0, 0), (self.padding_total, 0), (0, 0)])
        return self.conv(x)


class EncoderAttention(nn.Module):
    """Multi-head attention with RoPE for the causal encoder."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.head_dim = config.head_dim
        self.attn_dim = self.num_heads * self.head_dim
        self.sliding_window = config.sliding_window

        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.embed_dim, self.attn_dim, bias=True)
        self.k_proj = nn.Linear(self.embed_dim, self.attn_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.attn_dim, bias=True)
        self.out_proj = nn.Linear(self.attn_dim, self.embed_dim, bias=True)

        self.rope = nn.RoPE(
            self.head_dim, traditional=True, base=config.rope_theta
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.attn_dim)
        return self.out_proj(out)


class EncoderSwiGLU(nn.Module):
    """SwiGLU feed-forward for the encoder."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.encoder_ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.encoder_ffn_dim, config.d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class RealtimeEncoderLayer(nn.Module):
    """Causal encoder transformer layer with RMSNorm + RoPE + SwiGLU."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.self_attn = EncoderAttention(config)
        self.self_attn_layer_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.gate_proj = EncoderSwiGLU(config)
        self.final_layer_norm = nn.RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        r = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, mask=mask)
        x = r + x

        r = x
        x = self.final_layer_norm(x)
        x = self.gate_proj(x)
        x = r + x
        return x


class RealtimeEncoder(nn.Module):
    """Causal Whisper-style encoder with RoPE, RMSNorm, SwiGLU."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.downsample_factor = config.downsample_factor

        self.conv1 = CausalConv1d(config.num_mel_bins, embed_dim, kernel_size=3, stride=1)
        self.conv2 = CausalConv1d(embed_dim, embed_dim, kernel_size=3, stride=2)

        self.layers = [
            RealtimeEncoderLayer(config) for _ in range(config.encoder_layers)
        ]
        self.layer_norm = nn.RMSNorm(embed_dim, eps=config.rms_norm_eps)

    def _make_causal_mask(self, T: int, sliding_window: Optional[int] = None) -> mx.array:
        """Create causal attention mask with optional sliding window."""
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        if sliding_window is not None:
            # Zero out positions beyond the sliding window
            row_ids = mx.arange(T)[:, None]
            col_ids = mx.arange(T)[None, :]
            window_mask = mx.where(
                (row_ids - col_ids) >= sliding_window,
                mx.array(float("-inf")),
                mx.array(0.0),
            )
            mask = mask + window_mask
        return mask

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, n_mels]
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))

        # Truncate to align with downsample_factor
        T = x.shape[1]
        remainder = T % self.downsample_factor
        if remainder != 0:
            x = x[:, remainder:]

        T = x.shape[1]
        mask = self._make_causal_mask(T, self.config.sliding_window)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.layer_norm(x)


# =============================================================================
# Adapter (Multi-Modal Projector)
# =============================================================================


class MultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear_1 = nn.Linear(
            config.audio_config.intermediate_size,
            config.text_config.hidden_size,
            bias=False,
        )
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear_1(x)
        x = nn.gelu(x)
        x = self.linear_2(x)
        return x


# =============================================================================
# Time Conditioning
# =============================================================================


def time_embedding(n_delay_tokens: int, dim: int, theta: float = 10000.0) -> mx.array:
    """Sinusoidal embedding for encoding the delay token count."""
    t = mx.array([n_delay_tokens], dtype=mx.float32)
    half_dim = dim // 2
    inv_freq = mx.exp(
        -math.log(theta) * mx.arange(half_dim).astype(mx.float32) / half_dim
    )
    emb = t[:, None] * inv_freq[None, :]  # [1, D/2]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)  # [1, D]


class AdaptiveRMSNorm(nn.Module):
    """FiLM-style adaptive conditioning on t_cond.

    Applied after post_attention_layernorm, so no separate normalization needed.
    Computes: x * (1 + mlp(t_cond)) where mlp is hidden→cond_dim→hidden.
    """

    def __init__(self, hidden_size: int, cond_dim: int = 32):
        super().__init__()
        self.down = nn.Linear(hidden_size, cond_dim, bias=False)
        self.up = nn.Linear(cond_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array, t_cond: mx.array) -> mx.array:
        ada_scale = self.up(nn.gelu(self.down(t_cond)))  # [1, hidden_size]
        return x * (1.0 + ada_scale)


# =============================================================================
# Decoder Components
# =============================================================================


class DecoderAttention(nn.Module):
    """GQA attention for the decoder with RoPE and KV cache."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rope = nn.RoPE(
            self.head_dim, traditional=config.rope_traditional, base=config.rope_theta
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class DecoderSwiGLU(nn.Module):
    """SwiGLU MLP for the decoder (no biases)."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Decoder layer with adaptive RMSNorm for FiLM conditioning."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.self_attn = DecoderAttention(config)
        self.mlp = DecoderSwiGLU(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.ada_rms_norm_t_cond:
            self.ada_rms_norm = AdaptiveRMSNorm(
                config.hidden_size, cond_dim=config.ada_rms_norm_t_cond_dim,
            )
        else:
            self.ada_rms_norm = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        # Self-attention with pre-norm
        r = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask=mask, cache=cache)
        x = r + x

        # MLP with pre-norm + optional adaptive conditioning
        r = x
        x = self.post_attention_layernorm(x)
        if self.ada_rms_norm is not None and t_cond is not None:
            x = self.ada_rms_norm(x, t_cond)
        x = self.mlp(x)
        x = r + x

        return x


class MistralRealtimeModel(nn.Module):
    """Decoder-only transformer for the realtime model."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[List] = None,
        input_embeddings: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        mask = None
        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1])
            mask = mask.astype(h.dtype)

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h = layer(h, mask=mask, cache=layer_cache, t_cond=t_cond)

        return self.norm(h)


class LanguageModel(nn.Module):
    """Language model wrapper with tied embeddings."""

    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model = MistralRealtimeModel(config)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache: Optional[List] = None,
        input_embeddings: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(
            inputs, cache=cache, input_embeddings=input_embeddings, t_cond=t_cond
        )
        # Tied embeddings
        if self.config.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers


# =============================================================================
# Top-Level Model
# =============================================================================


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.text_config.vocab_size

        self.language_model = LanguageModel(config.text_config)
        self.audio_tower = RealtimeEncoder(config.audio_config)
        self.multi_modal_projector = MultiModalProjector(config)

    def get_audio_embeds(self, mel: mx.array) -> mx.array:
        """Encode mel spectrogram to audio embeddings."""
        # mel: [B, T, n_mels]
        encoded = self.audio_tower(mel)  # [B, T', 1280]

        B, T, D = encoded.shape
        # 4x downsample via reshape
        encoded = encoded.reshape(B, T // self.config.audio_config.downsample_factor, -1)
        # Project to text dim
        audio_embeds = self.multi_modal_projector(encoded)
        return audio_embeds

    def _merge_input_embeddings(
        self,
        input_ids: mx.array,
        input_features: Optional[mx.array] = None,
        cache: Optional[List] = None,
    ) -> mx.array:
        """Merge audio and text embeddings via summation.

        Audio embeddings are summed with the text embeddings at the prefix
        positions. The text prompt length should be >= audio embedding length.
        """
        text_embeds = self.language_model.model.embed_tokens(input_ids)

        if input_features is not None and (cache is None or cache[0].offset == 0):
            audio_embeds = self.get_audio_embeds(input_features)
            audio_len = audio_embeds.shape[1]
            text_len = text_embeds.shape[1]

            # Sum audio embeddings with the text prefix
            merge_len = min(audio_len, text_len)
            text_prefix = text_embeds[:, :merge_len, :]
            summed = text_prefix + audio_embeds[:, :merge_len, :]

            parts = [summed]
            if merge_len < text_len:
                parts.append(text_embeds[:, merge_len:, :])
            text_embeds = mx.concatenate(parts, axis=1)

        return text_embeds

    def __call__(
        self,
        input_ids: mx.array,
        input_features: Optional[mx.array] = None,
        cache: Optional[List] = None,
        t_cond: Optional[mx.array] = None,
    ) -> mx.array:
        inputs_embeds = self._merge_input_embeddings(
            input_ids=input_ids,
            input_features=input_features,
            cache=cache,
        )
        logits = self.language_model(
            input_embeddings=inputs_embeds, cache=cache, t_cond=t_cond
        )
        return logits

    def sanitize(self, weights):
        """Remap weight keys from Mistral-native format to MLX format."""
        sanitized = {}
        for k, v in weights.items():
            new_key = k

            # Encoder convolution layers
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv",
                "audio_tower.conv1.conv",
            )
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv",
                "audio_tower.conv2.conv",
            )

            # Encoder transformer layers
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm",
                r"audio_tower.layers.\1.self_attn_layer_norm",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq",
                r"audio_tower.layers.\1.self_attn.q_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk",
                r"audio_tower.layers.\1.self_attn.k_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv",
                r"audio_tower.layers.\1.self_attn.v_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo",
                r"audio_tower.layers.\1.self_attn.out_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm",
                r"audio_tower.layers.\1.final_layer_norm",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1",
                r"audio_tower.layers.\1.gate_proj.gate_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2",
                r"audio_tower.layers.\1.gate_proj.down_proj",
                new_key,
            )
            new_key = re.sub(
                r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3",
                r"audio_tower.layers.\1.gate_proj.up_proj",
                new_key,
            )

            # Encoder final norm
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm",
                "audio_tower.layer_norm",
            )

            # Multi-modal projector
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.audio_language_projection.0",
                "multi_modal_projector.linear_1",
            )
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.audio_language_projection.2",
                "multi_modal_projector.linear_2",
            )

            # Embed tokens (from mm_streams_embeddings)
            new_key = new_key.replace(
                "mm_streams_embeddings.embedding_module.tok_embeddings",
                "language_model.model.embed_tokens",
            )

            # Decoder layers
            new_key = re.sub(
                r"^layers\.(\d+)\.attention_norm",
                r"language_model.model.layers.\1.input_layernorm",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.attention\.wq",
                r"language_model.model.layers.\1.self_attn.q_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.attention\.wk",
                r"language_model.model.layers.\1.self_attn.k_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.attention\.wv",
                r"language_model.model.layers.\1.self_attn.v_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.attention\.wo",
                r"language_model.model.layers.\1.self_attn.o_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.ffn_norm",
                r"language_model.model.layers.\1.post_attention_layernorm",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.feed_forward\.w1",
                r"language_model.model.layers.\1.mlp.gate_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.feed_forward\.w2",
                r"language_model.model.layers.\1.mlp.down_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.feed_forward\.w3",
                r"language_model.model.layers.\1.mlp.up_proj",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.ada_rms_norm_t_cond\.0",
                r"language_model.model.layers.\1.ada_rms_norm.down",
                new_key,
            )
            new_key = re.sub(
                r"^layers\.(\d+)\.ada_rms_norm_t_cond\.2",
                r"language_model.model.layers.\1.ada_rms_norm.up",
                new_key,
            )

            # Final decoder norm
            if new_key == "norm.weight":
                new_key = "language_model.model.norm.weight"

            # Transpose conv weights: PyTorch [out, in, k] → MLX [out, k, in]
            if "conv" in new_key and "weight" in new_key and v.ndim == 3:
                if v.shape[-1] < v.shape[-2]:
                    v = v.transpose(0, 2, 1)

            sanitized[new_key] = v

        return sanitized

    def model_quant_predicate(self, p, m):
        return not p.startswith("audio_tower")

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Initialize tokenizer and processor after loading weights."""
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError:
            raise ImportError(
                "mistral-common is required for voxtral_realtime. "
                "Install with: pip install mistral-common>=1.8.2"
            )

        # Try to load tekken tokenizer
        tekken_path = model_path / "tekken.json"
        if tekken_path.exists():
            tokenizer = MistralTokenizer.from_file(str(tekken_path))
        else:
            # Fall back to loading from model path
            tokenizer = MistralTokenizer.from_model(str(model_path))

        model._tokenizer = tokenizer
        model._eos_token_ids = [2, 4, 32000]

        # Store model_repo
        if not hasattr(model.config, "model_repo") or model.config.model_repo is None:
            try:
                index = model_path.parts.index("hub")
                model.config.model_repo = (
                    model_path.parts[index + 1]
                    .replace("models--", "")
                    .replace("--", "/")
                )
            except (ValueError, IndexError):
                model.config.model_repo = str(model_path)

        return model

    def _compute_mel_spectrogram(self, audio: mx.array) -> mx.array:
        """Compute log-mel spectrogram from audio waveform."""
        from mlx_audio.stt.models.voxtral_realtime.audio_utils import (
            log_mel_spectrogram,
        )

        mel = log_mel_spectrogram(audio)
        return mel

    def _build_prompt(
        self, audio, language: Optional[str] = "en"
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Build prompt tokens and compute mel features for transcription."""
        from mlx_audio.stt.utils import load_audio as stt_load_audio

        # Load and preprocess audio
        if isinstance(audio, str):
            waveform = stt_load_audio(audio)
        elif isinstance(audio, list):
            waveform = stt_load_audio(audio[0]) if isinstance(audio[0], str) else audio[0]
        else:
            waveform = audio

        waveform_np = np.array(waveform, dtype=np.float32)

        from mistral_common.protocol.instruct.chunk import RawAudio
        from mistral_common.protocol.transcription.request import (
            StreamingMode,
            TranscriptionRequest,
        )
        from mistral_common.tokens.tokenizers.audio import Audio

        audio_obj = Audio(waveform_np, 16000, format="wav")
        request = TranscriptionRequest(
            audio=RawAudio.from_audio(audio_obj),
            language=language,
            streaming=StreamingMode.OFFLINE,
        )
        tokenized = self._tokenizer.encode_transcription(request)
        token_ids = tokenized.tokens
        audio_array = tokenized.audios[0].audio_array

        # Compute mel from the processed audio
        audio_mx = mx.array(audio_array, dtype=mx.float32)
        mel = self._compute_mel_spectrogram(audio_mx)
        if mel.ndim == 2:
            mel = mel[None, :, :]  # Add batch dim

        # Compute t_cond from delay tokens
        audio_config = self._tokenizer.instruct_tokenizer.audio_encoder.audio_config
        n_delay_tokens = audio_config.num_delay_tokens
        t_cond = time_embedding(n_delay_tokens, self.config.text_config.hidden_size)

        input_ids = mx.array([token_ids])
        return input_ids, mel, t_cond

    def stream_generate(
        self,
        input_ids: Optional[mx.array] = None,
        *,
        input_features: Optional[mx.array] = None,
        t_cond: Optional[mx.array] = None,
        max_tokens: int = 128,
        sampler: Optional[Callable] = None,
        generation_stream: bool = False,
        verbose: bool = False,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Stream token generation with per-step audio embedding summation.

        In the realtime model, audio embeddings are summed with text embeddings
        at every position, including during autoregressive generation — not just
        during prefill.
        """
        from mlx_lm.models.cache import KVCache

        # Default sampler: greedy
        if sampler is None:
            sampler = lambda logits: mx.argmax(logits, axis=-1)

        # Pre-compute all audio embeddings
        audio_embeds = None
        audio_len = 0
        if input_features is not None:
            audio_embeds = self.get_audio_embeds(input_features)  # [1, A, D]
            audio_len = audio_embeds.shape[1]

        # Get text embeddings for prompt
        text_embeds = self.language_model.model.embed_tokens(input_ids)  # [1, L, D]
        prompt_len = input_ids.shape[1]

        # Sum audio with text for the prompt positions
        if audio_embeds is not None:
            merge_len = min(audio_len, prompt_len)
            summed_prefix = (
                text_embeds[:, :merge_len, :] + audio_embeds[:, :merge_len, :]
            )
            if merge_len < prompt_len:
                prefill_embeds = mx.concatenate(
                    [summed_prefix, text_embeds[:, merge_len:, :]], axis=1
                )
            else:
                prefill_embeds = summed_prefix
        else:
            prefill_embeds = text_embeds

        # Create KV cache for each decoder layer
        cache = [KVCache() for _ in self.language_model.layers]

        from mlx_audio.stt.generate import generation_stream as gen_stream

        with wired_limit(self, [gen_stream] if generation_stream else None):
            # Prefill: process all prompt tokens through the decoder
            logits = self.language_model(
                input_embeddings=prefill_embeds, cache=cache, t_cond=t_cond
            )

            # Sample first token
            y = sampler(logits[0, -1, :])
            logprobs = logits[0, -1, :]

            if int(y) in self._eos_token_ids:
                return
            yield y, logprobs

            # Autoregressive loop — sum audio embed with token embed at each step
            current_pos = prompt_len
            for step in tqdm(
                range(max_tokens - 1),
                disable=not verbose,
                desc="Generating",
            ):
                # Embed the last generated token
                token_embed = self.language_model.model.embed_tokens(
                    y.reshape(1, 1)
                )  # [1, 1, D]

                # Sum with audio embedding at current position if available
                if audio_embeds is not None and current_pos < audio_len:
                    token_embed = (
                        token_embed
                        + audio_embeds[:, current_pos : current_pos + 1, :]
                    )

                # Forward pass through decoder with KV cache
                logits = self.language_model(
                    input_embeddings=token_embed, cache=cache, t_cond=t_cond
                )

                # Sample next token
                y = sampler(logits[0, -1, :])
                logprobs = logits[0, -1, :]

                if int(y) in self._eos_token_ids:
                    break

                yield y, logprobs
                current_pos += 1

    def generate(
        self,
        audio,
        *,
        message: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
        language: str = "en",
        verbose: bool = False,
        generation_stream: bool = False,
    ) -> STTOutput:
        """Transcribe audio to text."""
        start_time = time.time()

        input_ids, mel, t_cond = self._build_prompt(audio, language=language)

        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(
            temperature,
            top_p,
            min_p,
            min_tokens_to_keep=min_tokens_to_keep,
            top_k=top_k,
        )

        generated = []
        for token, _ in self.stream_generate(
            input_ids=input_ids,
            input_features=mel,
            t_cond=t_cond,
            max_tokens=max_tokens,
            sampler=sampler,
            generation_stream=generation_stream,
            verbose=verbose,
        ):
            generated.append(token)

        end_time = time.time()
        mx.clear_cache()

        # Decode tokens — convert mx.array items to int
        token_list = [int(t) for t in generated]
        text = self._tokenizer.instruct_tokenizer.tokenizer.decode(token_list)

        return STTOutput(
            text=text,
            prompt_tokens=input_ids.shape[1],
            generation_tokens=len(generated),
            total_tokens=input_ids.shape[1] + len(generated),
            total_time=end_time - start_time,
            prompt_tps=input_ids.shape[1] / (end_time - start_time),
            generation_tps=len(generated) / max(end_time - start_time, 1e-9),
        )
