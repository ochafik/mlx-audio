import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AudioConfig:
    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    head_dim: int = 64
    intermediate_size: int = 5120
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0
    num_mel_bins: int = 128
    encoder_layers: int = 32
    encoder_attention_heads: int = 32
    encoder_ffn_dim: int = 5120
    d_model: int = 1280
    max_source_positions: int = 1500
    scale_embedding: bool = False
    is_causal: bool = True
    sliding_window: Optional[int] = 750
    downsample_factor: int = 4

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class TextConfig:
    model_type: str = "mistral_realtime"
    vocab_size: int = 131072
    max_position_embeddings: int = 131072
    hidden_size: int = 3072
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    head_dim: int = 128
    tie_word_embeddings: bool = True
    rope_theta: float = 1000000.0
    rope_traditional: bool = True
    sliding_window: Optional[int] = 8192
    ada_rms_norm_t_cond: bool = True
    ada_rms_norm_t_cond_dim: int = 32
    bos_token_id: int = 1
    eos_token_id: int = 2

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class ModelConfig:
    audio_config: AudioConfig = None
    text_config: TextConfig = None
    model_repo: str = None
    model_type: str = "voxtral_realtime"
    audio_token_id: int = 24
    projector_hidden_act: str = "gelu"
    vocab_size: int = 131072
    hidden_size: int = 3072

    def __post_init__(self):
        if self.audio_config is None:
            self.audio_config = AudioConfig()
        if self.text_config is None:
            self.text_config = TextConfig()
        if isinstance(self.audio_config, dict):
            self.audio_config = AudioConfig.from_dict(self.audio_config)
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)

        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size

    @classmethod
    def from_dict(cls, params):
        params = params.copy()
        if "audio_config" in params and isinstance(params["audio_config"], dict):
            params["audio_config"] = AudioConfig.from_dict(params["audio_config"])
        elif "audio_config" not in params:
            params["audio_config"] = AudioConfig()

        if "text_config" in params and isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])
        elif "text_config" not in params:
            params["text_config"] = TextConfig()

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
