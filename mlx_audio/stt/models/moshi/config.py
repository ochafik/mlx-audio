from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.utils import from_dict


@dataclass
class ModelConfig:
    """Configuration for Moshi STT model."""

    model_type: str = "moshi_stt"
    hf_repo: str = "kyutai/stt-1b-en_fr-mlx"
    moshi_name: str = "model.safetensors"
    mimi_name: str = "tokenizer-e351c8d8-checkpoint125.safetensors"
    tokenizer_name: str = "tokenizer_spm_32k_3.model"

    # LM generation config
    text_temp: float = 0.0
    text_top_k: int = 50

    # STT-specific padding
    audio_silence_prefix_seconds: float = 0.0
    audio_delay_seconds: float = 0.0

    # Model architecture
    skip_depformer: bool = True
    quantized: Optional[int] = None  # 4 or 8 for quantization

    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build config from config.json format used by Moshi models."""
        # Extract Moshi-specific field names
        config = {
            "model_type": data.get("model_type", "moshi_stt"),
            "hf_repo": data.get("hf_repo", "kyutai/stt-1b-en_fr-mlx"),
            "moshi_name": data.get("moshi_name", "model.safetensors"),
            "mimi_name": data.get("mimi_name", "tokenizer-e351c8d8-checkpoint125.safetensors"),
            "tokenizer_name": data.get("tokenizer_name", "tokenizer_spm_32k_3.model"),
            "model_path": data.get("model_path"),
        }

        # Extract lm_gen_config
        lm_gen_config = data.get("lm_gen_config", {})
        if lm_gen_config:
            config["text_temp"] = lm_gen_config.get("temp_text", 0.0)
            config["text_top_k"] = lm_gen_config.get("top_k_text", 50)

        # Extract stt_config
        stt_config = data.get("stt_config", {})
        if stt_config:
            config["audio_silence_prefix_seconds"] = stt_config.get(
                "audio_silence_prefix_seconds", 0.0
            )
            config["audio_delay_seconds"] = stt_config.get("audio_delay_seconds", 0.0)

        # Detect quantization from repo name
        model_path = config.get("model_path", "") or config.get("hf_repo", "")
        if "-q4" in model_path:
            config["quantized"] = 4
        elif "-q8" in model_path:
            config["quantized"] = 8

        return from_dict(cls, config)
