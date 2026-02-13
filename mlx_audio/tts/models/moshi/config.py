from dataclasses import dataclass, field
from typing import Optional

from mlx_audio.utils import from_dict


@dataclass
class ModelConfig:
    """Configuration for Moshi TTS model."""

    model_type: str = "moshi_tts"
    hf_repo: str = "kyutai/tts-1.6b-en_fr"
    voice_repo: str = "kyutai/tts-voices"

    # Generation parameters
    temp: float = 0.8
    cfg_coef: float = 1.0
    n_q: int = 8
    max_gen_length: int = 4096
    padding_bonus: int = 0

    # Padding parameters
    initial_padding: int = 0
    max_padding: int = 0
    final_padding: int = 0
    padding_between: int = 0

    model_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Build config from config.json format used by Moshi TTS models."""
        config = {
            "model_type": data.get("model_type", "moshi_tts"),
            "hf_repo": data.get("hf_repo", "kyutai/tts-1.6b-en_fr"),
            "voice_repo": data.get("voice_repo", "kyutai/tts-voices"),
            "model_path": data.get("model_path"),
        }

        # Extract tts_config
        tts_config = data.get("tts_config", {})
        if tts_config:
            config["temp"] = tts_config.get("temp", 0.8)
            config["cfg_coef"] = tts_config.get("cfg_coef", 1.0)
            config["n_q"] = tts_config.get("n_q", 8)
            config["max_gen_length"] = tts_config.get("max_gen_length", 4096)
            config["padding_bonus"] = tts_config.get("padding_bonus", 0)
            config["initial_padding"] = tts_config.get("initial_padding", 0)
            config["max_padding"] = tts_config.get("max_padding", 0)
            config["final_padding"] = tts_config.get("final_padding", 0)
            config["padding_between"] = tts_config.get("padding_between", 0)

        return from_dict(cls, config)
