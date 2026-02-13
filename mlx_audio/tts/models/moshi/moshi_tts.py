"""Moshi TTS model with streaming support for MLX Audio."""

import json
import logging
import time
from pathlib import Path
from typing import Generator, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.tts.models.base import GenerationResult

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Constants
SAMPLE_RATE = 24000
FRAME_SIZE = 1920  # 80ms at 24kHz


class Model(nn.Module):
    """Moshi TTS model with streaming support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._tts_model = None
        self._mimi = None
        self._tokenizer = None

    def sanitize(self, weights):
        """Skip standard weight loading - we use post_load_hook."""
        return {}

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Load Moshi TTS model weights and components."""
        try:
            from moshi_mlx.models import Lm, LmConfig, mimi_202407
            from moshi_mlx.models.mimi import Mimi
            from moshi_mlx.models.tts import TTSModel
            import sentencepiece
        except ImportError as e:
            raise ImportError(
                f"Moshi dependencies not installed. Install with: pip install moshi-mlx\n{e}"
            ) from e

        config = model.config

        # Load config.json
        config_path = model_path / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

        # Get file names from config
        moshi_name = config_dict.get("moshi_name", "model.safetensors")
        mimi_name = config_dict.get("mimi_name", "tokenizer-e351c8d8-checkpoint125.safetensors")
        tokenizer_name = config_dict.get("tokenizer_name", "tokenizer_spm_32k_3.model")

        # Resolve paths
        moshi_weight = model_path / moshi_name
        mimi_weight = model_path / mimi_name
        tokenizer_path = model_path / tokenizer_name

        # Load tokenizer
        model._tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        # Build LmConfig
        if config_dict:
            lm_config = LmConfig.from_config_dict(config_dict)
        else:
            raise ValueError("No config.json found - required for Moshi TTS")

        # Load LM
        lm = Lm(lm_config)
        lm.set_dtype(mx.bfloat16)
        lm.load_pytorch_weights(str(moshi_weight), lm_config, strict=True)

        # Load Mimi decoder
        mimi_cfg = mimi_202407(config.n_q)
        mimi = Mimi(mimi_cfg)
        mimi.load_pytorch_weights(str(mimi_weight), strict=True)

        # Create TTS model wrapper
        model._tts_model = TTSModel(
            lm=lm,
            mimi=mimi,
            tokenizer=model._tokenizer,
            raw_config=config_dict,
            temp=config.temp,
            cfg_coef=config.cfg_coef,
            n_q=config.n_q,
            max_gen_length=config.max_gen_length,
            padding_bonus=config.padding_bonus,
            initial_padding=config.initial_padding,
            max_padding=config.max_padding,
            final_padding=config.final_padding,
            padding_between=config.padding_between,
        )
        model._mimi = mimi

        logger.info(f"Moshi TTS ready: temp={config.temp}, cfg={config.cfg_coef}")

        return model

    def generate(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        ref_audio: Optional[str] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate audio from text.

        Args:
            text: Text to synthesize.
            voice: Voice name (for multi-speaker models).
            ref_audio: Reference audio path for voice cloning.
            temperature: Sampling temperature (uses config default if not provided).
            stream: If True, yield audio chunks as they're generated.

        Yields:
            GenerationResult with audio data.
        """
        start_time = time.time()

        # Prepare script entries
        entries = self._tts_model.prepare_script([text])

        # Build condition attributes
        voices = [voice] if voice else None
        attributes = self._tts_model.make_condition_attributes(
            voices, self.config.cfg_coef
        )

        # Get prefix if reference audio provided
        prefixes = None
        if ref_audio and not self._tts_model.multi_speaker:
            prefixes = self._tts_model.get_prefix(ref_audio)

        if stream:
            # Streaming mode: collect frames via callback
            audio_chunks = []

            def on_frame(frame_codes: mx.array):
                # Decode frame to audio
                audio = self._mimi.decode_step(frame_codes)  # (1, 1, 1920)
                mx.eval(audio)
                audio_chunks.append(audio)

            # Run generation with frame callback
            result = self._tts_model.generate(
                all_entries=entries,
                attributes=[attributes],
                prefixes=prefixes,
                on_frame=on_frame,
            )

            # Yield each chunk
            total_samples = 0
            for i, chunk in enumerate(audio_chunks):
                chunk_np = np.array(chunk.squeeze())  # (1920,)
                samples = len(chunk_np)
                total_samples += samples
                duration = samples / SAMPLE_RATE

                yield GenerationResult(
                    audio=mx.array(chunk_np),
                    samples=samples,
                    sample_rate=SAMPLE_RATE,
                    segment_idx=i,
                    token_count=0,  # Not tracked per-chunk
                    audio_samples=samples,
                    audio_duration=f"{duration:.2f}",
                    real_time_factor=0.0,
                    prompt={"text": text},
                    processing_time_seconds=time.time() - start_time,
                    peak_memory_usage=0.0,
                    is_streaming_chunk=True,
                    is_final_chunk=(i == len(audio_chunks) - 1),
                )

        else:
            # Batch mode: generate all at once
            result = self._tts_model.generate(
                all_entries=entries,
                attributes=[attributes],
                prefixes=prefixes,
            )

            # Decode all frames
            all_codes = result.audio_codes  # Shape depends on model
            audio = self._mimi.decode(all_codes)
            mx.eval(audio)

            audio_np = np.array(audio.squeeze())
            samples = len(audio_np)
            duration = samples / SAMPLE_RATE
            elapsed = time.time() - start_time

            yield GenerationResult(
                audio=mx.array(audio_np),
                samples=samples,
                sample_rate=SAMPLE_RATE,
                segment_idx=0,
                token_count=len(text.split()),
                audio_samples=samples,
                audio_duration=f"{duration:.2f}",
                real_time_factor=elapsed / duration if duration > 0 else 0,
                prompt={"text": text},
                processing_time_seconds=elapsed,
                peak_memory_usage=0.0,
                is_streaming_chunk=False,
                is_final_chunk=True,
            )
