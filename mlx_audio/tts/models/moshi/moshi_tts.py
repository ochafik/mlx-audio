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

    def model_quant_predicate(self, p, m):
        """Skip quantization on embeddings and norm layers.

        Note: Moshi TTS handles quantization directly in post_load_hook
        using explicit nn.quantize() on specific layers (depformer, attention, gating).
        This predicate is provided for framework consistency.
        """
        skip_patterns = ["embed", "norm", "embedding"]
        return not any(pat in p.lower() for pat in skip_patterns)

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

        # Load Mimi decoder - use generated_codebooks from LM config, not config.n_q
        generated_codebooks = lm_config.generated_codebooks
        mimi_cfg = mimi_202407(generated_codebooks)
        mimi = Mimi(mimi_cfg)
        mimi.load_pytorch_weights(str(mimi_weight), strict=False)
        logger.info(f"Mimi loaded with {generated_codebooks} codebooks")

        # Use generated_codebooks for n_q (TTS models typically use 32, not 8)
        n_q = config.n_q if config.n_q > 0 else generated_codebooks

        # Create TTS model wrapper
        model._tts_model = TTSModel(
            lm=lm,
            mimi=mimi,
            tokenizer=model._tokenizer,
            raw_config=config_dict,
            temp=config.temp,
            cfg_coef=config.cfg_coef,
            n_q=n_q,
            max_gen_length=config.max_gen_length,
            padding_bonus=config.padding_bonus,
            initial_padding=config.initial_padding,
            max_padding=config.max_padding,
            final_padding=config.final_padding,
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
            voice: Voice name (for multi-speaker models). If not provided, uses a default voice.
            ref_audio: Reference audio path for voice cloning (for single-speaker models).
            temperature: Sampling temperature (uses config default if not provided).
            stream: If True, yield audio chunks as they're generated.

        Yields:
            GenerationResult with audio data.
        """
        start_time = time.time()

        # Prepare script entries
        entries = self._tts_model.prepare_script([text])

        # Build condition attributes with voice
        if self._tts_model.multi_speaker:
            # Multi-speaker model: use voice embedding
            if voice is None:
                # Use a default voice from expresso dataset
                voice = "expresso/ex01-ex02_default_001_channel1_168s.wav"
                logger.info(f"Using default voice: {voice}")
            voice_path = self._tts_model.get_voice_path(voice)
            voices = [voice_path]
            attributes = self._tts_model.make_condition_attributes(
                voices, cfg_coef=self.config.cfg_coef
            )
        else:
            # Single-speaker model: uses CFG or reference audio
            attributes = self._tts_model.make_condition_attributes(
                [], cfg_coef=self.config.cfg_coef
            )

        # Get prefix if reference audio provided (for single-speaker models)
        prefixes = None
        if ref_audio and not self._tts_model.multi_speaker:
            prefixes = self._tts_model.get_prefix(ref_audio)

        if stream:
            # Streaming mode: collect frames via callback
            audio_chunks = []

            def on_frame(frame_codes: mx.array):
                # Frame shape is (batch, codebooks), need to add time dimension for decode_step
                frame_with_time = frame_codes[:, :, None]  # (batch, codebooks, 1)
                audio = self._mimi.decode_step(frame_with_time)  # (1, 1, 1920)
                mx.eval(audio)
                audio_chunks.append(audio)

            # Run generation with frame callback
            result = self._tts_model.generate(
                all_entries=[entries],  # entries is a list of Entry objects, wrap in another list for batch
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
                all_entries=[entries],  # entries is a list of Entry objects, wrap in another list for batch
                attributes=[attributes],
                prefixes=prefixes,
            )

            # Decode frames step by step (like run_tts.py)
            # Skip first delay_steps frames - they contain garbage audio
            frames_to_decode = result.frames[self._tts_model.delay_steps:]
            wav_frames = []
            for frame in frames_to_decode:
                pcm = self._mimi.decode_step(frame)
                wav_frames.append(pcm)

            # Remove first 2 frames to avoid click/noise at the beginning
            if len(wav_frames) > 2:
                wav_frames = wav_frames[2:]

            if wav_frames:
                audio = mx.concat(wav_frames, axis=-1)
                mx.eval(audio)
                audio_np = np.array(audio.squeeze())
            else:
                audio_np = np.array([])

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
