"""Moshi STT model with streaming support for MLX Audio."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.stt.models.base import STTOutput, StreamingInputModel

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Constants from moshi_stt_mlx
SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME_SIZE = int(SAMPLE_RATE / FRAME_RATE)  # 1920 samples = 80ms


@dataclass
class StreamingMoshiSession:
    """Streaming session state for Moshi STT."""

    # Audio buffer for incomplete frames
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Accumulated text tokens
    text_tokens: List[int] = field(default_factory=list)

    # Previous decoded text (for delta computation)
    prev_text: str = ""

    # Step counter
    step_idx: int = 0

    # Finished flag
    finished: bool = False

    # Back-reference to model (for delegation)
    _model: Optional[object] = field(default=None, repr=False)

    # LmGen instance for streaming
    _lm_gen: Optional[object] = field(default=None, repr=False)

    def reset(self):
        """Reset session state for a new utterance."""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.text_tokens = []
        self.prev_text = ""
        self.step_idx = 0
        self.finished = False


class Model(nn.Module):
    """Moshi STT model with streaming support."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Lazy-loaded components
        self._lm = None
        self._mimi = None
        self._tokenizer = None
        self._lm_gen_class = None
        self._sampler_class = None
        self._lm_config = None

        # Pre-computed values
        self._other_codebooks = None
        self._main_codebooks = None
        self._silence_replace = None
        self._condition_tensor = None

    def supports_streaming_input(self) -> bool:
        """Moshi STT supports frame-by-frame streaming input."""
        return True

    def sanitize(self, weights):
        """Skip standard weight loading - we use post_load_hook."""
        return {}

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        """Load Moshi model weights and components."""
        try:
            from moshi_mlx import models, utils
            from moshi_mlx.models.mimi import Mimi, mimi_202407
            import sentencepiece
            from moshi_stt_mlx.engine import _lm_config_from_dict, SILENCE_TOKENS
        except ImportError as e:
            raise ImportError(
                f"Moshi dependencies not installed. Install with: pip install moshi-mlx moshi-stt-mlx\n{e}"
            ) from e

        config = model.config

        # Load config.json
        config_path = model_path / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)

        # Get file names from config
        moshi_name = config_dict.get("moshi_name", config.moshi_name)
        mimi_name = config_dict.get("mimi_name", config.mimi_name)
        tokenizer_name = config_dict.get("tokenizer_name", config.tokenizer_name)

        # Resolve paths
        moshi_weight = model_path / moshi_name
        mimi_weight = model_path / mimi_name
        tokenizer_path = model_path / tokenizer_name

        # Handle quantized weight files
        quantized = config.quantized
        runtime_quantize = False
        if quantized == 4:
            q4_path = model_path / "model.q4.safetensors"
            if q4_path.exists():
                moshi_weight = q4_path
            else:
                runtime_quantize = True
        elif quantized == 8:
            q8_path = model_path / "model.q8.safetensors"
            if q8_path.exists():
                moshi_weight = q8_path
            else:
                runtime_quantize = True

        # Load tokenizer
        model._tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

        # Build LmConfig
        if config_dict:
            model._lm_config = _lm_config_from_dict(config_dict)
            logger.info(
                f"Config: dim={config_dict.get('dim')}, "
                f"layers={config_dict.get('num_layers')}, "
                f"dep_q={config_dict.get('dep_q', 0)}, "
                f"n_q={config_dict.get('n_q')}"
            )
        else:
            model._lm_config = models.config_v0_1()
            logger.info("No config.json, using default Moshi 7B config")

        # Get temperature and top_k from config
        text_temp = config.text_temp
        text_top_k = config.text_top_k
        if text_temp == 0.0:
            lm_gen_config = config_dict.get("lm_gen_config", {})
            text_temp = lm_gen_config.get("temp_text", 0.0)
            text_top_k = lm_gen_config.get("top_k_text", 50)

        # Store sampler class for later
        model._sampler_class = utils.Sampler
        model._text_temp = text_temp
        model._text_top_k = text_top_k

        # Load LM model
        lm = models.Lm(model._lm_config)
        lm.set_dtype(mx.bfloat16)

        if quantized is not None and not runtime_quantize:
            group_size = 32 if quantized == 4 else 64
            nn.quantize(lm, bits=quantized, group_size=group_size)

        logger.info(f"Loading LM weights: {moshi_weight}")
        lm.load_weights(str(moshi_weight), strict=True)

        if quantized is not None and runtime_quantize:
            group_size = 32 if quantized == 4 else 64
            logger.info(f"Quantizing LM to {quantized}-bit")
            nn.quantize(lm, bits=quantized, group_size=group_size)

        # Condition tensor (for models with condition_provider)
        if hasattr(lm, "condition_provider") and lm.condition_provider is not None:
            model._condition_tensor = lm.condition_provider.condition_tensor(
                "description", "very_good"
            )
        else:
            model._condition_tensor = None

        logger.info("Warming up LM...")
        lm.warmup(model._condition_tensor)
        model._lm = lm

        # Store codebook counts
        model._other_codebooks = model._lm_config.other_codebooks
        model._main_codebooks = model._lm_config.generated_codebooks

        # Store LmGen class
        model._lm_gen_class = models.LmGen

        # Load Mimi encoder
        mimi_codebooks = max(model._main_codebooks, model._other_codebooks)
        mimi_cfg = mimi_202407(mimi_codebooks)
        mimi = Mimi(mimi_cfg)
        logger.info(f"Loading Mimi weights: {mimi_weight}")
        mimi.load_pytorch_weights(str(mimi_weight), strict=True)

        # Warmup Mimi
        logger.info("Warming up Mimi encoder...")
        dummy_pcm = mx.zeros((1, 1, FRAME_SIZE * 4))
        dummy_codes = mimi.encode(dummy_pcm)
        mx.eval(dummy_codes)
        logger.info(f"Mimi ready: {mimi_codebooks} codebooks")
        model._mimi = mimi

        # Pre-compute silence replacement for depformer skipping
        if model._main_codebooks > 0:
            silence = SILENCE_TOKENS[: model._main_codebooks]
            model._silence_replace = mx.array(silence, dtype=mx.int32)[None, :, None]
        else:
            model._silence_replace = None

        # Store STT config for padding
        stt_config = config_dict.get("stt_config", {})
        model._stt_config = stt_config

        logger.info(
            f"Moshi STT ready: q={quantized}, "
            f"dep_q={model._main_codebooks}, other_cb={model._other_codebooks}, "
            f"temp={text_temp}, top_k={text_top_k}"
        )

        return model

    def generate(
        self,
        audio: Union[str, np.ndarray, mx.array],
        *,
        stream: bool = False,
        **kwargs,
    ) -> STTOutput:
        """Transcribe audio to text (batch mode).

        Args:
            audio: Audio file path, numpy array, or MLX array (float32 at 24kHz).
            stream: If True, yield text deltas as they're generated.

        Returns:
            STTOutput with transcription results.
        """
        # Load audio if path provided
        if isinstance(audio, str):
            import sphn
            audio, _ = sphn.read(audio, sample_rate=SAMPLE_RATE)
            audio = audio[0].astype(np.float32)
        elif isinstance(audio, mx.array):
            audio = np.array(audio)

        # Ensure float32
        audio = audio.astype(np.float32)

        orig_duration = len(audio) / SAMPLE_RATE

        # Apply STT padding
        if self._stt_config:
            pad_left = int(
                self._stt_config.get("audio_silence_prefix_seconds", 0.0) * SAMPLE_RATE
            )
            pad_right = int(
                (self._stt_config.get("audio_delay_seconds", 0.0) + 1.0) * SAMPLE_RATE
            )
            audio = np.pad(audio, (pad_left, pad_right), mode="constant")

        # Pad to frame boundary
        remainder = len(audio) % FRAME_SIZE
        if remainder > 0:
            audio = np.concatenate(
                [audio, np.zeros(FRAME_SIZE - remainder, dtype=np.float32)]
            )

        num_frames = len(audio) // FRAME_SIZE
        max_steps = num_frames + 10

        start_time = time.time()

        # Phase 1: Batch encode all audio
        pcm_tensor = mx.array(audio[np.newaxis, np.newaxis, :])  # (1, 1, T)
        self._mimi.reset_all()
        all_codes = self._mimi.encode(pcm_tensor)  # (1, codebooks, num_frames)
        mx.eval(all_codes)
        encode_time = time.time() - start_time

        # Slice to other_codebooks
        all_codes = all_codes[:, : self._other_codebooks, :]

        # Phase 2: Run LM step loop
        for c in self._lm.transformer_cache:
            c.reset()

        gen = self._lm_gen_class(
            model=self._lm,
            max_steps=max_steps,
            text_sampler=self._sampler_class(temp=self._text_temp, top_k=self._text_top_k),
            audio_sampler=self._sampler_class(temp=0.8, top_k=250),
            cfg_coef=1.0,
            check=False,
        )

        text_tokens = []
        lm_start = time.time()
        prev_text = ""

        for idx in range(num_frames):
            other_audio_tokens = all_codes[:, :, idx]

            text_token, _ = gen.step(
                other_audio_tokens[0],
                ct=self._condition_tensor,
                depformer_replace_tokens=self._silence_replace,
            )

            text_token_val = text_token[0].item()

            if text_token_val not in (0, 3):
                text_tokens.append(int(text_token_val))

            # Stream text deltas if requested
            if stream:
                current_text = self._tokenizer.decode(text_tokens) if text_tokens else ""
                if current_text != prev_text:
                    delta = current_text[len(prev_text):]
                    if delta:
                        yield STTOutput(text=delta)
                    prev_text = current_text

        lm_time = time.time() - lm_start
        elapsed = time.time() - start_time

        final_text = ""
        if text_tokens:
            final_text = self._tokenizer.decode(text_tokens).strip()

        if stream:
            # Yield final result with stats
            yield STTOutput(
                text=final_text,
                total_time=elapsed,
            )
        else:
            return STTOutput(
                text=final_text,
                total_time=elapsed,
            )

    def create_streaming_session(self, **kwargs) -> StreamingMoshiSession:
        """Create a new streaming session.

        Args:
            **kwargs: Optional parameters like temperature.

        Returns:
            StreamingMoshiSession ready for audio input.
        """
        # Reset Mimi encoder state
        self._mimi.reset_all()

        # Reset transformer caches
        for c in self._lm.transformer_cache:
            c.reset()

        # Create LmGen for streaming
        text_temp = kwargs.get("temperature", self._text_temp)
        text_top_k = kwargs.get("top_k", self._text_top_k)

        lm_gen = self._lm_gen_class(
            model=self._lm,
            max_steps=4096,
            text_sampler=self._sampler_class(temp=text_temp, top_k=text_top_k),
            audio_sampler=self._sampler_class(temp=0.8, top_k=250),
            cfg_coef=1.0,
            check=False,
        )

        return StreamingMoshiSession(
            _model=self,
            _lm_gen=lm_gen,
        )

    def feed_audio(
        self,
        session: StreamingMoshiSession,
        pcm_data: Union[bytes, np.ndarray],
        **kwargs,
    ) -> List[str]:
        """Feed audio chunk to the session.

        Args:
            session: Streaming session from create_streaming_session().
            pcm_data: Audio data - either bytes (int16 PCM) or numpy array (float32).

        Returns:
            List of text deltas produced from this chunk.
        """
        if session.finished:
            return []

        # Convert bytes to float32
        if isinstance(pcm_data, bytes):
            # Assume int16 PCM at 24kHz
            int16_array = np.frombuffer(pcm_data, dtype=np.int16)
            float_data = int16_array.astype(np.float32) / 32768.0
        else:
            float_data = pcm_data.astype(np.float32)

        # Buffer audio
        session.audio_buffer = np.concatenate([session.audio_buffer, float_data])

        deltas = []

        # Process complete frames
        while len(session.audio_buffer) >= FRAME_SIZE:
            frame = session.audio_buffer[:FRAME_SIZE]
            session.audio_buffer = session.audio_buffer[FRAME_SIZE:]

            # Encode frame with Mimi
            pcm_tensor = mx.array(frame[np.newaxis, np.newaxis, :])  # (1, 1, 1920)
            codes = self._mimi.encode_step(pcm_tensor)  # (1, codebooks, 1)
            mx.eval(codes)

            # Get other codebooks
            other_codes = codes[:, : self._other_codebooks, 0]  # (1, other_cb)

            # Run LM step
            text_token, _ = session._lm_gen.step(
                other_codes[0],
                ct=self._condition_tensor,
                depformer_replace_tokens=self._silence_replace,
            )
            mx.eval(text_token)

            text_token_val = text_token[0].item()

            if text_token_val not in (0, 3):
                session.text_tokens.append(int(text_token_val))

            session.step_idx += 1

            # Compute delta
            current_text = self._tokenizer.decode(session.text_tokens) if session.text_tokens else ""
            if current_text != session.prev_text:
                delta = current_text[len(session.prev_text):]
                if delta:
                    deltas.append(delta)
                session.prev_text = current_text

        return deltas

    def finish_session(
        self,
        session: StreamingMoshiSession,
        **kwargs,
    ) -> str:
        """Flush remaining audio and finalize the session.

        Args:
            session: Streaming session to finalize.

        Returns:
            Final complete transcription text.
        """
        # Pad and process any remaining buffered audio
        if len(session.audio_buffer) > 0:
            remainder = len(session.audio_buffer)
            padding = FRAME_SIZE - remainder
            padded = np.concatenate([
                session.audio_buffer,
                np.zeros(padding, dtype=np.float32)
            ])

            # Encode final frame
            pcm_tensor = mx.array(padded[np.newaxis, np.newaxis, :])
            codes = self._mimi.encode_step(pcm_tensor)
            mx.eval(codes)

            other_codes = codes[:, : self._other_codebooks, 0]

            text_token, _ = session._lm_gen.step(
                other_codes[0],
                ct=self._condition_tensor,
                depformer_replace_tokens=self._silence_replace,
            )
            mx.eval(text_token)

            text_token_val = text_token[0].item()
            if text_token_val not in (0, 3):
                session.text_tokens.append(int(text_token_val))

            session.audio_buffer = np.array([], dtype=np.float32)

        session.finished = True

        # Decode final text
        if session.text_tokens:
            return self._tokenizer.decode(session.text_tokens).strip()
        return ""

    def reset_session(self, session: StreamingMoshiSession) -> None:
        """Reset session state for a new utterance.

        Args:
            session: Session to reset.
        """
        # Reset Mimi encoder state
        self._mimi.reset_all()

        # Reset transformer caches
        for c in self._lm.transformer_cache:
            c.reset()

        # Reset session state
        session.reset()

        # Create new LmGen
        session._lm_gen = self._lm_gen_class(
            model=self._lm,
            max_steps=4096,
            text_sampler=self._sampler_class(temp=self._text_temp, top_k=self._text_top_k),
            audio_sampler=self._sampler_class(temp=0.8, top_k=250),
            cfg_coef=1.0,
            check=False,
        )
