# Copyright (c) 2025, Prince Canuma and contributors (https://github.com/Blaizzy/mlx-audio)
# Streaming text input support for Qwen3-TTS

"""
Streaming TTS module for Qwen3-TTS.

This module provides support for streaming text input from LLMs, enabling
real-time text-to-speech synthesis with seamless audio transitions.

Key features:
- ICL-style conditioning: Each chunk is conditioned on previous audio for prosody continuity
- Audio crossfading: Eliminates clicks/pops at chunk boundaries
- Sentence-level chunking: Intelligent text buffering for natural speech
- Memory efficient: Bounded context window, no unbounded growth

Example usage:
    from mlx_audio.tts.utils import load_model
    from mlx_audio.tts.models.qwen3_tts.streaming import StreamingContext

    model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    ctx = StreamingContext(model, voice="Aiden", language="English")

    # From LLM stream
    for sentence in sentence_stream:
        for audio_chunk in ctx.add_text(sentence):
            play_audio(audio_chunk)

    # Finalize any remaining audio
    for audio_chunk in ctx.finalize():
        play_audio(audio_chunk)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from mlx_audio.tts.models.base import GenerationResult


# =============================================================================
# Audio Utilities
# =============================================================================


def crossfade(
    audio1: mx.array,
    audio2: mx.array,
    crossfade_samples: int,
) -> mx.array:
    """Apply equal-power crossfade between two audio segments.

    Uses cos²/sin² crossfade for energy preservation:
    - fade_out = cos²(t * π/2)
    - fade_in = sin²(t * π/2)

    This ensures: fade_out² + fade_in² = 1 (constant power)

    Args:
        audio1: First audio segment (will use tail for crossfade)
        audio2: Second audio segment (will use head for crossfade)
        crossfade_samples: Number of samples for crossfade region

    Returns:
        Merged audio with crossfade applied
    """
    if crossfade_samples <= 0:
        return mx.concatenate([audio1, audio2])

    # Ensure we have enough samples
    crossfade_samples = min(crossfade_samples, audio1.shape[0], audio2.shape[0])

    if crossfade_samples == 0:
        return mx.concatenate([audio1, audio2])

    # Split audio into regions
    audio1_head = audio1[:-crossfade_samples]
    audio1_tail = audio1[-crossfade_samples:]
    audio2_head = audio2[:crossfade_samples]
    audio2_tail = audio2[crossfade_samples:]

    # Create crossfade curves (equal power: cos²/sin²)
    t = mx.linspace(0, 1, crossfade_samples)
    fade_out = mx.cos(t * (mx.pi / 2)) ** 2
    fade_in = mx.sin(t * (mx.pi / 2)) ** 2

    # Apply crossfade
    crossfaded = audio1_tail * fade_out + audio2_head * fade_in

    # Concatenate all parts
    return mx.concatenate([audio1_head, crossfaded, audio2_tail])


def apply_fade_in(audio: mx.array, fade_samples: int) -> mx.array:
    """Apply fade-in to the beginning of audio."""
    if fade_samples <= 0 or audio.shape[0] == 0:
        return audio

    fade_samples = min(fade_samples, audio.shape[0])
    t = mx.linspace(0, 1, fade_samples)
    fade = mx.sin(t * (mx.pi / 2)) ** 2

    audio_head = audio[:fade_samples] * fade
    audio_tail = audio[fade_samples:]
    return mx.concatenate([audio_head, audio_tail])


def apply_fade_out(audio: mx.array, fade_samples: int) -> mx.array:
    """Apply fade-out to the end of audio."""
    if fade_samples <= 0 or audio.shape[0] == 0:
        return audio

    fade_samples = min(fade_samples, audio.shape[0])
    t = mx.linspace(0, 1, fade_samples)
    fade = mx.cos(t * (mx.pi / 2)) ** 2

    audio_head = audio[:-fade_samples]
    audio_tail = audio[-fade_samples:] * fade
    return mx.concatenate([audio_head, audio_tail])


# =============================================================================
# Sentence Detection
# =============================================================================

# Common abbreviations that shouldn't trigger sentence breaks
ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "inc",
    "ltd",
    "co",
    "corp",
    "st",
    "ave",
    "blvd",
    "e.g",
    "i.e",
    "cf",
    "al",
    "fig",
    "vol",
    "no",
    "pp",
    "ed",
    "rev",
    "gen",
    "col",
    "lt",
    "sgt",
    "capt",
    "maj",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
}


def is_sentence_boundary(text: str, pos: int) -> bool:
    """Check if position is a valid sentence boundary.

    Args:
        text: The full text
        pos: Position of the punctuation mark

    Returns:
        True if this is a valid sentence boundary
    """
    if pos >= len(text) - 1:
        return True  # End of text

    punct = text[pos]
    if punct not in ".!?":
        return False

    # Check what comes after
    after = text[pos + 1 :]
    if not after:
        return True

    # Must be followed by space, newline, or quote+space
    first_after = after[0]
    if first_after not in " \n\t\"'')]":
        return False

    # Check for abbreviations (only for period)
    if punct == ".":
        # Find the word before the period
        before = text[:pos]
        word_match = re.search(r"(\w+)$", before)
        if word_match:
            word = word_match.group(1).lower()
            if word in ABBREVIATIONS:
                return False

        # Check for decimal numbers (e.g., "3.14")
        if re.search(r"\d$", before) and re.match(r"\d", after.lstrip()):
            return False

        # Check for ellipsis
        if after.startswith(".."):
            return False

    return True


def split_into_sentences(text: str, min_length: int = 10) -> List[Tuple[str, bool]]:
    """Split text into sentences with metadata about completeness.

    Args:
        text: Text to split
        min_length: Minimum sentence length before allowing a split

    Returns:
        List of (sentence, is_complete) tuples
    """
    sentences = []
    current = ""
    i = 0

    while i < len(text):
        char = text[i]
        current += char

        if char in ".!?" and len(current.strip()) >= min_length:
            if is_sentence_boundary(text, i):
                # Include any trailing quotes or spaces
                while i + 1 < len(text) and text[i + 1] in "\"'') \t":
                    i += 1
                    current += text[i]

                sentences.append((current.strip(), True))
                current = ""
        i += 1

    # Handle remaining text
    if current.strip():
        sentences.append((current.strip(), False))

    return sentences


class SentenceBuffer:
    """Buffer for accumulating text and extracting complete sentences."""

    def __init__(self, min_sentence_length: int = 10):
        self.buffer = ""
        self.min_length = min_sentence_length

    def add(self, text: str) -> List[str]:
        """Add text to buffer and return any complete sentences.

        Args:
            text: Text to add

        Returns:
            List of complete sentences (may be empty)
        """
        self.buffer += text
        sentences = []

        parts = split_into_sentences(self.buffer, self.min_length)

        if not parts:
            return sentences

        # All complete sentences go to output
        for sentence, is_complete in parts[:-1]:
            if is_complete:
                sentences.append(sentence)

        # Last part stays in buffer (complete or not)
        last_sentence, last_complete = parts[-1]
        if last_complete and len(parts) > 1:
            sentences.append(last_sentence)
            self.buffer = ""
        elif last_complete:
            sentences.append(last_sentence)
            self.buffer = ""
        else:
            self.buffer = last_sentence

        return sentences

    def flush(self) -> Optional[str]:
        """Flush any remaining text from buffer.

        Returns:
            Remaining text or None if empty
        """
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None


# =============================================================================
# Streaming Context
# =============================================================================


@dataclass
class StreamingConfig:
    """Configuration for streaming TTS."""

    # Context preservation
    context_codes: int = 50
    """Number of codec frames to use as context (50 = ~4s at 12.5Hz)"""

    # Audio crossfade
    crossfade_ms: float = 80.0
    """Crossfade duration in milliseconds for seamless transitions"""

    # Sentence detection
    min_sentence_length: int = 10
    """Minimum characters before allowing sentence break"""

    # Generation parameters
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    """Slightly higher than default for better streaming stability"""

    max_tokens_per_chunk: int = 2048
    """Maximum tokens to generate per text chunk"""

    # Performance
    verbose: bool = False
    """Print debug information"""


@dataclass
class StreamingContext:
    """Manages state for streaming text-to-speech synthesis.

    This class handles:
    - Text buffering and sentence detection
    - Speaker embedding caching
    - ICL-style context conditioning (previous codes as reference)
    - Audio crossfading for seamless transitions

    The streaming approach uses the generated audio from previous chunks
    as "reference audio" for the next chunk, giving us:
    - Prosody continuity (model "hears" what it just said)
    - Voice consistency across chunks
    - Natural-sounding transitions

    Example:
        ctx = StreamingContext(model, voice="Aiden", language="English")

        for token in llm_stream:
            for audio in ctx.add_text(token):
                play(audio)

        for audio in ctx.finalize():
            play(audio)
    """

    model: "Model"  # Forward reference to avoid circular import
    voice: Optional[str] = None
    language: str = "auto"
    ref_audio: Optional[mx.array] = None
    ref_text: Optional[str] = None
    config: StreamingConfig = field(default_factory=StreamingConfig)

    # Internal state (initialized in __post_init__)
    _speaker_embed: Optional[mx.array] = field(default=None, repr=False)
    _prev_codes: Optional[mx.array] = field(default=None, repr=False)
    _prev_text: Optional[str] = field(default=None, repr=False)
    _prev_audio_tail: Optional[mx.array] = field(default=None, repr=False)
    _sentence_buffer: SentenceBuffer = field(default=None, repr=False)
    _chunk_idx: int = field(default=0, repr=False)
    _is_first_chunk: bool = field(default=True, repr=False)
    _crossfade_samples: int = field(default=0, repr=False)

    def __post_init__(self):
        """Initialize internal state."""
        self._sentence_buffer = SentenceBuffer(self.config.min_sentence_length)
        self._crossfade_samples = int(
            self.config.crossfade_ms * self.model.sample_rate / 1000
        )

        # Pre-compute speaker embedding if we have reference audio
        if self.ref_audio is not None and self.model.speaker_encoder is not None:
            self._speaker_embed = self.model.extract_speaker_embedding(self.ref_audio)
            mx.eval(self._speaker_embed)

        # If we have reference audio and text, encode the reference codes
        if (
            self.ref_audio is not None
            and self.ref_text is not None
            and self.model.speech_tokenizer is not None
            and self.model.speech_tokenizer.has_encoder
        ):
            ref_audio = self.ref_audio
            if ref_audio.ndim == 1:
                ref_audio = ref_audio[None, None, :]
            elif ref_audio.ndim == 2:
                ref_audio = ref_audio[None, :]
            self._prev_codes = self.model.speech_tokenizer.encode(ref_audio)
            self._prev_text = self.ref_text
            mx.eval(self._prev_codes)

    def add_text(self, text: str) -> Generator[mx.array, None, None]:
        """Add text and yield any generated audio chunks.

        This method buffers text until complete sentences are detected,
        then generates audio for each sentence with context conditioning.

        Args:
            text: Text to add (can be a single token or multiple words)

        Yields:
            Audio chunks (mx.array) ready for playback
        """
        sentences = self._sentence_buffer.add(text)

        for sentence in sentences:
            yield from self._generate_chunk(sentence)

    def finalize(self) -> Generator[mx.array, None, None]:
        """Finalize streaming and yield any remaining audio.

        Call this after all text has been added to ensure any
        buffered text is processed.

        Yields:
            Final audio chunks
        """
        remaining = self._sentence_buffer.flush()
        if remaining:
            yield from self._generate_chunk(remaining, is_final=True)

    def reset(self):
        """Reset context for a new conversation.

        This clears all accumulated context while preserving the
        initial speaker embedding and reference if provided.
        """
        self._sentence_buffer = SentenceBuffer(self.config.min_sentence_length)
        self._chunk_idx = 0
        self._is_first_chunk = True
        self._prev_audio_tail = None

        # Reset to initial reference if provided
        if (
            self.ref_audio is not None
            and self.ref_text is not None
            and self.model.speech_tokenizer is not None
            and self.model.speech_tokenizer.has_encoder
        ):
            ref_audio = self.ref_audio
            if ref_audio.ndim == 1:
                ref_audio = ref_audio[None, None, :]
            elif ref_audio.ndim == 2:
                ref_audio = ref_audio[None, :]
            self._prev_codes = self.model.speech_tokenizer.encode(ref_audio)
            self._prev_text = self.ref_text
        else:
            self._prev_codes = None
            self._prev_text = None

    def _generate_chunk(
        self,
        text: str,
        is_final: bool = False,
    ) -> Generator[mx.array, None, None]:
        """Generate audio for a single text chunk with context conditioning.

        Args:
            text: Text to synthesize
            is_final: Whether this is the final chunk

        Yields:
            Audio chunks
        """
        if not text.strip():
            return

        start_time = time.time()

        if self.config.verbose:
            print(f"[Chunk {self._chunk_idx}] Generating: {text[:50]}...")

        # Determine generation mode
        use_icl = (
            self._prev_codes is not None
            and self._prev_text is not None
            and self.model.speech_tokenizer is not None
            and self.model.speech_tokenizer.has_encoder
        )

        if use_icl:
            # Generate with ICL conditioning on previous codes
            audio, new_codes = self._generate_with_context(text)
        else:
            # Standard generation (first chunk or no encoder)
            audio, new_codes = self._generate_standard(text)

        if audio is None or audio.shape[0] == 0:
            if self.config.verbose:
                print(f"[Chunk {self._chunk_idx}] No audio generated")
            return

        # Apply crossfade with previous chunk
        if self._prev_audio_tail is not None and not self._is_first_chunk:
            audio = crossfade(
                self._prev_audio_tail,
                audio,
                self._crossfade_samples,
            )

        # For non-final chunks, hold back the tail for crossfading
        if not is_final and audio.shape[0] > self._crossfade_samples * 2:
            output_audio = audio[: -self._crossfade_samples]
            self._prev_audio_tail = audio[-self._crossfade_samples * 2 :]
        else:
            output_audio = audio
            if is_final:
                # Apply fade out on final chunk
                output_audio = apply_fade_out(output_audio, self._crossfade_samples)
            self._prev_audio_tail = audio[-self._crossfade_samples * 2 :] if not is_final else None

        # Update context for next chunk
        if new_codes is not None:
            # Keep last N codes as context
            context_len = min(self.config.context_codes, new_codes.shape[2])
            self._prev_codes = new_codes[:, :, -context_len:]
            self._prev_text = text

        self._is_first_chunk = False
        self._chunk_idx += 1

        if self.config.verbose:
            elapsed = time.time() - start_time
            duration = output_audio.shape[0] / self.model.sample_rate
            rtf = duration / elapsed if elapsed > 0 else 0
            print(
                f"[Chunk {self._chunk_idx - 1}] "
                f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF: {rtf:.2f}x)"
            )

        mx.eval(output_audio)
        yield output_audio

    def _generate_standard(
        self,
        text: str,
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Generate audio using standard (non-ICL) mode.

        Returns:
            Tuple of (audio, codes) or (None, None) if generation failed
        """
        model = self.model
        config = model.config.talker_config

        # Prepare inputs
        input_embeds, trailing_text_hidden, tts_pad_embed = (
            model._prepare_generation_inputs(
                text=text,
                language=self.language,
                speaker=self.voice,
                ref_audio=self.ref_audio,
                ref_text=self.ref_text,
            )
        )

        # Generate codes
        cache = model.talker.make_cache()
        generated_codes = []
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != eos_token_id
        ]
        trailing_idx = 0

        for step in range(self.config.max_tokens_per_chunk):
            logits, hidden = model.talker(input_embeds, cache=cache)

            next_token = model._sample_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                generated_tokens=(
                    [int(c[0, 0]) for c in generated_codes] if generated_codes else None
                ),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )

            if int(next_token[0, 0]) == eos_token_id:
                break

            # Generate remaining codebook tokens
            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]
            code_cache = model.talker.code_predictor.make_cache()

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = model.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = model.talker.code_predictor.codec_embedding[
                        code_idx - 1
                    ](code_tokens[-1])
                    code_input = code_embed

                code_logits, code_cache, _ = model.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )

                next_code = model._sample_token(
                    code_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
                code_tokens.append(next_code)

            all_codes = mx.concatenate(code_tokens, axis=1)
            generated_codes.append(all_codes)

            # Prepare next input
            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = model.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = (
                    codec_embed + model.talker.code_predictor.codec_embedding[i](code)
                )

            input_embeds = text_embed + codec_embed
            mx.eval(input_embeds)

        if not generated_codes:
            return None, None

        # Stack codes: [1, seq_len, num_code_groups]
        codes = mx.stack(generated_codes, axis=1)

        # Decode to audio
        audio, audio_lengths = model.speech_tokenizer.decode(codes)
        audio = audio[0]

        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        # Return codes in [1, num_code_groups, seq_len] format for ICL
        codes_for_context = mx.transpose(codes, (0, 2, 1))

        return audio, codes_for_context

    def _generate_with_context(
        self,
        text: str,
    ) -> Tuple[Optional[mx.array], Optional[mx.array]]:
        """Generate audio using ICL conditioning on previous codes.

        This method conditions the generation on previously generated codes,
        enabling prosody continuity across chunks.

        Returns:
            Tuple of (audio, codes) or (None, None) if generation failed
        """
        model = self.model
        config = model.config.talker_config

        # Prepare ICL inputs using previous codes as reference
        input_embeds, trailing_text_hidden, tts_pad_embed = (
            self._prepare_streaming_icl_inputs(
                text=text,
                ref_codes=self._prev_codes,
                ref_text=self._prev_text,
                language=self.language,
            )
        )

        # Generate codes
        cache = model.talker.make_cache()
        generated_codes = []
        eos_token_id = config.codec_eos_token_id
        suppress_tokens = [
            i
            for i in range(config.vocab_size - 1024, config.vocab_size)
            if i != eos_token_id
        ]
        trailing_idx = 0

        # Cap max tokens based on text length
        target_token_count = len(model.tokenizer.encode(text))
        effective_max_tokens = min(
            self.config.max_tokens_per_chunk, max(75, target_token_count * 6)
        )

        for step in range(effective_max_tokens):
            logits, hidden = model.talker(input_embeds, cache=cache)

            next_token = model._sample_token(
                logits,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                generated_tokens=(
                    [int(c[0, 0]) for c in generated_codes] if generated_codes else None
                ),
                suppress_tokens=suppress_tokens,
                eos_token_id=eos_token_id,
            )

            if int(next_token[0, 0]) == eos_token_id:
                break

            # Generate remaining codebook tokens
            code_tokens = [next_token]
            code_hidden = hidden[:, -1:, :]
            code_cache = model.talker.code_predictor.make_cache()

            for code_idx in range(config.num_code_groups - 1):
                if code_idx == 0:
                    code_0_embed = model.talker.get_input_embeddings()(next_token)
                    code_input = mx.concatenate([code_hidden, code_0_embed], axis=1)
                else:
                    code_embed = model.talker.code_predictor.codec_embedding[
                        code_idx - 1
                    ](code_tokens[-1])
                    code_input = code_embed

                code_logits, code_cache, _ = model.talker.code_predictor(
                    code_input,
                    cache=code_cache,
                    generation_step=code_idx,
                )

                next_code = model._sample_token(
                    code_logits,
                    temperature=self.config.temperature,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                )
                code_tokens.append(next_code)

            all_codes = mx.concatenate(code_tokens, axis=1)
            generated_codes.append(all_codes)

            # Prepare next input
            if trailing_idx < trailing_text_hidden.shape[1]:
                text_embed = trailing_text_hidden[:, trailing_idx : trailing_idx + 1, :]
                trailing_idx += 1
            else:
                text_embed = tts_pad_embed

            codec_embed = model.talker.get_input_embeddings()(next_token)
            for i, code in enumerate(code_tokens[1:]):
                codec_embed = (
                    codec_embed + model.talker.code_predictor.codec_embedding[i](code)
                )

            input_embeds = text_embed + codec_embed
            mx.eval(input_embeds)

        if not generated_codes:
            return None, None

        # Stack generated codes: [1, gen_len, num_code_groups]
        gen_codes = mx.stack(generated_codes, axis=1)

        # Prepend reference codes for decoding (ICL style)
        # ref_codes: [1, 16, ref_time] -> [1, ref_time, 16]
        ref_codes_t = mx.transpose(self._prev_codes, (0, 2, 1))
        full_codes = mx.concatenate([ref_codes_t, gen_codes], axis=1)

        ref_len = self._prev_codes.shape[2]
        total_len = full_codes.shape[1]

        # Decode combined codes
        audio, audio_lengths = model.speech_tokenizer.decode(full_codes)
        audio = audio[0]

        valid_len = int(audio_lengths[0])
        if valid_len > 0 and valid_len < audio.shape[0]:
            audio = audio[:valid_len]

        # Trim reference portion (proportional trimming)
        cut = int(ref_len / max(total_len, 1) * audio.shape[0])
        if cut > 0 and cut < audio.shape[0]:
            audio = audio[cut:]

        # Return new codes for next chunk context
        # gen_codes is [1, gen_len, 16], transpose to [1, 16, gen_len]
        codes_for_context = mx.transpose(gen_codes, (0, 2, 1))

        return audio, codes_for_context

    def _prepare_streaming_icl_inputs(
        self,
        text: str,
        ref_codes: mx.array,
        ref_text: str,
        language: str = "auto",
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Prepare ICL inputs using pre-encoded reference codes.

        This is a modified version of _prepare_icl_generation_inputs that
        accepts pre-encoded codes instead of raw audio, enabling efficient
        streaming without re-encoding.

        Args:
            text: Target text to synthesize
            ref_codes: Pre-encoded reference codes [1, num_quantizers, ref_time]
            ref_text: Transcript of the reference
            language: Language code

        Returns:
            input_embeds, trailing_text_hidden, tts_pad_embed
        """
        model = self.model
        config = model.config.talker_config

        ref_time = ref_codes.shape[2]

        # Tokenize ref_text and target_text
        ref_chat = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_ids = mx.array(model.tokenizer.encode(ref_chat))[None, :]
        ref_text_ids = ref_ids[:, 3:-2]

        target_chat = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        target_ids = mx.array(model.tokenizer.encode(target_chat))[None, :]
        text_ids = target_ids[:, 3:-5]

        # TTS special tokens
        tts_tokens = mx.array(
            [
                [
                    model.config.tts_bos_token_id,
                    model.config.tts_eos_token_id,
                    model.config.tts_pad_token_id,
                ]
            ]
        )
        tts_embeds = model.talker.text_projection(
            model.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos_embed = tts_embeds[:, 0:1, :]
        tts_eos_embed = tts_embeds[:, 1:2, :]
        tts_pad_embed = tts_embeds[:, 2:3, :]

        # Build text embeddings
        combined_text_ids = mx.concatenate([ref_text_ids, text_ids], axis=1)
        text_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(combined_text_ids)
        )
        text_embed = mx.concatenate([text_embed, tts_eos_embed], axis=1)
        text_lens = text_embed.shape[1]

        # Build codec embeddings from pre-encoded codes
        first_cb_codes = ref_codes[:, 0, :]
        ref_codec_embed = model.talker.get_input_embeddings()(first_cb_codes)
        for i in range(config.num_code_groups - 1):
            cb_codes = ref_codes[:, i + 1, :]
            ref_codec_embed = (
                ref_codec_embed + model.talker.code_predictor.codec_embedding[i](cb_codes)
            )

        codec_bos_embed = model.talker.get_input_embeddings()(
            mx.array([[config.codec_bos_id]])
        )
        codec_embed_icl = mx.concatenate([codec_bos_embed, ref_codec_embed], axis=1)
        codec_lens = codec_embed_icl.shape[1]

        # Non-streaming overlay
        codec_pad_embed = model.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id]])
        )
        text_with_codec_pad = text_embed + mx.broadcast_to(
            codec_pad_embed, (1, text_lens, codec_pad_embed.shape[-1])
        )
        codec_with_text_pad = codec_embed_icl + mx.broadcast_to(
            tts_pad_embed, (1, codec_lens, tts_pad_embed.shape[-1])
        )
        icl_input_embed = mx.concatenate(
            [text_with_codec_pad, codec_with_text_pad], axis=1
        )
        trailing_text_hidden = tts_pad_embed

        # Language ID
        language_id = None
        if language.lower() != "auto" and config.codec_language_id:
            if language.lower() in config.codec_language_id:
                language_id = config.codec_language_id[language.lower()]

        # Codec prefix
        if language_id is None:
            codec_prefill = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_prefill = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]

        codec_prefix_embed = model.talker.get_input_embeddings()(
            mx.array([codec_prefill])
        )
        codec_prefix_suffix = model.talker.get_input_embeddings()(
            mx.array([[config.codec_pad_id, config.codec_bos_id]])
        )

        # Add speaker embedding if available
        if self._speaker_embed is not None:
            codec_prefix_embed = mx.concatenate(
                [
                    codec_prefix_embed,
                    self._speaker_embed.reshape(1, 1, -1),
                    codec_prefix_suffix,
                ],
                axis=1,
            )
        else:
            codec_prefix_embed = mx.concatenate(
                [codec_prefix_embed, codec_prefix_suffix], axis=1
            )

        # Role embedding
        role_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(target_ids[:, :3])
        )

        # Build prefix
        pad_count = codec_prefix_embed.shape[1] - 2
        pad_embeds = mx.broadcast_to(
            tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1])
        )
        combined_prefix = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
        combined_prefix = combined_prefix + codec_prefix_embed[:, :-1, :]

        # Full input embeddings
        input_embeds = mx.concatenate(
            [role_embed, combined_prefix, icl_input_embed], axis=1
        )

        return input_embeds, trailing_text_hidden, tts_pad_embed


# =============================================================================
# High-Level API
# =============================================================================


def stream_from_text_iterator(
    model: "Model",
    text_iterator: Iterator[str],
    voice: Optional[str] = None,
    language: str = "auto",
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    config: Optional[StreamingConfig] = None,
) -> Generator[mx.array, None, None]:
    """Stream TTS from a text iterator (e.g., LLM token stream).

    This is the main entry point for streaming TTS. It handles:
    - Text buffering and sentence detection
    - Audio generation with context conditioning
    - Seamless audio crossfading

    Args:
        model: Loaded Qwen3-TTS model
        text_iterator: Iterator yielding text chunks (tokens, words, etc.)
        voice: Speaker name (optional)
        language: Language code
        ref_audio: Reference audio for voice cloning (optional)
        ref_text: Transcript of reference audio (required if ref_audio provided)
        config: Streaming configuration

    Yields:
        Audio chunks ready for playback

    Example:
        # With LLM
        async def llm_stream():
            async for token in llm.generate("Tell me a story"):
                yield token

        for audio in stream_from_text_iterator(model, llm_stream()):
            play_audio(audio)
    """
    if config is None:
        config = StreamingConfig()

    ctx = StreamingContext(
        model=model,
        voice=voice,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        config=config,
    )

    for text_chunk in text_iterator:
        yield from ctx.add_text(text_chunk)

    yield from ctx.finalize()


def stream_text(
    model: "Model",
    text: str,
    voice: Optional[str] = None,
    language: str = "auto",
    ref_audio: Optional[mx.array] = None,
    ref_text: Optional[str] = None,
    config: Optional[StreamingConfig] = None,
) -> Generator[mx.array, None, None]:
    """Stream TTS from a complete text string.

    This splits the text into sentences and generates audio for each
    with seamless transitions. Useful for processing pre-existing text
    with streaming output.

    Args:
        model: Loaded Qwen3-TTS model
        text: Complete text to synthesize
        voice: Speaker name (optional)
        language: Language code
        ref_audio: Reference audio for voice cloning (optional)
        ref_text: Transcript of reference audio
        config: Streaming configuration

    Yields:
        Audio chunks ready for playback
    """
    # Split into words and stream as if from LLM
    words = text.split()

    def word_iterator():
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    yield from stream_from_text_iterator(
        model=model,
        text_iterator=word_iterator(),
        voice=voice,
        language=language,
        ref_audio=ref_audio,
        ref_text=ref_text,
        config=config,
    )
