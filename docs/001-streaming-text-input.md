# Streaming Text Input for TTS

**Status**: Implemented
**Target Model**: Qwen3-TTS
**Date**: 2025-01-25

## Overview

This document describes the streaming text input feature for Qwen3-TTS, which enables real-time text-to-speech synthesis from LLM token streams with seamless audio transitions.

## Problem Statement

When integrating TTS with streaming LLM outputs, we want to:
1. **Start speaking immediately** - Don't wait for the complete LLM response
2. **Maintain voice consistency** - Same voice/prosody across all chunks
3. **Eliminate audio artifacts** - No clicks, pops, or unnatural breaks

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         StreamingContext                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Sentence   │    │  ICL-Style   │    │   Audio Crossfade     │  │
│  │   Buffer    │───▶│  Generation  │───▶│   & Stitching         │  │
│  │             │    │  (context)   │    │                       │  │
│  └─────────────┘    └──────────────┘    └───────────────────────┘  │
│         │                  │                       │                │
│         ▼                  ▼                       ▼                │
│   Token Stream      Previous Codes          Seamless Audio         │
│   → Sentences       as "Reference"          Output Stream          │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Sentence Buffer
- Accumulates tokens until sentence boundaries detected
- Handles abbreviations (Dr., Mr., etc.) and decimals (3.14)
- Minimum length threshold to avoid tiny utterances

#### 2. ICL-Style Context Conditioning
- Uses previously generated codes as "reference" for next chunk
- Model "hears" what it just said → prosody continuity
- Speaker embedding computed once and reused

#### 3. Audio Crossfading
- Equal-power crossfade (cos²/sin²) at chunk boundaries
- Eliminates any discontinuities or clicks
- Configurable crossfade duration (default: 80ms)

## Implementation

### Files Added/Modified

```
mlx_audio/tts/models/qwen3_tts/
├── streaming.py          # NEW: Streaming implementation
├── __init__.py           # MODIFIED: Export streaming classes
└── qwen3_tts.py          # Unchanged (streaming uses existing methods)

examples/
└── streaming_tts_example.py  # NEW: Usage examples
```

### Core Classes

#### `StreamingContext`
Main class for managing streaming state:

```python
from mlx_audio.tts.models.qwen3_tts.streaming import StreamingContext, StreamingConfig

config = StreamingConfig(
    context_codes=50,       # ~4 seconds of context for ICL
    crossfade_ms=80.0,      # 80ms crossfade
    temperature=0.9,
    verbose=True,
)

ctx = StreamingContext(
    model=model,
    voice="Aiden",
    language="English",
    config=config,
)

# Feed tokens from LLM
for token in llm_stream:
    for audio_chunk in ctx.add_text(token):
        play(audio_chunk)

# Get any remaining audio
for audio_chunk in ctx.finalize():
    play(audio_chunk)
```

#### `StreamingConfig`
Configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `context_codes` | 50 | Codec frames for ICL context (~4s at 12.5Hz) |
| `crossfade_ms` | 80.0 | Crossfade duration in milliseconds |
| `min_sentence_length` | 10 | Min chars before allowing sentence break |
| `temperature` | 0.9 | Sampling temperature |
| `top_k` | 50 | Top-k sampling |
| `top_p` | 1.0 | Nucleus sampling threshold |
| `repetition_penalty` | 1.2 | Slightly higher for streaming stability |
| `max_tokens_per_chunk` | 2048 | Max tokens per text chunk |

### High-Level API

```python
from mlx_audio.tts.models.qwen3_tts.streaming import stream_from_text_iterator, stream_text

# From iterator (LLM stream)
for audio in stream_from_text_iterator(model, llm_token_stream, voice="Aiden"):
    play(audio)

# From complete text
for audio in stream_text(model, "Hello world. How are you?", voice="Aiden"):
    play(audio)
```

## How It Works

### Chunk Generation Flow

```
Chunk 1 (first):
├── Standard generation (or ICL with user's ref_audio)
├── Save last 50 codes as context
├── Save audio tail for crossfade
└── Yield audio (with fade-in)

Chunk 2+ (subsequent):
├── Prepare ICL inputs with prev_codes as "reference"
├── Generate conditioned on context
├── Decode: [prev_codes | new_codes] → audio
├── Trim reference portion from audio
├── Crossfade with prev_audio_tail
├── Update context (last 50 codes)
└── Yield audio

Final chunk:
├── Same as above
├── Apply fade-out
└── Yield final audio
```

### ICL Conditioning Details

The key insight is reusing the existing ICL (In-Context Learning) mechanism:

1. **Normal ICL**: Encode reference audio → use codes as context
2. **Streaming ICL**: Use previously generated codes directly as context

This gives us:
- **Prosody continuity**: Model conditions on what it just said
- **Voice consistency**: Same speaker characteristics maintained
- **Natural transitions**: Intonation flows across sentences

### Audio Crossfade

Equal-power crossfade ensures constant energy:

```
Audio 1: [.......AAAA]
                └─80ms─┐
Audio 2:         [BBBB.......]

Output:  [.......XXXX.......]
         where XXXX = A*cos²(t) + B*sin²(t)
```

This eliminates:
- Clicks from amplitude discontinuities
- Phase misalignment artifacts
- Abrupt timbre changes

## Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| Latency to first audio | ~sentence generation time | Depends on sentence length |
| Real-time factor | 2-4x | Generates faster than playback |
| Memory overhead | O(context_codes) | Bounded, doesn't grow |
| Context window | ~4 seconds | Configurable via `context_codes` |

## Usage Examples

### Basic LLM Integration

```python
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.qwen3_tts.streaming import StreamingContext

model = load_model("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
ctx = StreamingContext(model, voice="Aiden", language="English")

async for token in llm.stream_generate("Tell me a story"):
    for audio in ctx.add_text(token):
        await audio_player.play(audio)

for audio in ctx.finalize():
    await audio_player.play(audio)
```

### With Voice Cloning

```python
ref_audio = load_audio("my_voice.wav")

ctx = StreamingContext(
    model=model,
    ref_audio=ref_audio,
    ref_text="This is what my voice sounds like.",
    language="English",
)

# Now all generated audio will match the reference voice
for token in llm_stream:
    for audio in ctx.add_text(token):
        play(audio)
```

### Multi-Turn Conversation

```python
ctx = StreamingContext(model, voice="Aiden")

# Turn 1
for audio in ctx.add_text("Hello! How can I help?"):
    play(audio)
for audio in ctx.finalize():
    play(audio)

# Turn 2 - context maintained for prosody continuity
for audio in ctx.add_text("That's a great question!"):
    play(audio)
for audio in ctx.finalize():
    play(audio)

# Reset for new conversation
ctx.reset()
```

## Limitations

1. **Encoder Required for ICL**: Full context conditioning requires the speech tokenizer encoder. Without it, each chunk is generated independently.

2. **Sentence-Level Granularity**: Currently buffers until sentence boundaries. Sub-sentence streaming would require speculative generation.

3. **No Interruption Handling**: If user interrupts, need to call `ctx.reset()` and start fresh.

## Future Improvements

- [ ] Sub-sentence streaming with speculative prefill
- [ ] Interruption/barge-in support
- [ ] Adaptive crossfade duration based on audio content
- [ ] Pitch/energy normalization at boundaries
- [ ] Parallel chunk generation (pipeline next while playing current)
