# Plan: Moshi STT/TTS + Streaming Infrastructure Fixes

## Context

The `feat/streaming-stt` branch adds streaming audio input support for 3 STT models (Voxtral Realtime, VibeVoice-ASR, LASR-CTC). This plan adds Moshi STT/TTS models by importing from the MIT-licensed `moshi-mlx` package, and fixes 5 pending issues: session ABC conformance, WebSocket streaming, async API, and docs.

## Overview (7 workstreams)

| # | What | Files | Effort |
|---|------|-------|--------|
| 0 | **Dependency setup** | `pyproject.toml` | Small |
| A | **Moshi STT model** | `mlx_audio/stt/models/moshi/` (3 new files) | Medium |
| B | **Moshi TTS model** | `mlx_audio/tts/models/moshi/` (3 new files) | Medium |
| C | **Session ABC conformance** | 3 existing session files + `base.py` | Small |
| D | **WebSocket streaming mode** | `mlx_audio/server.py` | Medium |
| E | **Async streaming API** | `mlx_audio/stt/async_streaming.py` (new) | Small |
| F | **Commit docs/** | `docs/streaming-analysis.md` | Trivial |

## Implementation order

```
0 (deps) → C (ABC fix) → A (Moshi STT) → B (Moshi TTS) → D (WebSocket) → E (async) → F (docs)
```

---

## 0. Dependency Setup

**Modify: `pyproject.toml`**

Add `moshi` optional dependency group:
```toml
moshi = [
    "moshi-mlx>=0.3.0",
    "sentencepiece>=0.2.0",
]
```
Add `"moshi-mlx>=0.3.0"` to `all` group.

---

## A. Moshi STT Model

### Create: `mlx_audio/stt/models/moshi/__init__.py`
Export `Model`, `StreamingMoshiSTTSession`, `ModelConfig`.

### Create: `mlx_audio/stt/models/moshi/config.py`
Dataclass `ModelConfig` with:
- `model_type = "moshi_stt"`
- `hf_repo`, `moshi_name`, `mimi_name`, `tokenizer_name` (from config.json)
- `text_temp`, `text_top_k` (from `lm_gen_config` in config.json)
- `audio_silence_prefix_seconds`, `audio_delay_seconds` (from `stt_config`)
- `skip_depformer = True`
- `quantized: Optional[int] = None`
- `from_dict(cls, data)` extracts these from Moshi's config.json format

### Create: `mlx_audio/stt/models/moshi/moshi_stt.py`

**`StreamingMoshiSTTSession(StreamingSTTSessionBase)`** — dataclass:
- Fields: `audio_buffer`, `text_tokens`, `prev_text`, `step_idx`, `finished`, `_model` (backreference), `_lm_gen` (LmGen instance)
- `feed_audio(pcm_bytes)` → delegates to `self._model.feed_audio(self, pcm_bytes)`
- `finish()` → delegates to `self._model.finish_session(self)`
- `reset()` → delegates to `self._model.reset_session(self)`

**`Model(nn.Module)`**:
- `__init__(config)` — stores config, sets `_lm`, `_mimi`, `_tokenizer` to None
- `supports_streaming_input()` → `True`

**Batch transcription** — `generate(audio, *, stream, **kwargs)`:
- Load/resample audio to 24kHz float32
- Apply STT padding (audio_silence_prefix_seconds, audio_delay_seconds)
- Pad to frame boundary (FRAME_SIZE=1920)
- Phase 1: `mimi.encode(pcm_tensor)` → all_codes (batch)
- Phase 2: LmGen step loop per frame → text tokens (filter tokens 0, 3)
- If `stream=True`, yield text deltas per-step
- Decode via SentencePiece, return STTOutput

**Streaming API**:
- `create_streaming_session(temperature)`:
  - `mimi.reset_all()`, reset transformer caches
  - Create `LmGen(model, max_steps=4096, text_sampler, audio_sampler)`
  - Return `StreamingMoshiSTTSession(_model=self, _lm_gen=lm_gen)`

- `feed_audio(session, pcm_24k)`:
  - Convert bytes→float32 (int16 / 32768)
  - Buffer until ≥ FRAME_SIZE (1920 samples = 80ms)
  - Per complete frame:
    - `codes = mimi.encode_step(pcm[1,1,1920])` → (1, codebooks, 1)
    - `other_codes = codes[:, :other_codebooks, 0]`
    - `text_token, _ = lm_gen.step(other_codes[0], ct=ct, depformer_replace_tokens=silence_replace)`
    - `mx.eval(text_token)`
    - If token not in {0, 3}: append to `session.text_tokens`
    - Decode accumulated tokens, diff against `session.prev_text` → delta
  - Return list of text deltas

- `finish_session(session)`:
  - Zero-pad remaining buffer to frame boundary, feed it
  - Return `tokenizer.decode(session.text_tokens).strip()`

- `reset_session(session)`:
  - Clear all session fields, `mimi.reset_all()`, reset caches

**Weight loading** — `sanitize(weights)` returns `{}` (skip standard loading), `post_load_hook(model, model_path)`:
- Import from moshi_mlx: `Lm, LmConfig, Mimi, mimi_202407, LmGen, Sampler`
- Import `sentencepiece`, `_lm_config_from_dict` from `moshi_stt_mlx.engine`
- Load config.json, build LmConfig via `_lm_config_from_dict()`
- Load LM: `Lm(lm_config)`, `set_dtype(bfloat16)`, `load_weights(model.safetensors)`
- Load Mimi: `Mimi(mimi_202407(codebooks))`, `load_pytorch_weights(tokenizer-*.safetensors)`
- Warmup both
- Load SentencePiece tokenizer
- Prepare SILENCE_TOKENS replacement array
- Prepare condition tensor if model has condition_provider
- Attach all to model instance

### Modify: `mlx_audio/stt/utils.py`
Add to MODEL_REMAPPING:
```python
"moshi": "moshi",
"moshi_stt": "moshi",
```

### Modify: `mlx_audio/utils.py` — `base_load_model()`
Add config-based model type detection fallback after line 365:
```python
# Detect Moshi models by config signature
if model_type is None or (model_type not in model_remapping and ...):
    if "dep_q" in config and "delays" in config:
        model_type = "moshi"
```
This handles `kyutai/stt-1b-en_fr-mlx` where config.json has no `model_type` field.

### Create: `mlx_audio/tests/test_moshi_stt_streaming.py`
Tests (unit, no weights needed):
- `test_session_creation()` — verify fields initialized correctly
- `test_config_from_dict()` — parse Moshi config.json format
- `test_feed_audio_buffering()` — verify frame buffering logic

Integration tests (require model):
- `test_batch_transcription()` — end-to-end
- `test_streaming_produces_text()` — feed_audio returns text deltas
- `test_finish_flushes()` — finish returns final transcript
- `test_reset()` — reset clears state

---

## B. Moshi TTS Model

### Create: `mlx_audio/tts/models/moshi/__init__.py`
Export `Model`, `ModelConfig`.

### Create: `mlx_audio/tts/models/moshi/config.py`
Dataclass `ModelConfig`:
- `model_type = "moshi_tts"`
- `hf_repo = "kyutai/tts-1.6b-en_fr"`
- `voice_repo = "kyutai/tts-voices"`
- `temp`, `cfg_coef`, `n_q`, `max_gen_length`, `padding_bonus`
- `initial_padding`, `max_padding`, `final_padding`, `padding_between`
- `from_dict()` extracts from Moshi TTS config.json

### Create: `mlx_audio/tts/models/moshi/moshi_tts.py`

**`Model(nn.Module)`**:
- `__init__(config)` — stores config, `_tts_model = None`

**`generate(text, *, voice, ref_audio, temperature, stream, **kwargs)`** → yields `GenerationResult`:
- Build script entries via `tts_model.prepare_script([text])`
- Build condition attributes via `tts_model.make_condition_attributes(voices, cfg_coef)`
- If ref_audio and not multi_speaker: `tts_model.get_prefix(ref_audio)` for prefix
- If `stream=True`:
  - Use `on_frame` callback in `tts_model.generate()` to collect frames
  - For each frame: `mimi.decode_step(codes)` → 1920 samples audio
  - Yield `GenerationResult(is_streaming_chunk=True)` per frame
- If `stream=False`:
  - Run `tts_model.generate()`, batch decode all frames via `mimi.decode()`
  - Yield single `GenerationResult(is_final_chunk=True)`

**Weight loading** — same pattern as STT:
- `sanitize()` returns `{}`
- `post_load_hook()`:
  - Import `TTSModel` from `moshi_mlx.models.tts`
  - Import `Lm, LmConfig, Mimi, mimi_202407`
  - Load config, build LmConfig via `LmConfig.from_config_dict()`
  - Load LM with `load_pytorch_weights(moshi_weight, lm_config, strict=True)`
  - Load Mimi, load SentencePiece tokenizer
  - Create `TTSModel(lm, mimi, tokenizer, raw_config=raw_config, ...)`
  - Attach to `model._tts_model`

### Modify: `mlx_audio/tts/utils.py`
Add to MODEL_REMAPPING:
```python
"moshi_tts": "moshi",
```

### Modify: `mlx_audio/utils.py` — config detection
Add TTS detection:
```python
if "tts_config" in config and "delays" in config:
    model_type = "moshi"
```

### Create: `mlx_audio/tests/test_moshi_tts.py`
- `test_config_from_dict()`
- Integration: `test_generate_audio()`, `test_stream_audio()`

---

## C. Session ABC Conformance

### Modify: `mlx_audio/stt/models/base.py`
- Keep `StreamingSTTSessionBase` as-is (already well-designed)

### Modify: `mlx_audio/stt/models/voxtral_realtime/voxtral_realtime.py`
`StreamingSTTSession` changes:
- Add `(StreamingSTTSessionBase)` to class inheritance
- Add `_model: object = field(default=None, repr=False)` field
- Add `feed_audio(self, pcm_bytes)` → `self._model.feed_audio(self, pcm_bytes)`
- Add `finish(self)` → `self._model.finish_session(self)`
- Add `reset(self)` → re-create session state from model
- Update `create_streaming_session()` to pass `_model=self`

### Modify: `mlx_audio/stt/models/vibevoice_asr/vibevoice_asr.py`
`StreamingVibeVoiceSession` changes:
- Same pattern: inherit `StreamingSTTSessionBase`, add `_model`, delegate methods
- `feed_audio()` still returns `[]` (two-phase design), but conforms to ABC

### Modify: `mlx_audio/stt/models/lasr_ctc/lasr.py`
`StreamingLasrSession` changes:
- Inherit `StreamingSTTSessionBase`, add `_model`
- `feed_audio(pcm_bytes)` → must return `List[str]` not `List[List[int]]`
  - Store tokenizer reference on session, decode CTC tokens to text
  - If no tokenizer available, join token IDs as space-separated string
- `finish()` → decode remaining tokens to text
- `reset()` → clear audio_buffer, prev_chunk_tokens

---

## D. WebSocket Streaming Mode

### Modify: `mlx_audio/server.py`

Add streaming session mode to `stt_realtime_transcriptions()` WebSocket handler. After model loading (line 532), add detection:

```python
# Check if model supports streaming audio input
use_streaming_session = False
mode = config.get("mode", "auto")  # "auto", "vad", "streaming"

if mode != "vad" and hasattr(stt_model, 'supports_streaming_input'):
    if callable(getattr(stt_model, 'supports_streaming_input', None)):
        use_streaming_session = stt_model.supports_streaming_input()
    if mode == "auto":
        # Auto: use streaming if supported, fall back to VAD
        pass
    elif mode == "streaming" and not use_streaming_session:
        await websocket.send_json({"error": "Model does not support streaming input"})
        return
```

**If `use_streaming_session=True`**: New code path:
- `session = stt_model.create_streaming_session()`
- In the receive loop: `text_pieces = session.feed_audio(message["bytes"])`
- Send each text piece as `{"text": piece, "is_partial": True}`
- On `action: "stop"` or disconnect: `final = session.finish()`
- Send `{"text": final, "is_partial": False}`

**If `use_streaming_session=False`**: Existing VAD-based path (unchanged).

This is backward-compatible — existing clients using VAD mode are unaffected. New clients can request `"mode": "streaming"` for models that support it.

---

## E. Async Streaming API

### Create: `mlx_audio/stt/async_streaming.py`

**`AsyncStreamingSTTSession`** class:
- Wraps any `StreamingSTTSessionBase`
- `async feed_audio(pcm_bytes)` → `await loop.run_in_executor(None, session.feed_audio, pcm_bytes)`
- `async finish()` → `await loop.run_in_executor(None, session.finish)`
- `async reset()` → `await loop.run_in_executor(None, session.reset)`
- `async stream_audio(audio_chunks: AsyncIterator[bytes])` → async generator yielding text pieces

This offloads blocking MLX compute to a thread pool, keeping the event loop responsive.

---

## F. Commit docs/

### `git add docs/ && git commit`
The `docs/streaming-analysis.md` is currently untracked.

---

## Key imports from moshi_mlx

```python
# STT model
from moshi_mlx.models import Lm, LmConfig, LmGen, mimi_202407
from moshi_mlx.models.mimi import Mimi
from moshi_mlx.utils import Sampler
from moshi_stt_mlx.engine import _lm_config_from_dict, SILENCE_TOKENS

# TTS model
from moshi_mlx.models.tts import TTSModel
from moshi_mlx.models import Lm, LmConfig, mimi_202407
from moshi_mlx.models.mimi import Mimi
```

## Verification

1. **Unit tests** (no model weights): `pytest mlx_audio/tests/test_moshi_stt_streaming.py -k "not integration"`
2. **STT integration**: `python -m mlx_audio.stt.generate --model kyutai/stt-1b-en_fr-mlx --audio test.wav`
3. **TTS integration**: `python -m mlx_audio.tts.generate --model kyutai/tts-1.6b-en_fr --text "Hello world"`
4. **WebSocket streaming**: Start server, connect WS with `{"model": "kyutai/stt-1b-en_fr-mlx", "mode": "streaming"}`, send audio chunks, verify text deltas arrive
5. **Existing tests pass**: `pytest mlx_audio/tests/test_voxtral_streaming.py mlx_audio/tests/test_vibevoice_streaming.py mlx_audio/tests/test_lasr_streaming.py`
6. **ABC conformance**: verify all sessions pass `isinstance(session, StreamingSTTSessionBase)`
