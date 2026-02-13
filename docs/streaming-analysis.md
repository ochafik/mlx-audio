# mlx-audio Streaming STT Analysis (`feat/streaming-stt`)

> Analysis of streaming speech-to-text support across three model architectures.
> Branch: `feat/streaming-stt` (6 commits)

---

## Commits (oldest → newest)

1. **Add Voxtral Mini 4B Realtime model** — Full model implementation (encoder, decoder, tokenizer, weight converter)
2. **Add streaming STT for Voxtral Realtime** — Streaming mel, encoder caches, `StreamingSTTSession`
3. **Add streaming STT for VibeVoice-ASR** — Two-phase design (encode all → decode)
4. **Add chunk-based streaming for LASR-CTC** — Overlap-stitch approach for non-causal encoder
5. **Use SDPA causal mask for decoder prefill** — Optimization
6. **Add incremental encoder for faster TTFT** — The headline optimization: **4.7x TTFT improvement** (2700ms → 570ms for 127s audio)

---

## Architecture Overview: Three Streaming Strategies

| Model | Encoder | Streaming Mode | Output Latency | Key Innovation |
|-------|---------|---------------|----------------|----------------|
| **Voxtral Realtime** | 32-layer causal transformer | Frame-by-frame | Per-token (~80ms) | Incremental encoding via generator |
| **VibeVoice-ASR** | Dual SConv1d (acoustic + semantic) | Two-phase (encode → decode) | After `finish_session()` | Parallel tokenizer caching |
| **LASR-CTC** | Bidirectional transformer | 2s overlapping chunks | Per-chunk (~1.5-2s) | CTC overlap stitching |

---

## The Star: Incremental Encoder (Voxtral)

The most architecturally interesting piece. The insight:

> Since the encoder is **causal** with a sliding window (750 frames), you don't need to encode ALL audio before starting decoding. Encode just enough for the prompt (BOS + left_pad + delay tokens), start decoding, then lazily encode remaining audio chunks during the autoregressive loop.

```
Traditional:  [encode ALL audio ~~~~~~~~] → [prefill] → [decode tokens...]
                      2700ms                   50ms        variable

Incremental:  [encode 1 chunk] → [prefill] → [decode + encode remaining...]
                   400ms            50ms        overlapped
                              TTFT: 570ms (4.7x faster)
```

Implementation in `encoder.py:encode_chunks()`:
- Generator yields encoded chunks of `sliding_window=750` frames
- Uses `RotatingKVCache` per layer (bounded memory)
- `_encode_and_prefill()` consumes just enough chunks for prompt, then passes the generator to the decode loop
- Decode loop calls `next(enc_chunk_gen)` on demand when `pos >= adapter_len`

### Why This Works

The causal encoder processes left-to-right with a fixed attention window. Each chunk's output depends only on the current chunk and the KV cache from previous chunks — never on future audio. This means encoding and decoding can be **interleaved** without changing the output.

For short audio (≤750 frames), there's no overhead — the encoder processes everything in a single pass as before.

---

## Streaming Session API (`base.py`)

Clean abstract protocol:

```python
class StreamingSTTSessionBase(ABC):
    def feed_audio(self, pcm_bytes: bytes) -> List[str]   # feed chunk, get text deltas
    def finish(self) -> str                                  # flush remaining
    def reset(self) -> None                                  # reuse session
```

Each model implements this differently:

- **Voxtral**: `feed_audio()` → mel → encoder → decoder token → text delta (real streaming)
- **VibeVoice**: `feed_audio()` → encode features only, `finish()` → LM decode (two-phase)
- **LASR**: `feed_audio()` → buffer → CTC decode chunk when buffer full (chunk-based)

---

## Model Deep Dives

### 1. Voxtral Realtime

**Architecture**: 0.6B causal audio encoder (32 layers, RoPE, sliding window=750) + 3.4B GQA decoder (26 layers)

**Inference pipeline**:
1. Resample audio to 16kHz, pad (left silence + right silence)
2. Compute mel spectrogram (streaming: overlap buffer with window=400, hop=160)
3. Run causal encoder → 4x downsample → adapter MLP
4. Construct prompt: `[BOS] + [STREAMING_PAD] * (n_left_pad + n_delay)`
5. For each position: `input = audio_embed + tok_embed(token_id)`
6. Prefill decoder, then autoregressive generation until EOS
7. Decode tokens via Tekken tokenizer

**Session state** (`StreamingSTTSession`):
- `mel_state`: STFT overlap buffer + feature accumulator
- `encoder_state`: conv caches (conv0, conv1), KV caches (per layer), downsample buffer
- `decoder_cache`: KV cache for autoregressive generation
- `adapter_buffer`: buffered embeddings awaiting decoding
- `generated_tokens`: accumulated output tokens
- `decoder_position`: RoPE position counter

**Streaming mel**: `compute_mel_streaming()` maintains an overlap buffer of `window_size` (400) samples across chunk boundaries, computing STFT frames incrementally.

**Streaming encoder**: Each transformer layer maintains:
- Conv cache: `kernel_size - stride` frames for causal continuity
- KV cache: sliding window of 750 frames (bounded memory)
- Positions tracked via `state.position` for absolute RoPE

**Downsample buffering**: The 4x downsampler needs 4 encoder frames to produce 1 adapter token. A `downsample_buffer` accumulates frames until a full group is available.

**Text delta via tokenizer diff**: Decodes accumulated tokens after each new token, diffs against previous text to yield only the delta. Handles BPE boundary issues where a token's text representation changes when the next token arrives.

### 2. VibeVoice-ASR

**Architecture**: Dual SConv1d tokenizer encoders (acoustic + semantic) → Qwen2 language model

**Two-phase design**:
- **Phase 1 — Encoding** (`feed_audio()`): Audio chunks run through both tokenizer encoders with per-layer conv caches. Features accumulate in memory. No text produced.
- **Phase 2 — Decoding** (`finish_session()`): Concatenated features projected through connectors, combined, fed to LM with streaming token generation.

**Conv caching**: Each `SConv1d` layer stores `kernel_size - stride` frames. `Block1D` (conv + norm + gating) maintains a cache tuple `(input_cache, running_state)`. Total caches per encoder: ~10-15 (downsample layers + block residuals + head).

**Sample rate**: 24 kHz (higher than Voxtral's 16 kHz).

**Tradeoff**: All features must be accumulated before LM decode begins. Useful for scenarios where audio is recorded first, less useful for real-time interim transcripts.

### 3. LASR-CTC

**Architecture**: Bidirectional transformer encoder + CTC head (linear projection to vocab)

**Chunk-based strategy**: The bidirectional encoder prevents true frame-by-frame streaming. Instead:
1. Buffer incoming audio until 2s chunk accumulated (32000 samples at 16 kHz)
2. Extract chunk with 0.5s overlap (stride = 1.5s)
3. Compute mel spectrogram (numpy-only, no MLX for mel)
4. Forward through encoder + CTC head
5. Greedy CTC decode: `argmax(logits)` → remove consecutive duplicates → remove blanks
6. Overlap deduplication: match tail of prev chunk with head of current

**Session state** (`StreamingLasrSession`):
- `chunk_samples`: 32000 (2s at 16kHz)
- `overlap_samples`: 8000 (0.5s)
- `stride`: 24000 (1.5s)
- `audio_buffer`: accumulates until full chunk
- `prev_chunk_tokens`: for overlap deduplication

---

## Key Implementation Patterns

### 1. Conv Cache Streaming

Both Voxtral and VibeVoice cache `kernel_size - stride` frames across chunk boundaries for causal convolution continuity:

```python
# Cache stores last (kernel_size - stride) frames for left-only padding
# Avoids recomputing left-padded context on each call
pad_size = kernel_size - stride  # e.g., 3-1=2 for conv0, 3-2=1 for conv1
new_cache = concatenated_output[-pad_size:]
```

### 2. Explicit `mx.eval()`

Critical for MLX's lazy evaluation — without periodic eval calls, the computation graph grows unbounded. Called after prefill and each encoder chunk.

### 3. Generator-Based Chunked Encoding

`encode_chunks()` is a Python generator that yields encoded chunks:
```python
def encode_chunks(self, conv_out):
    for chunk_start in range(0, seq_len, sw):
        x = conv_out[chunk_start:chunk_end]
        for i, layer in enumerate(self.transformer_layers):
            mask = caches[i].make_mask(chunk_len, window_size=sw)
            x = layer(x, rope_cos, rope_sin, mask, cache=caches[i])
        yield self.transformer_norm(x)
```

This lets the caller (decode loop) pull encoded audio on demand, interleaving encoding with decoding.

### 4. CTC Overlap Stitching

LASR handles chunk boundaries by overlapping and deduplicating:
- Previous chunk's tail tokens compared with current chunk's head tokens
- Matching prefix removed from current chunk
- Produces seamless transcript across chunk boundaries

---

## Test Coverage

| File | Lines | Covers |
|------|-------|--------|
| `test_voxtral_streaming.py` | 218 | Mel streaming matches batch, encoder state init, tiny input buffering, end-to-end streaming |
| `test_vibevoice_streaming.py` | 198 | SConv1d streaming (stride=1 and stride=2), depthwise conv, Block1D, TokenizerEncoder cache/shape, session creation |
| `test_lasr_streaming.py` | 118 | Session creation, audio buffering thresholds, chunk→token production, PCM int16 input, mel feature shape |

---

## Benchmark (`benchmark.py`)

Voxtral-specific performance evaluation script:
- **TTFT measurement**: First-token latency across multiple runs
- **Throughput**: Tokens/second during decode phase
- **Streaming output**: Token-by-token generation with visible output
- **Optional TTS generation** for test audio (via `pocket-tts`)

```bash
python -m mlx_audio.stt.models.voxtral_realtime.benchmark \
    --model /path/to/converted --generate-audio
```

---

## Potential Improvements

1. **Voxtral streaming session doesn't yield text during `feed_audio()`** for the real-time case — it accumulates mel/encoder state but the decoder only runs during `finish_session()`. The incremental encoder work enables frame-by-frame decoding during feed, but the session API doesn't wire this up yet.

2. **No async API** — all operations are synchronous/blocking. For integration with event-loop-based voice agents, you'd need a thread wrapper or async generator bridge.

3. **`StreamingSTTSessionBase` is minimal** — no way to query partial transcripts without `finish()`, no confidence scores, no word timestamps during streaming.

4. **VibeVoice two-phase design** means zero text output until audio is complete — less useful for real-time scenarios where you want interim transcripts. The conv caching is correct but the LM decode is deferred.

---

## Cross-Model Comparison

| Feature | Voxtral | VibeVoice | LASR-CTC |
|---------|---------|-----------|----------|
| **Causal encoder** | Yes (sliding window=750) | Yes (SConv1d causal chains) | No (bidirectional) |
| **Streaming mode** | Frame-by-frame | Two-phase (encode all → decode) | Chunk-based (2s) |
| **Sample rate** | 16 kHz | 24 kHz | 16 kHz |
| **Output latency** | Per-token (~12.5 tok/s) | After `finish_session()` | Per-chunk (1.5-2s) |
| **Memory profile** | Sliding windows (bounded) | Accumulated features (linear) | Current + prev chunk |
| **Decoding** | Autoregressive LM | LM during finish | CTC (greedy collapse) |
| **Text during feed** | Possible (not yet wired) | No | Yes (per-chunk) |
| **Thread safety** | Session-isolated | Session-isolated | Session-isolated |
