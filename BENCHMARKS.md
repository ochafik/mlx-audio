# MLX-Audio Model Benchmarks

## Methodology

### Test Configuration
- **STT Audio:** `examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav` (~6.6s)
- **TTS Text:** "In the beginning, God created the heaven and the earth."
- **Hardware:** Apple Silicon with MLX framework
- **Quantization:** F16 (full precision), Q8 (8-bit), Q4 (4-bit)

### Metrics
- **RTF (Real-Time Factor):** inference_time / audio_duration
- **Lower is better** (<1.0 means faster than real-time)

### Reproducibility
```bash
python benchmark_models.py --all
```

---

## Results

### STT Models (Speech-to-Text)

| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |
|-------|-----------|----------|----------|------------|-----------|-----------|
| Moshi STT | **0.23x** | 0.23x | 0.22x | 0.29x | 0.27x | 0.29x |
| Voxtral Realtime | 1.52x | 1.55x | 1.60x | 1.30x | 1.29x | **1.29x** |
| VibeVoice ASR | 1.52x | 2.03x | **1.50x** | 3.54x | 2.69x | **1.51x** |

### TTS Models (Text-to-Speech)

| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |
|-------|-----------|----------|----------|------------|-----------|-----------|
| Moshi TTS | 2.78x | 2.50x | 2.38x | 1.02x | 0.97x | **0.92x** |
| Pocket TTS* | **0.14x** | - | - | - | - | - |

*\*Pocket TTS is a separate package (`pip install pocket-tts[mlx]`). Quantization not supported due to small layer sizes.*

---

## Key Findings

### STT
1. **Moshi STT** is fastest at **0.22x RTF** (~5x realtime) with excellent quality
2. **Voxtral Realtime** streaming is 15% faster than batch (1.29x vs 1.52x)
3. **VibeVoice ASR** outputs JSON diarization format with speaker segments
4. **Quantization** has minimal impact on STT performance

### TTS
1. **Pocket TTS** is fastest at **0.14x RTF** (~7x realtime) - separate package
2. **Moshi TTS streaming** achieves **0.92x RTF** (faster than realtime!) with Q4
3. **Streaming is 2.6x faster** than batch mode for Moshi TTS
4. **Q4 quantization** provides the best Moshi TTS performance

---

## Model Details

| Model | Type | Params | Sample Rate | Streaming | Package |
|-------|------|--------|-------------|-----------|---------|
| Moshi STT | STT | 1B | 24kHz | Yes | mlx-audio |
| Voxtral Realtime | STT | 4B | 16kHz | Yes | mlx-audio |
| VibeVoice ASR | STT | - | 16kHz | Yes* | mlx-audio |
| Moshi TTS | TTS | 1.6B | 24kHz | Yes | mlx-audio |
| Pocket TTS | TTS | ~1B | 24kHz | No | pocket-tts |

*\*VibeVoice uses two-phase streaming (encode during feed, decode at finish)*

---

## Performance Comparison

### Fastest STT (Batch)
1. **Moshi STT** - 0.22x RTF (5x realtime)
2. VibeVoice ASR - 1.50x RTF
3. Voxtral Realtime - 1.52x RTF

### Fastest TTS (Batch)
1. **Pocket TTS** - 0.14x RTF (7x realtime)
2. Moshi TTS Q4 - 2.38x RTF

### Best Streaming TTS
1. **Moshi TTS Q4** - 0.92x RTF (faster than realtime!)
