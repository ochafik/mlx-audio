#!/usr/bin/env python
"""Comprehensive benchmark for STT and TTS models.

Tests all models with f16, q8, q4 quantization levels in both batch and streaming modes.
For STT: measures WER (Word Error Rate) and RTF (Real-Time Factor)
For TTS: measures generation time and audio quality metrics

Usage:
    python benchmark_models.py --stt  # Run STT benchmarks
    python benchmark_models.py --tts  # Run TTS benchmarks
    python benchmark_models.py --all  # Run all benchmarks
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reference transcription for WER calculation
# Genesis 1:1 from the Bible
REFERENCE_TEXT = "In the beginning God created the heaven and the earth"

# Test audio file
TEST_AUDIO = "examples/bible-audiobook/audios/bible-akjv/af_heart/00000001-Genesis-1:1.wav"

# Test text for TTS
TTS_TEST_TEXT = "In the beginning, God created the heaven and the earth."


@dataclass
class STTBenchmarkResult:
    model: str
    quantization: str
    mode: str  # batch or streaming
    load_time: float
    inference_time: float
    audio_duration: float
    rtf: float
    wer: float
    text: str
    error: Optional[str] = None


@dataclass
class TTSBenchmarkResult:
    model: str
    quantization: str
    mode: str  # batch or streaming
    load_time: float
    inference_time: float
    audio_duration: float
    rtf: float
    text_length: int
    error: Optional[str] = None


# STT Models to benchmark (only publicly available ones)
STT_MODELS = [
    {
        "name": "Moshi STT",
        "model_path": "kyutai/stt-1b-en_fr-mlx",
        "model_type": "moshi",
        "sample_rate": 24000,
        "streaming": True,
        "hf_repo": "kyutai/stt-1b-en_fr-mlx",
    },
    {
        "name": "Voxtral Realtime",
        "model_path": "shreyask/voxtral-mini-4b-realtime-mlx-fp16",
        "model_type": "voxtral_realtime",
        "sample_rate": 16000,
        "streaming": True,
        "hf_repo": "mistralai/Voxtral-Mini-4B-Realtime",
    },
    {
        "name": "VibeVoice ASR",
        "model_path": "microsoft/VibeVoice-ASR",
        "model_type": "vibevoice_asr",
        "sample_rate": 16000,
        "streaming": True,
        "hf_repo": "microsoft/VibeVoice-ASR",
    },
]

# TTS Models to benchmark (only publicly available ones)
TTS_MODELS = [
    {
        "name": "Moshi TTS",
        "model_path": "kyutai/tts-1.6b-en_fr",
        "model_type": "moshi",
        "sample_rate": 24000,
        "streaming": True,
        "hf_repo": "kyutai/tts-1.6b-en_fr",
    },
]

# Quantization levels
QUANTIZATION_LEVELS = ["f16", "q8", "q4"]


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    """Load and resample audio."""
    try:
        import sphn
        audio, sr = sphn.read(path, sample_rate=sample_rate)
        return audio[0].astype(np.float32)
    except ImportError:
        import miniaudio
        decoded = miniaudio.decode_file(path, sample_rate=sample_rate)
        audio = np.frombuffer(decoded.samples, dtype=np.float32)
        return audio


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    import re

    # Normalize text
    def normalize(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    ref_words = normalize(reference)
    hyp_words = normalize(hypothesis)

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 100.0

    # Dynamic programming for edit distance
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return (dp[n][m] / len(ref_words)) * 100


def get_quantization_arg(quant: str) -> Optional[int]:
    """Convert quantization string to argument for model loading."""
    if quant == "q4":
        return 4
    elif quant == "q8":
        return 8
    return None


def benchmark_stt_model(
    model_info: Dict, quantization: str, test_audio: str
) -> Tuple[STTBenchmarkResult, Optional[STTBenchmarkResult]]:
    """Benchmark a single STT model with given quantization."""
    from mlx_audio.stt.utils import load_model

    quant_arg = get_quantization_arg(quantization)
    name = model_info["name"]
    sample_rate = model_info["sample_rate"]
    supports_streaming = model_info["streaming"]

    # Load model
    load_start = time.time()
    try:
        model = load_model(model_info["model_path"], quantization=quant_arg)
    except Exception as e:
        return STTBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=0,
            inference_time=0,
            audio_duration=0,
            rtf=0,
            wer=0,
            text="",
            error=f"Load failed: {e}",
        ), None
    load_time = time.time() - load_start

    # Load audio
    audio = load_audio(test_audio, sample_rate)
    audio_duration = len(audio) / sample_rate

    # Batch mode
    batch_start = time.time()
    try:
        result = model.generate(audio)
        batch_time = time.time() - batch_start
        batch_text = result.text
        batch_wer = compute_wer(REFERENCE_TEXT, batch_text)
        batch_result = STTBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=load_time,
            inference_time=batch_time,
            audio_duration=audio_duration,
            rtf=batch_time / audio_duration,
            wer=batch_wer,
            text=batch_text,
        )
    except Exception as e:
        batch_result = STTBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=load_time,
            inference_time=0,
            audio_duration=audio_duration,
            rtf=0,
            wer=100,
            text="",
            error=f"Batch failed: {e}",
        )

    # Streaming mode
    streaming_result = None
    if supports_streaming:
        stream_start = time.time()
        try:
            session = model.create_streaming_session()
            chunk_size = int(sample_rate * 0.1)  # 100ms chunks

            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                int16_chunk = (chunk * 32768).astype(np.int16)
                model.feed_audio(session, int16_chunk.tobytes())

            stream_text = model.finish_session(session)
            stream_time = time.time() - stream_start
            stream_wer = compute_wer(REFERENCE_TEXT, stream_text)

            streaming_result = STTBenchmarkResult(
                model=name,
                quantization=quantization,
                mode="streaming",
                load_time=load_time,
                inference_time=stream_time,
                audio_duration=audio_duration,
                rtf=stream_time / audio_duration,
                wer=stream_wer,
                text=stream_text,
            )
        except Exception as e:
            streaming_result = STTBenchmarkResult(
                model=name,
                quantization=quantization,
                mode="streaming",
                load_time=load_time,
                inference_time=0,
                audio_duration=audio_duration,
                rtf=0,
                wer=100,
                text="",
                error=f"Streaming failed: {e}",
            )

    return batch_result, streaming_result


def benchmark_tts_model(
    model_info: Dict, quantization: str, test_text: str
) -> Tuple[TTSBenchmarkResult, Optional[TTSBenchmarkResult]]:
    """Benchmark a single TTS model with given quantization."""
    from mlx_audio.tts.utils import load_model as load_tts_model

    quant_arg = get_quantization_arg(quantization)
    name = model_info["name"]
    supports_streaming = model_info["streaming"]

    # Load model
    load_start = time.time()
    try:
        model = load_tts_model(model_info["model_path"], quantization=quant_arg)
    except Exception as e:
        return TTSBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=0,
            inference_time=0,
            audio_duration=0,
            rtf=0,
            text_length=len(test_text),
            error=f"Load failed: {e}",
        ), None
    load_time = time.time() - load_start

    # Batch mode
    batch_start = time.time()
    try:
        result = list(model.generate(test_text, stream=False))
        batch_time = time.time() - batch_start
        audio_duration = result[0].audio_duration if result else 0
        batch_result = TTSBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=load_time,
            inference_time=batch_time,
            audio_duration=float(audio_duration) if audio_duration else 0,
            rtf=batch_time / float(audio_duration) if audio_duration else 0,
            text_length=len(test_text),
        )
    except Exception as e:
        batch_result = TTSBenchmarkResult(
            model=name,
            quantization=quantization,
            mode="batch",
            load_time=load_time,
            inference_time=0,
            audio_duration=0,
            rtf=0,
            text_length=len(test_text),
            error=f"Batch failed: {e}",
        )

    # Streaming mode
    streaming_result = None
    if supports_streaming:
        stream_start = time.time()
        try:
            total_audio = 0
            for chunk in model.generate(test_text, stream=True):
                if hasattr(chunk, "audio_duration"):
                    total_audio += float(chunk.audio_duration)
            stream_time = time.time() - stream_start

            streaming_result = TTSBenchmarkResult(
                model=name,
                quantization=quantization,
                mode="streaming",
                load_time=load_time,
                inference_time=stream_time,
                audio_duration=total_audio,
                rtf=stream_time / total_audio if total_audio > 0 else 0,
                text_length=len(test_text),
            )
        except Exception as e:
            streaming_result = TTSBenchmarkResult(
                model=name,
                quantization=quantization,
                mode="streaming",
                load_time=load_time,
                inference_time=0,
                audio_duration=0,
                rtf=0,
                text_length=len(test_text),
                error=f"Streaming failed: {e}",
            )

    return batch_result, streaming_result


def format_cell(result, metric: str) -> str:
    """Format a cell for the markdown table."""
    if result is None:
        return "-"
    if result.error:
        return f"ERROR"

    if metric == "rtf":
        return f"{result.rtf:.2f}x"
    elif metric == "wer":
        return f"{result.wer:.1f}%"
    elif metric == "time":
        return f"{result.inference_time:.2f}s"
    elif metric == "load":
        return f"{result.load_time:.1f}s"

    return str(getattr(result, metric, "?"))


def generate_stt_markdown_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table for STT results."""
    lines = []
    lines.append("## STT Benchmarks\n")
    lines.append(
        f"**Test Audio:** {TEST_AUDIO} ({REFERENCE_TEXT[:50]}...)\n"
    )
    lines.append("**Metrics:**")
    lines.append("- RTF: Real-Time Factor (lower is better, <1.0 means faster than real-time)")
    lines.append("- WER: Word Error Rate (lower is better)\n")

    # RTF Table
    lines.append("### Real-Time Factor (RTF)\n")
    lines.append("| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |")
    lines.append("|-------|-----------|----------|----------|------------|-----------|-----------|")

    for model_name in STT_MODELS:
        name = model_name["name"]
        row = f"| {name} |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                batch = results[name][quant].get("batch")
                row += f" {format_cell(batch, 'rtf')} |"
            else:
                row += " - |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                stream = results[name][quant].get("streaming")
                row += f" {format_cell(stream, 'rtf')} |"
            else:
                row += " - |"
        lines.append(row)

    # WER Table
    lines.append("\n### Word Error Rate (WER)\n")
    lines.append("| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |")
    lines.append("|-------|-----------|----------|----------|------------|-----------|-----------|")

    for model_name in STT_MODELS:
        name = model_name["name"]
        row = f"| {name} |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                batch = results[name][quant].get("batch")
                row += f" {format_cell(batch, 'wer')} |"
            else:
                row += " - |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                stream = results[name][quant].get("streaming")
                row += f" {format_cell(stream, 'wer')} |"
            else:
                row += " - |"
        lines.append(row)

    # Load Time Table
    lines.append("\n### Model Load Time\n")
    lines.append("| Model | F16 | Q8 | Q4 |")
    lines.append("|-------|-----|-----|-----|")

    for model_name in STT_MODELS:
        name = model_name["name"]
        row = f"| {name} |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                batch = results[name][quant].get("batch")
                row += f" {format_cell(batch, 'load')} |"
            else:
                row += " - |"
        lines.append(row)

    # Transcription Samples
    lines.append("\n### Sample Transcriptions (F16 Batch)\n")
    lines.append("| Model | Transcription |")
    lines.append("|-------|---------------|")

    for model_name in STT_MODELS:
        name = model_name["name"]
        if "f16" in results.get(name, {}):
            batch = results[name]["f16"].get("batch")
            if batch and not batch.error:
                text = batch.text[:80] + "..." if len(batch.text) > 80 else batch.text
                lines.append(f"| {name} | {text} |")

    return "\n".join(lines)


def generate_tts_markdown_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table for TTS results."""
    lines = []
    lines.append("## TTS Benchmarks\n")
    lines.append(f"**Test Text:** {TTS_TEST_TEXT}\n")
    lines.append("**Metrics:**")
    lines.append("- RTF: Real-Time Factor (lower is better)")
    lines.append("- Audio Duration: Generated audio length\n")

    # RTF Table
    lines.append("### Real-Time Factor (RTF)\n")
    lines.append("| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |")
    lines.append("|-------|-----------|----------|----------|------------|-----------|-----------|")

    for model_name in TTS_MODELS:
        name = model_name["name"]
        row = f"| {name} |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                batch = results[name][quant].get("batch")
                row += f" {format_cell(batch, 'rtf')} |"
            else:
                row += " - |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                stream = results[name][quant].get("streaming")
                row += f" {format_cell(stream, 'rtf')} |"
            else:
                row += " - |"
        lines.append(row)

    # Generation Time Table
    lines.append("\n### Generation Time\n")
    lines.append("| Model | Batch F16 | Batch Q8 | Batch Q4 | Stream F16 | Stream Q8 | Stream Q4 |")
    lines.append("|-------|-----------|----------|----------|------------|-----------|-----------|")

    for model_name in TTS_MODELS:
        name = model_name["name"]
        row = f"| {name} |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                batch = results[name][quant].get("batch")
                row += f" {format_cell(batch, 'time')} |"
            else:
                row += " - |"
        for quant in QUANTIZATION_LEVELS:
            if quant in results.get(name, {}):
                stream = results[name][quant].get("streaming")
                row += f" {format_cell(stream, 'time')} |"
            else:
                row += " - |"
        lines.append(row)

    return "\n".join(lines)


def run_stt_benchmarks(
    models: List[Dict], quantization_levels: List[str], test_audio: str
) -> Dict:
    """Run all STT benchmarks."""
    results = {}

    for model_info in models:
        name = model_info["name"]
        results[name] = {}
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        for quant in quantization_levels:
            print(f"\n--- {quant.upper()} ---")
            try:
                batch_result, streaming_result = benchmark_stt_model(
                    model_info, quant, test_audio
                )
                results[name][quant] = {"batch": batch_result}
                if streaming_result:
                    results[name][quant]["streaming"] = streaming_result

                if batch_result.error:
                    print(f"Batch ERROR: {batch_result.error}")
                else:
                    print(
                        f"Batch: RTF={batch_result.rtf:.2f}x, WER={batch_result.wer:.1f}%"
                    )

                if streaming_result:
                    if streaming_result.error:
                        print(f"Streaming ERROR: {streaming_result.error}")
                    else:
                        print(
                            f"Streaming: RTF={streaming_result.rtf:.2f}x, WER={streaming_result.wer:.1f}%"
                        )
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                results[name][quant] = {
                    "batch": STTBenchmarkResult(
                        model=name,
                        quantization=quant,
                        mode="batch",
                        load_time=0,
                        inference_time=0,
                        audio_duration=0,
                        rtf=0,
                        wer=100,
                        text="",
                        error=str(e),
                    )
                }

    return results


def run_tts_benchmarks(
    models: List[Dict], quantization_levels: List[str], test_text: str
) -> Dict:
    """Run all TTS benchmarks."""
    results = {}

    for model_info in models:
        name = model_info["name"]
        results[name] = {}
        print(f"\n{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        for quant in quantization_levels:
            print(f"\n--- {quant.upper()} ---")
            try:
                batch_result, streaming_result = benchmark_tts_model(
                    model_info, quant, test_text
                )
                results[name][quant] = {"batch": batch_result}
                if streaming_result:
                    results[name][quant]["streaming"] = streaming_result

                if batch_result.error:
                    print(f"Batch ERROR: {batch_result.error}")
                else:
                    print(
                        f"Batch: RTF={batch_result.rtf:.2f}x, Time={batch_result.inference_time:.2f}s"
                    )

                if streaming_result:
                    if streaming_result.error:
                        print(f"Streaming ERROR: {streaming_result.error}")
                    else:
                        print(
                            f"Streaming: RTF={streaming_result.rtf:.2f}x, Time={streaming_result.inference_time:.2f}s"
                        )
            except Exception as e:
                print(f"FAILED: {e}")
                traceback.print_exc()
                results[name][quant] = {
                    "batch": TTSBenchmarkResult(
                        model=name,
                        quantization=quant,
                        mode="batch",
                        load_time=0,
                        inference_time=0,
                        audio_duration=0,
                        rtf=0,
                        text_length=len(test_text),
                        error=str(e),
                    )
                }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark STT and TTS models")
    parser.add_argument("--stt", action="store_true", help="Run STT benchmarks")
    parser.add_argument("--tts", action="store_true", help="Run TTS benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument(
        "--models", type=str, help="Comma-separated list of models to test"
    )
    parser.add_argument(
        "--quant", type=str, default="f16,q8,q4", help="Quantization levels to test"
    )
    parser.add_argument("--output", type=str, default="BENCHMARKS.md", help="Output file")
    parser.add_argument(
        "--quick", action="store_true", help="Quick test with only f16"
    )
    args = parser.parse_args()

    if not (args.stt or args.tts or args.all):
        args.all = True

    quant_levels = args.quant.split(",") if not args.quick else ["f16"]

    output_lines = []
    output_lines.append("# MLX-Audio Model Benchmarks\n")
    output_lines.append(
        "Generated by `python benchmark_models.py`\n\n"
    )
    output_lines.append("## Methodology\n\n")
    output_lines.append("### STT Benchmark\n")
    output_lines.append(f"- **Test Audio:** `{TEST_AUDIO}`")
    output_lines.append(f"- **Reference Text:** \"{REFERENCE_TEXT}\"")
    output_lines.append("- **Metrics:**")
    output_lines.append("  - RTF (Real-Time Factor): inference_time / audio_duration")
    output_lines.append(
        "  - WER (Word Error Rate): edit distance / reference_length * 100"
    )
    output_lines.append("- **Quantization:**")
    output_lines.append("  - F16: Full precision (bfloat16)")
    output_lines.append("  - Q8: 8-bit quantization")
    output_lines.append("  - Q4: 4-bit quantization\n")
    output_lines.append("### TTS Benchmark\n")
    output_lines.append(f"- **Test Text:** \"{TTS_TEST_TEXT}\"")
    output_lines.append("- **Metrics:**")
    output_lines.append("  - RTF (Real-Time Factor): inference_time / audio_duration")
    output_lines.append("  - Lower is better (<1.0 = faster than real-time)\n")
    output_lines.append("---\n")

    if args.stt or args.all:
        print("\n" + "=" * 60)
        print("STT BENCHMARKS")
        print("=" * 60)

        models_to_test = STT_MODELS
        if args.models:
            model_names = args.models.split(",")
            models_to_test = [m for m in STT_MODELS if m["name"] in model_names]

        stt_results = run_stt_benchmarks(models_to_test, quant_levels, TEST_AUDIO)
        output_lines.append(generate_stt_markdown_table(stt_results))

    if args.tts or args.all:
        print("\n" + "=" * 60)
        print("TTS BENCHMARKS")
        print("=" * 60)

        models_to_test = TTS_MODELS
        if args.models:
            model_names = args.models.split(",")
            models_to_test = [m for m in TTS_MODELS if m["name"] in model_names]

        tts_results = run_tts_benchmarks(models_to_test, quant_levels, TTS_TEST_TEXT)
        output_lines.append(generate_tts_markdown_table(tts_results))

    # Write output
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\n\nResults written to {args.output}")


if __name__ == "__main__":
    main()
