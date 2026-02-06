"""Convert Voxtral-Mini-4B-Realtime-2602 from Mistral-native format to mlx-audio format.

Usage:
    python -m mlx_audio.stt.models.voxtral_realtime.convert \
        --input-path ~/.cache/huggingface/hub/models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots/<hash> \
        --output-path ./converted-voxtral-realtime

This converts:
    - params.json → config.json
    - consolidated.safetensors → model.safetensors (with key remapping)
    - Copies tekken.json as-is
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np


def convert_params_to_config(params: dict) -> dict:
    """Convert Mistral params.json to mlx-audio config.json format."""
    # Handle Mistral-native nested structure:
    # params.json has multimodal.whisper_model_args.encoder_args for audio config
    multimodal = params.get("multimodal", {})
    whisper_args = multimodal.get("whisper_model_args", {})
    encoder_args = whisper_args.get("encoder_args", {})
    downsample_args = whisper_args.get("downsample_args", {})

    # Use encoder_args if available, otherwise fall back to audio_config
    audio_params = encoder_args or params.get("audio_config", {})
    audio_encoding = audio_params.get("audio_encoding_args", {})
    text_params = params.get("text_config", {})

    # Extract encoder config from Mistral format
    audio_config = {
        "hidden_size": audio_params.get("dim", audio_params.get("hidden_size", 1280)),
        "num_hidden_layers": audio_params.get(
            "n_layers", audio_params.get("num_hidden_layers", 32)
        ),
        "num_attention_heads": audio_params.get(
            "n_heads", audio_params.get("num_attention_heads", 32)
        ),
        "num_key_value_heads": audio_params.get(
            "n_heads", audio_params.get("num_key_value_heads", 32)
        ),
        "head_dim": audio_params.get("head_dim", 64),
        "intermediate_size": audio_params.get(
            "hidden_dim", audio_params.get("intermediate_size", 5120)
        ),
        "rms_norm_eps": audio_params.get("norm_eps", audio_params.get("rms_norm_eps", 1e-5)),
        "rope_theta": audio_params.get("rope_theta", 1e6),
        "num_mel_bins": audio_encoding.get("num_mel_bins", audio_params.get("num_mel_bins", 128)),
        "encoder_layers": audio_params.get(
            "n_layers", audio_params.get("encoder_layers", 32)
        ),
        "encoder_attention_heads": audio_params.get(
            "n_heads", audio_params.get("encoder_attention_heads", 32)
        ),
        "encoder_ffn_dim": audio_params.get(
            "hidden_dim", audio_params.get("encoder_ffn_dim", 5120)
        ),
        "d_model": audio_params.get("dim", audio_params.get("d_model", 1280)),
        "is_causal": audio_params.get("causal", audio_params.get("is_causal", True)),
        "sliding_window": audio_params.get("sliding_window", 750),
        "downsample_factor": downsample_args.get(
            "downsample_factor",
            audio_params.get("block_pool_size", audio_params.get("downsample_factor", 4)),
        ),
    }

    # Extract text/decoder config
    dim = text_params.get("dim", params.get("dim", 3072))
    n_layers = text_params.get("n_layers", params.get("n_layers", 26))
    n_heads = text_params.get("n_heads", params.get("n_heads", 32))
    n_kv_heads = text_params.get("n_kv_heads", params.get("n_kv_heads", 8))
    head_dim = text_params.get("head_dim", params.get("head_dim", 128))

    text_config = {
        "model_type": "mistral_realtime",
        "vocab_size": text_params.get("vocab_size", params.get("vocab_size", 131072)),
        "hidden_size": dim,
        "intermediate_size": text_params.get(
            "hidden_dim", params.get("hidden_dim", 9216)
        ),
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": text_params.get("rms_norm_eps", params.get("norm_eps", 1e-5)),
        "rope_theta": text_params.get("rope_theta", params.get("rope_theta", 1e6)),
        "rope_traditional": True,
        "tie_word_embeddings": True,
        "sliding_window": text_params.get(
            "sliding_window", params.get("sliding_window", 8192)
        ),
        "ada_rms_norm_t_cond": True,
        "ada_rms_norm_t_cond_dim": 32,
    }

    config = {
        "model_type": "voxtral_realtime",
        "audio_config": audio_config,
        "text_config": text_config,
        "audio_token_id": params.get("audio_token_id", 24),
        "projector_hidden_act": "gelu",
    }

    return config


def remap_key(key: str) -> str:
    """Remap a single weight key from Mistral-native to MLX format."""
    new_key = key

    # Encoder convolutions
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.0.conv",
        "audio_tower.conv1.conv",
    )
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.whisper_encoder.conv_layers.1.conv",
        "audio_tower.conv2.conv",
    )

    # Encoder transformer
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention_norm",
        r"audio_tower.layers.\1.self_attn_layer_norm",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wq",
        r"audio_tower.layers.\1.self_attn.q_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wk",
        r"audio_tower.layers.\1.self_attn.k_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wv",
        r"audio_tower.layers.\1.self_attn.v_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.attention\.wo",
        r"audio_tower.layers.\1.self_attn.out_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.ffn_norm",
        r"audio_tower.layers.\1.final_layer_norm",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w1",
        r"audio_tower.layers.\1.gate_proj.gate_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w2",
        r"audio_tower.layers.\1.gate_proj.down_proj",
        new_key,
    )
    new_key = re.sub(
        r"mm_streams_embeddings\.embedding_module\.whisper_encoder\.transformer\.layers\.(\d+)\.feed_forward\.w3",
        r"audio_tower.layers.\1.gate_proj.up_proj",
        new_key,
    )

    # Encoder final norm
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.whisper_encoder.transformer.norm",
        "audio_tower.layer_norm",
    )

    # Multi-modal projector
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.audio_language_projection.0",
        "multi_modal_projector.linear_1",
    )
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.audio_language_projection.2",
        "multi_modal_projector.linear_2",
    )

    # Embed tokens
    new_key = new_key.replace(
        "mm_streams_embeddings.embedding_module.tok_embeddings",
        "language_model.model.embed_tokens",
    )

    # Decoder layers
    new_key = re.sub(
        r"^layers\.(\d+)\.attention_norm",
        r"language_model.model.layers.\1.input_layernorm",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.attention\.wq",
        r"language_model.model.layers.\1.self_attn.q_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.attention\.wk",
        r"language_model.model.layers.\1.self_attn.k_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.attention\.wv",
        r"language_model.model.layers.\1.self_attn.v_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.attention\.wo",
        r"language_model.model.layers.\1.self_attn.o_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.ffn_norm",
        r"language_model.model.layers.\1.post_attention_layernorm",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.feed_forward\.w1",
        r"language_model.model.layers.\1.mlp.gate_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.feed_forward\.w2",
        r"language_model.model.layers.\1.mlp.down_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.feed_forward\.w3",
        r"language_model.model.layers.\1.mlp.up_proj",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.ada_rms_norm_t_cond\.0",
        r"language_model.model.layers.\1.ada_rms_norm.down",
        new_key,
    )
    new_key = re.sub(
        r"^layers\.(\d+)\.ada_rms_norm_t_cond\.2",
        r"language_model.model.layers.\1.ada_rms_norm.up",
        new_key,
    )

    # Final norm
    if new_key == "norm.weight":
        new_key = "language_model.model.norm.weight"

    return new_key


def convert(input_path: str, output_path: str) -> None:
    """Convert model from Mistral-native to mlx-audio format."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Convert params.json → config.json
    params_file = input_path / "params.json"
    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
        config = convert_params_to_config(params)
    else:
        raise FileNotFoundError(f"params.json not found at {input_path}")

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config.json to {output_path / 'config.json'}")

    # 2. Convert weights
    import glob as glob_mod

    weight_files = sorted(glob_mod.glob(str(input_path / "consolidated*.safetensors")))
    if not weight_files:
        weight_files = sorted(glob_mod.glob(str(input_path / "*.safetensors")))

    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found at {input_path}")

    all_weights = {}
    for wf in weight_files:
        print(f"Loading {wf}...")
        all_weights.update(mx.load(wf))

    # Remap keys and transpose conv weights
    remapped = {}
    for k, v in all_weights.items():
        new_key = remap_key(k)

        # Transpose conv weights: PyTorch [out, in, k] → MLX [out, k, in]
        if "conv" in new_key and "weight" in new_key and v.ndim == 3:
            if v.shape[-1] < v.shape[-2]:
                v = v.transpose(0, 2, 1)

        remapped[new_key] = v

    # Save as sharded safetensors (4GB shards)
    shard_size = 4 * 1024 * 1024 * 1024  # 4GB
    current_shard = {}
    current_size = 0
    shard_idx = 0

    for k, v in sorted(remapped.items()):
        nbytes = v.nbytes
        if current_size + nbytes > shard_size and current_shard:
            shard_name = f"model-{shard_idx:05d}-of-*.safetensors"
            mx.save_safetensors(str(output_path / shard_name), current_shard)
            print(f"Wrote shard {shard_idx} ({current_size / 1e9:.2f} GB)")
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[k] = v
        current_size += nbytes

    if current_shard:
        if shard_idx == 0:
            fname = "model.safetensors"
        else:
            fname = f"model-{shard_idx:05d}-of-*.safetensors"
        mx.save_safetensors(str(output_path / fname), current_shard)
        print(f"Wrote {fname} ({current_size / 1e9:.2f} GB)")

    # 3. Copy tekken.json
    tekken_src = input_path / "tekken.json"
    if tekken_src.exists():
        shutil.copy2(tekken_src, output_path / "tekken.json")
        print(f"Copied tekken.json")

    print(f"\nConversion complete! Output at: {output_path}")
    print(f"Total weights: {len(remapped)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Voxtral-Mini-4B-Realtime-2602 to mlx-audio format"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to the Mistral-native model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path for the converted output directory",
    )
    args = parser.parse_args()
    convert(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
