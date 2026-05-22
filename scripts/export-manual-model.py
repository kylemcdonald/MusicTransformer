#!/usr/bin/env python3
"""Export the TensorFlow checkpoint as raw browser-readable float32 tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


CHECKPOINT_PREFIX = "checkpoint/unconditional_model_16.ckpt"
OUTPUT_DIR = "public/manual-model"

HIDDEN_SIZE = 512
FILTER_SIZE = 2048
NUM_HEADS = 8
NUM_LAYERS = 16
VOCAB_SIZE = 310


def checkpoint_name(layer: int, suffix: str) -> str:
    return f"transformer/body/decoder/layer_{layer}/{suffix}"


def tensor_names() -> list[tuple[str, str]]:
    names: list[tuple[str, str]] = [
        (
            "embedding",
            "transformer/symbol_modality_310_512/shared/weights_0",
        ),
    ]

    for layer in range(NUM_LAYERS):
        prefix = f"layers.{layer}"
        names.extend(
            [
                (
                    f"{prefix}.attn_ln_scale",
                    checkpoint_name(
                        layer,
                        "self_attention/layer_prepostprocess/layer_norm/layer_norm_scale",
                    ),
                ),
                (
                    f"{prefix}.attn_ln_bias",
                    checkpoint_name(
                        layer,
                        "self_attention/layer_prepostprocess/layer_norm/layer_norm_bias",
                    ),
                ),
                (
                    f"{prefix}.wq",
                    checkpoint_name(layer, "self_attention/multihead_attention/q/kernel"),
                ),
                (
                    f"{prefix}.wk",
                    checkpoint_name(layer, "self_attention/multihead_attention/k/kernel"),
                ),
                (
                    f"{prefix}.wv",
                    checkpoint_name(layer, "self_attention/multihead_attention/v/kernel"),
                ),
                (
                    f"{prefix}.wo",
                    checkpoint_name(
                        layer,
                        "self_attention/multihead_attention/output_transform/kernel",
                    ),
                ),
                (
                    f"{prefix}.ffn_ln_scale",
                    checkpoint_name(
                        layer,
                        "ffn/layer_prepostprocess/layer_norm/layer_norm_scale",
                    ),
                ),
                (
                    f"{prefix}.ffn_ln_bias",
                    checkpoint_name(
                        layer,
                        "ffn/layer_prepostprocess/layer_norm/layer_norm_bias",
                    ),
                ),
                (f"{prefix}.w1", checkpoint_name(layer, "ffn/conv1/kernel")),
                (f"{prefix}.b1", checkpoint_name(layer, "ffn/conv1/bias")),
                (f"{prefix}.w2", checkpoint_name(layer, "ffn/conv2/kernel")),
                (f"{prefix}.b2", checkpoint_name(layer, "ffn/conv2/bias")),
            ]
        )

    names.extend(
        [
            (
                "final_ln_scale",
                "transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_scale",
            ),
            (
                "final_ln_bias",
                "transformer/body/decoder/layer_prepostprocess/layer_norm/layer_norm_bias",
            ),
        ]
    )
    return names


def write_tensor(binary_file, manifest: dict, public_name: str, array: np.ndarray) -> None:
    array = np.asarray(array, dtype="<f4", order="C")
    offset = binary_file.tell()
    array.tofile(binary_file)
    manifest["tensors"][public_name] = {
        "shape": list(array.shape),
        "offset": offset,
        "bytes": int(array.nbytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PREFIX)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = tf.train.load_checkpoint(str(checkpoint))
    shape_map = reader.get_variable_to_shape_map()
    manifest = {
        "format": "music-transformer-manual-v1",
        "dtype": "float32",
        "weights": "weights.bin",
        "config": {
            "vocabSize": VOCAB_SIZE,
            "hiddenSize": HIDDEN_SIZE,
            "filterSize": FILTER_SIZE,
            "numHeads": NUM_HEADS,
            "headSize": HIDDEN_SIZE // NUM_HEADS,
            "numLayers": NUM_LAYERS,
            "eosId": 1,
        },
        "tensors": {},
    }

    with (output_dir / "weights.bin").open("wb") as binary_file:
        embedding = None
        for public_name, ckpt_name in tensor_names():
            if ckpt_name not in shape_map:
                raise KeyError(f"Missing checkpoint tensor: {ckpt_name}")
            value = reader.get_tensor(ckpt_name)
            write_tensor(binary_file, manifest, public_name, value)
            if public_name == "embedding":
                embedding = value

        if embedding is None:
            raise RuntimeError("Embedding tensor was not exported")
        write_tensor(binary_file, manifest, "embedding_t", np.transpose(embedding))

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_mb = (output_dir / "weights.bin").stat().st_size / (1024 * 1024)
    print(f"Exported {len(manifest['tensors'])} tensors to {output_dir}")
    print(f"weights.bin: {total_mb:.1f} MiB")


if __name__ == "__main__":
    main()
