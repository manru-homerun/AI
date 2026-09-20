from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.inference.shared_next_poi import DEFAULT_ARTIFACT_DIR
from src.model.gru_content import GRUContentRecommender


ROOT = Path(__file__).resolve().parents[2]


def load_model(checkpoint_path: Path) -> tuple[GRUContentRecommender, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GRUContentRecommender(
        vocab_size=int(checkpoint["vocab_size"]),
        user_feature_dim=int(checkpoint["user_feature_dim"]),
        embedding_dim=int(checkpoint["config"]["embedding_dim"]),
        hidden_dim=int(checkpoint["config"]["hidden_dim"]),
        dropout=float(checkpoint["config"]["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export shared Next-POI GRU checkpoint to ONNX.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--checkpoint-name", default="best_shared_next_poi_gru.pt")
    parser.add_argument("--output-name", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    model, checkpoint = load_model(artifact_dir / args.checkpoint_name)
    user_features = torch.zeros((1, int(checkpoint["user_feature_dim"])), dtype=torch.float32)
    max_sequence_len = int(checkpoint["max_sequence_len"])
    bos_token_id = int(checkpoint["bos_token_id"])
    sequence_length = max(1, min(3, max_sequence_len))
    sequences = torch.full((1, sequence_length), bos_token_id, dtype=torch.long)
    lengths = torch.tensor([sequence_length], dtype=torch.long)

    output_path = artifact_dir / args.output_name
    torch.onnx.export(
        model,
        (user_features, sequences, lengths),
        output_path,
        input_names=["user_features", "sequences", "lengths"],
        output_names=["logits"],
        dynamic_axes={
            "user_features": {0: "batch"},
            "sequences": {0: "batch", 1: "sequence_length"},
            "lengths": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
    )

    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
        ort_logits = session.run(
            None,
            {
                "user_features": user_features.numpy(),
                "sequences": sequences.numpy().astype(np.int64),
                "lengths": lengths.numpy().astype(np.int64),
            },
        )[0]
        with torch.no_grad():
            torch_logits = model(user_features, sequences, lengths).numpy()
        max_abs_diff = float(np.max(np.abs(torch_logits - ort_logits)))
        output_shape = list(ort_logits.shape)
    except ImportError:
        max_abs_diff = None
        output_shape = None

    train_config = {
        "model_version": checkpoint["model_version"],
        "vocab_version": checkpoint["vocab_version"],
        "max_sequence_len": int(checkpoint["max_sequence_len"]),
        "course_pois_per_day": 6,
        "pad_token_id": int(checkpoint["pad_token_id"]),
        "bos_token_id": int(checkpoint["bos_token_id"]),
        "scalar_context_columns": list(checkpoint["scalar_context_columns"]),
        "preferred_area_column": checkpoint["preferred_area_column"],
        "missing_context_token": checkpoint["missing_context_token"],
        "primary_metric": checkpoint.get("primary_metric_name"),
        "selection_metric": checkpoint.get("selection_metric"),
        "report_ks": list(checkpoint.get("report_ks", [])),
        "service_top_k": checkpoint.get("service_top_k"),
        "model_config": checkpoint["config"],
        "vocab_size": int(checkpoint["vocab_size"]),
        "user_feature_dim": int(checkpoint["user_feature_dim"]),
    }
    with open(artifact_dir / "train_config.json", "w", encoding="utf-8") as fp:
        json.dump(train_config, fp, ensure_ascii=False, indent=2)

    metadata = {
        "onnx_path": str(output_path.relative_to(ROOT)),
        "onnx_bytes": output_path.stat().st_size,
        "checkpoint": args.checkpoint_name,
        "opset": args.opset,
        "torch_onnx_max_abs_diff": max_abs_diff,
        "onnx_output_shape": output_shape,
    }
    with open(artifact_dir / "onnx_export.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()

