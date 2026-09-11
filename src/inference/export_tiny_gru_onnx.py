from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model.gru_content import GRUContentRecommender


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "tiny_gru_onnx_experiment"


def load_model(checkpoint_path: Path) -> tuple[GRUContentRecommender, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GRUContentRecommender(
        vocab_size=checkpoint["vocab_size"],
        user_feature_dim=checkpoint["user_feature_dim"],
        embedding_dim=checkpoint["config"]["embedding_dim"],
        hidden_dim=checkpoint["config"]["hidden_dim"],
        dropout=checkpoint["config"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Tiny GRU checkpoint to ONNX.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--checkpoint-name", default="best_tiny_gru_contentid.pt")
    parser.add_argument("--output-name", default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    model, checkpoint = load_model(artifact_dir / args.checkpoint_name)
    user_features = torch.zeros((1, checkpoint["user_feature_dim"]), dtype=torch.float32)
    sequences = torch.ones((1, min(3, checkpoint["max_sequence_len"])), dtype=torch.long)
    lengths = torch.tensor([sequences.size(1)], dtype=torch.long)

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
    except ImportError:
        max_abs_diff = None

    metadata = {
        "onnx_path": str(output_path.relative_to(ROOT)),
        "onnx_bytes": output_path.stat().st_size,
        "checkpoint": args.checkpoint_name,
        "opset": args.opset,
        "torch_onnx_max_abs_diff": max_abs_diff,
    }
    with open(artifact_dir / "onnx_export.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
