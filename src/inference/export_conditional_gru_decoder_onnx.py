from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.model.conditional_gru_decoder import (
    ConditionalGRUCourseDecoder,
    CourseDecoderStepOnnx,
    CourseUserEncoderOnnx,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "conditional_gru_decoder_experiment"


def load_model(checkpoint_path: Path) -> tuple[ConditionalGRUCourseDecoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = ConditionalGRUCourseDecoder(
        vocab_size=int(checkpoint["vocab_size"]),
        static_feature_dim=int(checkpoint["static_feature_dim"]),
        step_feature_dim=int(checkpoint["step_feature_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        special_token_ids=tuple(int(token_id) for token_id in checkpoint["special_token_ids"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Conditional GRU course decoder to ONNX.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--checkpoint-name", default="best_conditional_gru_decoder.pt")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    model, checkpoint = load_model(artifact_dir / args.checkpoint_name)
    user_encoder = CourseUserEncoderOnnx(model).eval()
    decoder_step = CourseDecoderStepOnnx(model).eval()

    static_features = torch.zeros((1, int(checkpoint["static_feature_dim"])), dtype=torch.float32)
    current_token = torch.tensor([int(checkpoint["start_token_id"])], dtype=torch.long)
    hidden_state = torch.zeros((1, int(checkpoint["config"]["hidden_dim"])), dtype=torch.float32)
    step_features = torch.zeros((1, int(checkpoint["step_feature_dim"])), dtype=torch.float32)

    user_encoder_path = artifact_dir / "course_user_encoder.onnx"
    decoder_step_path = artifact_dir / "course_decoder_step.onnx"
    torch.onnx.export(
        user_encoder,
        (static_features,),
        user_encoder_path,
        input_names=["static_features"],
        output_names=["hidden_state"],
        dynamic_axes={"static_features": {0: "batch"}, "hidden_state": {0: "batch"}},
        opset_version=args.opset,
    )
    torch.onnx.export(
        decoder_step,
        (current_token, hidden_state, step_features),
        decoder_step_path,
        input_names=["current_token", "hidden_state", "step_features"],
        output_names=["logits", "next_hidden_state"],
        dynamic_axes={
            "current_token": {0: "batch"},
            "hidden_state": {0: "batch"},
            "step_features": {0: "batch"},
            "logits": {0: "batch"},
            "next_hidden_state": {0: "batch"},
        },
        opset_version=args.opset,
    )

    try:
        import onnxruntime as ort

        user_session = ort.InferenceSession(str(user_encoder_path), providers=["CPUExecutionProvider"])
        decoder_session = ort.InferenceSession(str(decoder_step_path), providers=["CPUExecutionProvider"])
        ort_hidden = user_session.run(None, {"static_features": static_features.numpy()})[0]
        ort_logits, ort_next_hidden = decoder_session.run(
            None,
            {
                "current_token": current_token.numpy().astype(np.int64),
                "hidden_state": hidden_state.numpy(),
                "step_features": step_features.numpy(),
            },
        )
        with torch.no_grad():
            torch_hidden = user_encoder(static_features).numpy()
            torch_logits, torch_next_hidden = decoder_step(current_token, hidden_state, step_features)
        max_user_diff = float(np.max(np.abs(torch_hidden - ort_hidden)))
        max_logits_diff = float(np.max(np.abs(torch_logits.numpy() - ort_logits)))
        max_hidden_diff = float(np.max(np.abs(torch_next_hidden.numpy() - ort_next_hidden)))
    except ImportError:
        max_user_diff = None
        max_logits_diff = None
        max_hidden_diff = None

    metadata = {
        "user_encoder_onnx_path": str(user_encoder_path.relative_to(ROOT)),
        "decoder_step_onnx_path": str(decoder_step_path.relative_to(ROOT)),
        "user_encoder_onnx_bytes": user_encoder_path.stat().st_size,
        "decoder_step_onnx_bytes": decoder_step_path.stat().st_size,
        "checkpoint": args.checkpoint_name,
        "opset": args.opset,
        "torch_onnx_user_hidden_max_abs_diff": max_user_diff,
        "torch_onnx_step_logits_max_abs_diff": max_logits_diff,
        "torch_onnx_step_hidden_max_abs_diff": max_hidden_diff,
    }
    with open(artifact_dir / "onnx_export.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
