from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model.gru_content import GRUContentRecommender
from src.training.tiny_gru_experiment import (
    DATA_DIR,
    DEFAULT_ARTIFACT_DIR,
    ROOT,
    TOP_KS,
    UNK_TOKEN,
    load_model_input,
    make_loader,
    make_next_place_samples,
    metric_at_k,
    predict_scores,
    split_input,
)


DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "tiny_gru_prefix_ablation"
CONDITIONS = (
    {"name": "full", "prefix_mode": "full", "prefix_last_n": None},
    {"name": "last-3", "prefix_mode": "last_n", "prefix_last_n": 3},
    {"name": "last-2", "prefix_mode": "last_n", "prefix_last_n": 2},
    {"name": "last-1", "prefix_mode": "last_n", "prefix_last_n": 1},
    {"name": "no-seq", "prefix_mode": "none", "prefix_last_n": None},
)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[GRUContentRecommender, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = GRUContentRecommender(
        vocab_size=int(checkpoint["vocab_size"]),
        user_feature_dim=int(checkpoint["user_feature_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_condition(
    model: GRUContentRecommender,
    test_df: pd.DataFrame,
    feature_encoder,
    feature_columns: list[str],
    content_id_to_token: dict[str, int],
    condition: dict,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], np.ndarray, dict]:
    samples = make_next_place_samples(
        test_df,
        feature_encoder,
        feature_columns,
        content_id_to_token,
        prefix_mode=str(condition["prefix_mode"]),
        prefix_last_n=condition["prefix_last_n"],
    )
    loader = make_loader(samples, batch_size=batch_size, shuffle=False)
    labels, scores = predict_scores(model, loader, device)
    metrics = metric_at_k(labels, scores, TOP_KS)
    prefix_lengths = np.array([len(sample["prefix"]) for sample in samples], dtype=np.int64)
    metrics.update(
        {
            "sample_count": int(len(samples)),
            "avg_prefix_len": float(prefix_lengths.mean()) if len(prefix_lengths) else 0.0,
            "min_prefix_len": int(prefix_lengths.min()) if len(prefix_lengths) else 0,
            "max_prefix_len": int(prefix_lengths.max()) if len(prefix_lengths) else 0,
        }
    )
    return samples, labels, metrics


def check_same_eval_set(reference_samples: list[dict], reference_labels: np.ndarray, samples: list[dict], labels: np.ndarray) -> None:
    if len(samples) != len(reference_samples):
        raise AssertionError(f"sample count mismatch: {len(samples)} != {len(reference_samples)}")
    if not np.array_equal(labels, reference_labels):
        raise AssertionError("label order mismatch between prefix conditions")
    reference_trip_ids = [sample["trip_id"] for sample in reference_samples]
    trip_ids = [sample["trip_id"] for sample in samples]
    if trip_ids != reference_trip_ids:
        raise AssertionError("trip order mismatch between prefix conditions")


def interpret(results: dict[str, dict]) -> str:
    full_recall = results["full"]["recall@10"]
    last1_recall = results["last-1"]["recall@10"]
    last3_recall = results["last-3"]["recall@10"]
    last2_recall = results["last-2"]["recall@10"]
    noseq_recall = results["no-seq"]["recall@10"]

    lines = []
    if full_recall - last1_recall >= 0.05:
        lines.append("- `full` is at least 5 percentage points above `last-1` on Recall@10, so longer prefixes appear useful.")
    else:
        lines.append("- `full` is less than 5 percentage points above `last-1` on Recall@10, so the long-prefix effect may be small.")

    if abs(full_recall - last3_recall) <= 0.02 or abs(full_recall - last2_recall) <= 0.02:
        lines.append("- `full` is within about 1-2 percentage points of `last-2` or `last-3`, suggesting the model mostly uses recent POIs.")
    else:
        lines.append("- `full` is more than 2 percentage points away from `last-2` and `last-3`, so recent POIs alone lose signal.")

    if abs(last1_recall - noseq_recall) <= 0.02:
        lines.append("- `last-1` and `no-seq` are close, so user/trip features may dominate over sequence history.")
    else:
        lines.append("- `last-1` differs from `no-seq`, so the immediately previous POI is still meaningful.")
    return "\n".join(lines)


def write_report(output_path: Path, results: dict[str, dict], metadata: dict) -> None:
    rows = []
    for condition in CONDITIONS:
        name = condition["name"]
        metrics = results[name]
        rows.append(
            "| {name} | {sample_count} | {avg_prefix_len:.2f} | {recall1:.4f} | {recall3:.4f} | {recall5:.4f} | {recall10:.4f} | {mrr10:.4f} | {ndcg10:.4f} |".format(
                name=name,
                sample_count=metrics["sample_count"],
                avg_prefix_len=metrics["avg_prefix_len"],
                recall1=metrics["recall@1"],
                recall3=metrics["recall@3"],
                recall5=metrics["recall@5"],
                recall10=metrics["recall@10"],
                mrr10=metrics["mrr@10"],
                ndcg10=metrics["ndcg@10"],
            )
        )

    report = [
        "# Tiny GRU Prefix Length Ablation",
        "",
        "This evaluation freezes the trained Tiny GRU checkpoint and changes only the test-time prefix length.",
        "",
        "## Metadata",
        "",
        f"- checkpoint: `{metadata['checkpoint_path']}`",
        f"- input_path: `{metadata['input_path']}`",
        f"- sequence_path: `{metadata['sequence_path']}`",
        f"- batch_size: `{metadata['batch_size']}`",
        "",
        "## Metrics",
        "",
        "| condition | samples | avg prefix len | recall@1 | recall@3 | recall@5 | recall@10 | mrr@10 | ndcg@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Interpretation",
        "",
        interpret(results),
        "",
        "## Notes",
        "",
        f"- `no-seq` uses one `{UNK_TOKEN}` token instead of an empty sequence to keep GRU inference stable.",
        "- `top_k` is a ranking metric cutoff, not a model input feature.",
    ]
    output_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tiny GRU with different prefix lengths.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--input-path", type=Path, default=DATA_DIR / "total_input.csv")
    parser.add_argument("--sequence-path", type=Path, default=DATA_DIR / "total_travel_seq_with_contentid.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(artifact_dir / "best_tiny_gru_contentid.pt", device)
    vocab = load_json(artifact_dir / "content_id_vocab.json")
    with open(artifact_dir / "feature_encoder.pkl", "rb") as fp:
        feature_encoder = pickle.load(fp)

    input_df = pd.read_csv(args.input_path, encoding="utf-8-sig")
    model_input, coverage = load_model_input(args.input_path, args.sequence_path)
    _, _, test_df, split_sizes = split_input(model_input)

    content_id_to_token = vocab["content_id_to_token"]
    feature_columns = list(checkpoint.get("feature_columns") or feature_encoder.feature_names_in_)
    missing_columns = [column for column in feature_columns if column not in input_df.columns]
    if missing_columns:
        raise ValueError(f"Missing feature columns in input data: {missing_columns}")

    batch_size = int(args.batch_size or checkpoint["config"].get("batch_size", 64))
    results: dict[str, dict] = {}
    reference_samples = None
    reference_labels = None
    for condition in CONDITIONS:
        samples, labels, metrics = evaluate_condition(
            model=model,
            test_df=test_df,
            feature_encoder=feature_encoder,
            feature_columns=feature_columns,
            content_id_to_token=content_id_to_token,
            condition=condition,
            batch_size=batch_size,
            device=device,
        )
        if reference_samples is None:
            reference_samples = samples
            reference_labels = labels
        else:
            assert reference_labels is not None
            check_same_eval_set(reference_samples, reference_labels, samples, labels)
        results[str(condition["name"])] = metrics

    metadata = {
        "checkpoint_path": str((artifact_dir / "best_tiny_gru_contentid.pt").relative_to(ROOT)),
        "input_path": str(args.input_path.resolve().relative_to(ROOT)),
        "sequence_path": str(args.sequence_path.resolve().relative_to(ROOT)),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "device": str(device),
        "batch_size": batch_size,
        "coverage": coverage,
        "split_sizes": split_sizes,
        "checkpoint_config": checkpoint["config"],
    }
    payload = {"metadata": metadata, "conditions": list(CONDITIONS), "results": results}
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    write_report(output_dir / "prefix_ablation_report.md", results, metadata)
    print(json.dumps({"results": results, "output_dir": metadata["output_dir"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
