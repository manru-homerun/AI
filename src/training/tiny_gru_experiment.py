from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.model.gru_content import GRUContentConfig, GRUContentRecommender


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "tiny_gru_onnx_experiment"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEED = 42
MIN_SEQUENCE_LEN = 2
MAX_SEQUENCE_LEN = 20
TOP_KS = (1, 3, 5, 10)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_content_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def load_model_input(input_path: Path, sequence_path: Path) -> tuple[pd.DataFrame, dict]:
    input_df = pd.read_csv(input_path, encoding="utf-8-sig")
    seq_df = pd.read_csv(sequence_path, encoding="utf-8-sig")
    coverage = {
        "input_rows": len(input_df),
        "input_trips": input_df["trip_id"].nunique(),
        "sequence_rows": len(seq_df),
        "sequence_trips": seq_df["travel_id"].nunique(),
        "content_rows": int(seq_df["CONTENT_ID"].notna().sum()),
        "content_trips": int(seq_df.loc[seq_df["CONTENT_ID"].notna(), "travel_id"].nunique()),
    }

    matched_seq = seq_df.loc[seq_df["CONTENT_ID"].notna()].copy()
    matched_seq["CONTENT_ID"] = normalize_content_id(matched_seq["CONTENT_ID"])
    matched_seq = matched_seq.sort_values(["travel_id", "day_index", "visit_order"], kind="mergesort")
    trip_sequences = (
        matched_seq.groupby("travel_id")["CONTENT_ID"]
        .apply(lambda values: [str(value) for value in values if pd.notna(value)])
        .reset_index(name="content_sequence")
    )
    trip_sequences["sequence_len"] = trip_sequences["content_sequence"].map(len)
    trip_sequences = trip_sequences.loc[trip_sequences["sequence_len"] >= MIN_SEQUENCE_LEN].copy()

    model_input = input_df.merge(
        trip_sequences,
        left_on="trip_id",
        right_on="travel_id",
        how="inner",
        validate="one_to_one",
    )
    return model_input, coverage


def split_input(model_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_trips, temp_trips = train_test_split(
        model_input["trip_id"], test_size=0.30, random_state=SEED, shuffle=True
    )
    valid_trips, test_trips = train_test_split(temp_trips, test_size=0.50, random_state=SEED, shuffle=True)
    split_map = {"train": set(train_trips), "valid": set(valid_trips), "test": set(test_trips)}
    train_df = model_input[model_input["trip_id"].isin(split_map["train"])].reset_index(drop=True)
    valid_df = model_input[model_input["trip_id"].isin(split_map["valid"])].reset_index(drop=True)
    test_df = model_input[model_input["trip_id"].isin(split_map["test"])].reset_index(drop=True)
    return train_df, valid_df, test_df, {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}


def build_vocab(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    train_content_ids = sorted({cid for seq in train_df["content_sequence"] for cid in seq})
    content_id_to_token = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    content_id_to_token.update({content_id: idx + 2 for idx, content_id in enumerate(train_content_ids)})
    token_to_content_id = {idx: content_id for content_id, idx in content_id_to_token.items()}
    return content_id_to_token, token_to_content_id


def build_feature_encoder(input_df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], int]:
    numeric_features = ["trip_days", "has_child", "has_elderly", "has_disabled", "companion_count", "p0_age", "p1_age"]
    feature_columns = [column for column in input_df.columns if column != "trip_id"]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    feature_encoder = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )
    feature_encoder.fit(train_df[feature_columns])
    user_feature_dim = int(feature_encoder.transform(train_df[feature_columns]).shape[1])
    return feature_encoder, feature_columns, user_feature_dim


def make_next_place_samples(
    df: pd.DataFrame,
    feature_encoder: ColumnTransformer,
    feature_columns: list[str],
    content_id_to_token: dict[str, int],
) -> list[dict]:
    encoded_features = feature_encoder.transform(df[feature_columns]).astype(np.float32)
    samples: list[dict] = []
    for row_idx, row in df.reset_index(drop=True).iterrows():
        tokens = [content_id_to_token.get(str(content_id), content_id_to_token[UNK_TOKEN]) for content_id in row["content_sequence"]]
        for target_pos in range(1, len(tokens)):
            label = tokens[target_pos]
            if label == content_id_to_token[UNK_TOKEN]:
                continue
            prefix = tokens[max(0, target_pos - MAX_SEQUENCE_LEN) : target_pos]
            samples.append(
                {
                    "trip_id": row["trip_id"],
                    "user_features": encoded_features[row_idx],
                    "prefix": prefix,
                    "label": label,
                }
            )
    return samples


class TravelSequenceDataset(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    user_features = torch.tensor(np.stack([item["user_features"] for item in batch]), dtype=torch.float32)
    lengths = torch.tensor([len(item["prefix"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    sequences = torch.zeros((len(batch), max_len), dtype=torch.long)
    for idx, item in enumerate(batch):
        sequences[idx, : len(item["prefix"])] = torch.tensor(item["prefix"], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return {"user_features": user_features, "sequences": sequences, "lengths": lengths, "labels": labels}


def metric_at_k(labels: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = TOP_KS) -> dict[str, float]:
    metrics: dict[str, float] = {}
    order = np.argsort(-scores, axis=1)
    for k in ks:
        use_k = min(k, scores.shape[1])
        topk = order[:, :use_k]
        hits = topk == labels[:, None]
        metrics[f"recall@{k}"] = float(hits.any(axis=1).mean())
        reciprocal_ranks = []
        ndcgs = []
        for row_hits in hits:
            hit_positions = np.flatnonzero(row_hits)
            reciprocal_ranks.append(0.0 if len(hit_positions) == 0 else 1.0 / float(hit_positions[0] + 1))
            ndcgs.append(0.0 if len(hit_positions) == 0 else 1.0 / math.log2(float(hit_positions[0] + 2)))
        metrics[f"mrr@{k}"] = float(np.mean(reciprocal_ranks))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return metrics


def baseline_scores(train_samples: list[dict], eval_samples: list[dict], vocab_size: int) -> np.ndarray:
    counts = np.ones(vocab_size, dtype=np.float32) * 1e-6
    counts[0] = -np.inf
    counts[1] = -np.inf
    for sample in train_samples:
        counts[sample["label"]] += 1.0
    return np.repeat(counts[None, :], repeats=len(eval_samples), axis=0)


def make_loader(samples: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TravelSequenceDataset(samples), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_rows = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(is_train):
            logits = model(batch["user_features"], batch["sequences"], batch["lengths"])
            loss = loss_fn(logits, batch["labels"])
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        rows = batch["labels"].size(0)
        total_loss += float(loss.item()) * rows
        total_rows += rows
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores = []
    labels = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["user_features"], batch["sequences"], batch["lengths"])
        scores.append(logits.detach().cpu().numpy())
        labels.append(batch["labels"].detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


def train_one_config(
    config: GRUContentConfig,
    train_samples: list[dict],
    valid_samples: list[dict],
    vocab_size: int,
    user_feature_dim: int,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    seed_everything(SEED)
    model = GRUContentRecommender(
        vocab_size=vocab_size,
        user_feature_dim=user_feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(train_samples, config.batch_size, shuffle=True)
    valid_loader = make_loader(valid_samples, config.batch_size, shuffle=False)
    history = []
    best_state = None
    best_recall = -1.0
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        valid_loss = run_epoch(model, valid_loader, device)
        labels, scores = predict_scores(model, valid_loader, device)
        valid_metrics = metric_at_k(labels, scores)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "epoch_seconds": time.perf_counter() - started,
            **valid_metrics,
        }
        history.append(row)
        print(json.dumps({**asdict(config), **row}, ensure_ascii=False))
        if valid_metrics["recall@10"] > best_recall:
            best_recall = valid_metrics["recall@10"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"config": asdict(config), "history": history, "best_valid_recall@10": best_recall}


def parse_configs(raw_configs: list[str]) -> list[GRUContentConfig]:
    configs = []
    for raw in raw_configs:
        values = json.loads(raw)
        configs.append(GRUContentConfig(**values))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Tiny GRU next-POI experiments.")
    parser.add_argument("--input-path", type=Path, default=DATA_DIR / "total_input.csv")
    parser.add_argument("--sequence-path", type=Path, default=DATA_DIR / "total_travel_seq_with_contentid.csv")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help='JSON config. Example: {"embedding_dim":16,"hidden_dim":32,"epochs":8}',
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny one-epoch smoke test.")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_path, encoding="utf-8-sig")
    model_input, coverage = load_model_input(args.input_path, args.sequence_path)
    train_df, valid_df, test_df, split_sizes = split_input(model_input)
    content_id_to_token, token_to_content_id = build_vocab(train_df)
    feature_encoder, feature_columns, user_feature_dim = build_feature_encoder(input_df, train_df)

    train_samples = make_next_place_samples(train_df, feature_encoder, feature_columns, content_id_to_token)
    valid_samples = make_next_place_samples(valid_df, feature_encoder, feature_columns, content_id_to_token)
    test_samples = make_next_place_samples(test_df, feature_encoder, feature_columns, content_id_to_token)
    if args.smoke:
        train_samples = train_samples[:512]
        valid_samples = valid_samples[:256]
        test_samples = test_samples[:256]

    configs = parse_configs(args.config) if args.config else [
        GRUContentConfig(embedding_dim=16, hidden_dim=32, dropout=0.2, learning_rate=1e-3, batch_size=64, epochs=8),
        GRUContentConfig(embedding_dim=16, hidden_dim=48, dropout=0.2, learning_rate=1e-3, batch_size=64, epochs=8),
        GRUContentConfig(embedding_dim=16, hidden_dim=64, dropout=0.2, learning_rate=1e-3, batch_size=64, epochs=8),
    ]

    valid_labels = np.array([sample["label"] for sample in valid_samples])
    baseline_valid_metrics = metric_at_k(
        valid_labels,
        baseline_scores(train_samples, valid_samples, len(content_id_to_token)),
    )

    experiment_results = []
    best_model = None
    best_result = None
    for config in configs:
        model, result = train_one_config(
            config,
            train_samples,
            valid_samples,
            len(content_id_to_token),
            user_feature_dim,
            device,
        )
        experiment_results.append(result)
        if best_result is None or result["best_valid_recall@10"] > best_result["best_valid_recall@10"]:
            best_model = model
            best_result = result

    assert best_model is not None
    assert best_result is not None
    test_loader = make_loader(test_samples, best_result["config"]["batch_size"], shuffle=False)
    test_labels, test_scores = predict_scores(best_model, test_loader, device)
    gru_test_metrics = metric_at_k(test_labels, test_scores)
    baseline_test_metrics = metric_at_k(
        test_labels,
        baseline_scores(train_samples, test_samples, len(content_id_to_token)),
    )

    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "model_class": "GRUContentRecommender",
        "vocab_size": len(content_id_to_token),
        "user_feature_dim": user_feature_dim,
        "config": best_result["config"],
        "max_sequence_len": MAX_SEQUENCE_LEN,
        "pad_token_id": content_id_to_token[PAD_TOKEN],
        "unk_token_id": content_id_to_token[UNK_TOKEN],
        "feature_columns": feature_columns,
    }
    torch.save(checkpoint, artifact_dir / "best_tiny_gru_contentid.pt")
    with open(artifact_dir / "content_id_vocab.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "content_id_to_token": content_id_to_token,
                "token_to_content_id": token_to_content_id,
                "pad_token": PAD_TOKEN,
                "unk_token": UNK_TOKEN,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    with open(artifact_dir / "feature_encoder.pkl", "wb") as fp:
        pickle.dump(feature_encoder, fp)

    metrics = {
        "coverage": coverage,
        "usable_trips": len(model_input),
        "split_sizes": split_sizes,
        "sample_sizes": {"train": len(train_samples), "valid": len(valid_samples), "test": len(test_samples)},
        "baseline_valid": baseline_valid_metrics,
        "baseline_test": baseline_test_metrics,
        "tiny_gru_test": gru_test_metrics,
        "experiments": experiment_results,
    }
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    with open(artifact_dir / "train_config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "seed": SEED,
                "min_sequence_len": MIN_SEQUENCE_LEN,
                "max_sequence_len": MAX_SEQUENCE_LEN,
                "top_ks": TOP_KS,
                "input_path": str(args.input_path.resolve().relative_to(ROOT)),
                "sequence_path": str(args.sequence_path.resolve().relative_to(ROOT)),
                "best_config": best_result["config"],
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )

    pt_size = (artifact_dir / "best_tiny_gru_contentid.pt").stat().st_size
    print(json.dumps({"best_config": best_result["config"], "tiny_gru_test": gru_test_metrics, "pt_bytes": pt_size}, ensure_ascii=False))


if __name__ == "__main__":
    main()

