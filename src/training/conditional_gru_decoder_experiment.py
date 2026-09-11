from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

from src.model.conditional_gru_decoder import ConditionalGRUCourseDecoder, ConditionalGRUDecoderConfig
from src.training.tiny_gru_experiment import metric_at_k


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts" / "conditional_gru_decoder_experiment"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
START_TOKEN = "<START>"
SEED = 42
MIN_SEQUENCE_LEN = 2
TOP_KS = (1, 3, 5, 10)
STEP_FEATURE_COLUMNS = [
    "day_index",
    "slot_index",
    "absolute_step_index",
    "remaining_poi_count",
    "remaining_days",
    "is_day_start",
]
EXTRA_STATIC_COLUMNS = ["desired_poi_count", "poi_per_day"]


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_content_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True)


def load_course_input(input_path: Path, sequence_path: Path) -> tuple[pd.DataFrame, dict]:
    input_df = pd.read_csv(input_path, encoding="utf-8-sig")
    seq_df = pd.read_csv(sequence_path, encoding="utf-8-sig")
    coverage = {
        "input_rows": len(input_df),
        "input_trips": int(input_df["trip_id"].nunique()),
        "sequence_rows": len(seq_df),
        "sequence_trips": int(seq_df["travel_id"].nunique()),
        "content_rows": int(seq_df["CONTENT_ID"].notna().sum()),
        "content_trips": int(seq_df.loc[seq_df["CONTENT_ID"].notna(), "travel_id"].nunique()),
    }

    matched = seq_df.loc[seq_df["CONTENT_ID"].notna()].copy()
    matched["CONTENT_ID"] = normalize_content_id(matched["CONTENT_ID"])
    matched["day_index"] = pd.to_numeric(matched["day_index"], errors="coerce").fillna(1).astype(int)
    matched["visit_order"] = pd.to_numeric(matched["visit_order"], errors="coerce").fillna(1).astype(int)
    matched = matched.sort_values(["travel_id", "day_index", "visit_order"], kind="mergesort")
    matched["slot_index"] = matched.groupby(["travel_id", "day_index"]).cumcount() + 1
    matched["absolute_step_index"] = matched.groupby("travel_id").cumcount() + 1

    grouped = matched.groupby("travel_id", sort=False).apply(records_for_trip, include_groups=False).reset_index(name="course_records")
    grouped["content_sequence"] = grouped["course_records"].map(lambda rows: [row["content_id"] for row in rows])
    grouped["step_records"] = grouped["course_records"].map(lambda rows: [{key: row[key] for key in STEP_FEATURE_COLUMNS} for row in rows])
    grouped["sequence_len"] = grouped["content_sequence"].map(len)
    grouped = grouped.loc[grouped["sequence_len"] >= MIN_SEQUENCE_LEN].copy()

    model_input = input_df.merge(
        grouped[["travel_id", "content_sequence", "step_records", "sequence_len"]],
        left_on="trip_id",
        right_on="travel_id",
        how="inner",
        validate="one_to_one",
    )
    model_input["trip_days"] = pd.to_numeric(model_input["trip_days"], errors="coerce").fillna(1).clip(lower=1).astype(int)
    model_input["desired_poi_count"] = model_input["sequence_len"].astype(int)
    model_input["poi_per_day"] = np.ceil(model_input["desired_poi_count"] / model_input["trip_days"]).clip(lower=1).astype(int)
    model_input["step_records"] = model_input.apply(adjust_remaining_days, axis=1)
    return model_input.reset_index(drop=True), coverage


def records_for_trip(group: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for row in group.itertuples(index=False):
        rows.append(
            {
                "content_id": str(row.CONTENT_ID),
                "day_index": int(row.day_index),
                "slot_index": int(row.slot_index),
                "absolute_step_index": int(row.absolute_step_index),
                "remaining_poi_count": 0,
                "remaining_days": 0,
                "is_day_start": int(row.slot_index == 1),
            }
        )
    total = len(rows)
    for idx, row in enumerate(rows):
        row["remaining_poi_count"] = total - idx
    return rows


def adjust_remaining_days(row: pd.Series) -> list[dict[str, Any]]:
    trip_days = max(int(row["trip_days"]), 1)
    adjusted = []
    for record in row["step_records"]:
        item = dict(record)
        item["remaining_days"] = max(trip_days - int(item["day_index"]) + 1, 0)
        adjusted.append(item)
    return adjusted


def split_input(model_input: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_trips, temp_trips = train_test_split(model_input["trip_id"], test_size=0.30, random_state=SEED, shuffle=True)
    valid_trips, test_trips = train_test_split(temp_trips, test_size=0.50, random_state=SEED, shuffle=True)
    train_df = model_input[model_input["trip_id"].isin(set(train_trips))].reset_index(drop=True)
    valid_df = model_input[model_input["trip_id"].isin(set(valid_trips))].reset_index(drop=True)
    test_df = model_input[model_input["trip_id"].isin(set(test_trips))].reset_index(drop=True)
    return train_df, valid_df, test_df, {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)}


def build_vocab(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    train_content_ids = sorted({cid for seq in train_df["content_sequence"] for cid in seq})
    content_id_to_token = {PAD_TOKEN: 0, UNK_TOKEN: 1, START_TOKEN: 2}
    content_id_to_token.update({content_id: idx + 3 for idx, content_id in enumerate(train_content_ids)})
    token_to_content_id = {idx: content_id for content_id, idx in content_id_to_token.items()}
    return content_id_to_token, token_to_content_id


def build_static_feature_encoder(input_df: pd.DataFrame, train_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], int]:
    base_feature_columns = [column for column in input_df.columns if column != "trip_id"]
    static_feature_columns = [*base_feature_columns, *EXTRA_STATIC_COLUMNS]
    numeric_features = [
        "trip_days",
        "has_child",
        "has_elderly",
        "has_disabled",
        "companion_count",
        "p0_age",
        "p1_age",
        "desired_poi_count",
        "poi_per_day",
    ]
    categorical_features = [column for column in static_feature_columns if column not in numeric_features]
    encoder = ColumnTransformer(
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
    encoder.fit(train_df[static_feature_columns])
    static_feature_dim = int(encoder.transform(train_df[static_feature_columns]).shape[1])
    return encoder, static_feature_columns, static_feature_dim


def build_step_feature_encoder(train_df: pd.DataFrame) -> tuple[StandardScaler, int]:
    step_rows = [record for records in train_df["step_records"] for record in records]
    step_df = pd.DataFrame(step_rows, columns=STEP_FEATURE_COLUMNS)
    encoder = StandardScaler()
    encoder.fit(step_df[STEP_FEATURE_COLUMNS].astype(np.float32))
    return encoder, len(STEP_FEATURE_COLUMNS)


def encode_course_items(
    df: pd.DataFrame,
    static_encoder: ColumnTransformer,
    static_feature_columns: list[str],
    step_encoder: StandardScaler,
    content_id_to_token: dict[str, int],
) -> list[dict]:
    static_features = static_encoder.transform(df[static_feature_columns]).astype(np.float32)
    items = []
    for row_idx, row in df.reset_index(drop=True).iterrows():
        tokens = [content_id_to_token.get(str(content_id), content_id_to_token[UNK_TOKEN]) for content_id in row["content_sequence"]]
        previous_tokens = [content_id_to_token[START_TOKEN], *tokens[:-1]]
        labels = [token if token > content_id_to_token[START_TOKEN] else content_id_to_token[PAD_TOKEN] for token in tokens]
        step_df = pd.DataFrame(row["step_records"], columns=STEP_FEATURE_COLUMNS)
        step_features = step_encoder.transform(step_df[STEP_FEATURE_COLUMNS].astype(np.float32)).astype(np.float32)
        items.append(
            {
                "trip_id": row["trip_id"],
                "static_features": static_features[row_idx],
                "previous_tokens": previous_tokens,
                "labels": labels,
                "step_features": step_features,
                "content_tokens": tokens,
                "trip_days": int(row["trip_days"]),
                "desired_poi_count": int(row["desired_poi_count"]),
            }
        )
    return items


class CourseDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_course_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    static_features = torch.tensor(np.stack([item["static_features"] for item in batch]), dtype=torch.float32)
    lengths = torch.tensor([len(item["labels"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    step_dim = int(batch[0]["step_features"].shape[1])
    previous_tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.zeros((len(batch), max_len), dtype=torch.long)
    step_features = torch.zeros((len(batch), max_len, step_dim), dtype=torch.float32)
    for idx, item in enumerate(batch):
        length = len(item["labels"])
        previous_tokens[idx, :length] = torch.tensor(item["previous_tokens"], dtype=torch.long)
        labels[idx, :length] = torch.tensor(item["labels"], dtype=torch.long)
        step_features[idx, :length, :] = torch.tensor(item["step_features"], dtype=torch.float32)
    return {
        "static_features": static_features,
        "previous_tokens": previous_tokens,
        "step_features": step_features,
        "labels": labels,
        "lengths": lengths,
    }


def make_loader(items: list[dict], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(CourseDataset(items), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_course_batch)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    total_steps = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(is_train):
            logits = model(batch["static_features"], batch["previous_tokens"], batch["step_features"])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["labels"].reshape(-1))
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        valid_steps = int((batch["labels"] != 0).sum().item())
        total_loss += float(loss.item()) * valid_steps
        total_steps += valid_steps
    return total_loss / max(total_steps, 1)


@torch.no_grad()
def predict_teacher_forcing(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    scores = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        logits = model(batch["static_features"], batch["previous_tokens"], batch["step_features"])
        mask = batch["labels"] != 0
        labels.append(batch["labels"][mask].detach().cpu().numpy())
        scores.append(logits[mask].detach().cpu().numpy())
    return np.concatenate(labels), np.concatenate(scores)


@torch.no_grad()
def generate_tokens(
    model: ConditionalGRUCourseDecoder,
    item: dict,
    device: torch.device,
    special_token_ids: tuple[int, ...],
    duplicate_masking: bool = True,
) -> tuple[list[int], list[float]]:
    model.eval()
    static_features = torch.tensor(item["static_features"][None, :], dtype=torch.float32, device=device)
    step_features = torch.tensor(item["step_features"], dtype=torch.float32, device=device)
    hidden = model.initial_hidden(static_features)
    current_token = torch.tensor([special_token_ids[2]], dtype=torch.long, device=device)
    generated: list[int] = []
    scores: list[float] = []
    for step_idx in range(int(item["desired_poi_count"])):
        logits, hidden = model.decode_step(current_token, hidden, step_features[step_idx : step_idx + 1])
        masked = logits[0].clone()
        masked[list(special_token_ids)] = -1e9
        if duplicate_masking and generated:
            masked[generated] = -1e9
        if bool(torch.isneginf(masked).all()) or float(masked.max().item()) <= -1e8:
            masked = logits[0].clone()
            masked[list(special_token_ids)] = -1e9
        selected = int(torch.argmax(masked).item())
        generated.append(selected)
        scores.append(float(masked[selected].item()))
        current_token = torch.tensor([selected], dtype=torch.long, device=device)
    return generated, scores


def route_generation_metrics(
    model: ConditionalGRUCourseDecoder,
    items: list[dict],
    device: torch.device,
    special_token_ids: tuple[int, ...],
) -> dict[str, float]:
    overlaps = []
    position_hits = []
    duplicate_rates = []
    special_rates = []
    length_hits = []
    day_start_hits = []
    latencies = []
    for item in items:
        started = time.perf_counter()
        generated, _ = generate_tokens(model, item, device, special_token_ids)
        latencies.append(time.perf_counter() - started)
        target = [int(token) for token in item["content_tokens"] if int(token) not in special_token_ids]
        target_set = set(target)
        generated_set = set(generated)
        overlaps.append(len(target_set & generated_set) / max(len(target_set), 1))
        length_hits.append(float(len(generated) == int(item["desired_poi_count"])))
        position_hits.extend(float(g == t) for g, t in zip(generated, target))
        duplicate_rates.append((len(generated) - len(set(generated))) / max(len(generated), 1))
        special_rates.append(sum(1 for token in generated if token in special_token_ids) / max(len(generated), 1))
        for idx, record in enumerate(item["step_features_raw"]):
            if int(record["is_day_start"]) == 1 and idx < len(target):
                day_start_hits.append(float(generated[idx] == target[idx]))
    return {
        "route_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "position_accuracy": float(np.mean(position_hits)) if position_hits else 0.0,
        "duplicate_poi_rate": float(np.mean(duplicate_rates)) if duplicate_rates else 0.0,
        "special_token_rate": float(np.mean(special_rates)) if special_rates else 0.0,
        "route_length_correct": float(np.mean(length_hits)) if length_hits else 0.0,
        "day_start_position_accuracy": float(np.mean(day_start_hits)) if day_start_hits else 0.0,
        "avg_generation_latency_ms": float(np.mean(latencies) * 1000.0) if latencies else 0.0,
        "p95_generation_latency_ms": float(np.quantile(latencies, 0.95) * 1000.0) if latencies else 0.0,
    }


def attach_raw_step_records(items: list[dict], df: pd.DataFrame) -> list[dict]:
    by_trip_id = {row["trip_id"]: row["step_records"] for _, row in df.iterrows()}
    output = []
    for item in items:
        next_item = dict(item)
        next_item["step_features_raw"] = by_trip_id[item["trip_id"]]
        output.append(next_item)
    return output


def train_one_config(
    config: ConditionalGRUDecoderConfig,
    train_items: list[dict],
    valid_items: list[dict],
    vocab_size: int,
    static_feature_dim: int,
    step_feature_dim: int,
    special_token_ids: tuple[int, ...],
    device: torch.device,
) -> tuple[ConditionalGRUCourseDecoder, dict]:
    seed_everything(SEED)
    model = ConditionalGRUCourseDecoder(
        vocab_size=vocab_size,
        static_feature_dim=static_feature_dim,
        step_feature_dim=step_feature_dim,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        special_token_ids=special_token_ids,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_loader = make_loader(train_items, config.batch_size, shuffle=True)
    valid_loader = make_loader(valid_items, config.batch_size, shuffle=False)
    best_state = None
    best_recall = -1.0
    history = []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        valid_loss = run_epoch(model, valid_loader, device)
        labels, scores = predict_teacher_forcing(model, valid_loader, device)
        valid_metrics = metric_at_k(labels, scores, TOP_KS)
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


def parse_configs(raw_configs: list[str]) -> list[ConditionalGRUDecoderConfig]:
    return [ConditionalGRUDecoderConfig(**json.loads(raw)) for raw in raw_configs]


def write_report(path: Path, metrics: dict) -> None:
    rows = []
    for experiment in metrics["experiments"]:
        config = experiment["config"]
        test = experiment["teacher_forcing_test"]
        route = experiment["greedy_route_test"]
        rows.append(
            "| h={hidden} | {r10:.4f} | {mrr10:.4f} | {overlap:.4f} | {pos:.4f} | {dup:.4f} | {lat:.2f} |".format(
                hidden=config["hidden_dim"],
                r10=test["recall@10"],
                mrr10=test["mrr@10"],
                overlap=route["route_overlap"],
                pos=route["position_accuracy"],
                dup=route["duplicate_poi_rate"],
                lat=route["avg_generation_latency_ms"],
            )
        )
    report = [
        "# Conditional GRU Course Decoder",
        "",
        "This experiment generates a fixed-length CONTENT_ID sequence from user/trip features.",
        "",
        "## Metrics",
        "",
        "| model | teacher recall@10 | teacher mrr@10 | route overlap | position accuracy | duplicate rate | avg latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Notes",
        "",
        "- Teacher forcing metrics are step-level next-token ranking metrics.",
        "- Greedy route metrics use autoregressive generation with duplicate masking.",
        "- The v1 decoder controls length with desired_poi_count and does not use EOS.",
    ]
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a length-conditioned conditional GRU course decoder.")
    parser.add_argument("--input-path", type=Path, default=DATA_DIR / "total_input.csv")
    parser.add_argument("--sequence-path", type=Path, default=DATA_DIR / "total_travel_seq_with_contentid.csv")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(args.input_path, encoding="utf-8-sig")
    model_input, coverage = load_course_input(args.input_path, args.sequence_path)
    train_df, valid_df, test_df, split_sizes = split_input(model_input)
    content_id_to_token, token_to_content_id = build_vocab(train_df)
    special_token_ids = (
        content_id_to_token[PAD_TOKEN],
        content_id_to_token[UNK_TOKEN],
        content_id_to_token[START_TOKEN],
    )
    static_encoder, static_feature_columns, static_feature_dim = build_static_feature_encoder(input_df, train_df)
    step_encoder, step_feature_dim = build_step_feature_encoder(train_df)

    train_items = encode_course_items(train_df, static_encoder, static_feature_columns, step_encoder, content_id_to_token)
    valid_items = encode_course_items(valid_df, static_encoder, static_feature_columns, step_encoder, content_id_to_token)
    test_items = encode_course_items(test_df, static_encoder, static_feature_columns, step_encoder, content_id_to_token)
    train_items = attach_raw_step_records(train_items, train_df)
    valid_items = attach_raw_step_records(valid_items, valid_df)
    test_items = attach_raw_step_records(test_items, test_df)
    if args.smoke:
        train_items = train_items[:128]
        valid_items = valid_items[:64]
        test_items = test_items[:64]

    configs = parse_configs(args.config) if args.config else [
        ConditionalGRUDecoderConfig(embedding_dim=16, hidden_dim=48, dropout=0.2, learning_rate=1e-3, batch_size=64, epochs=8),
        ConditionalGRUDecoderConfig(embedding_dim=16, hidden_dim=64, dropout=0.2, learning_rate=1e-3, batch_size=64, epochs=8),
    ]
    if args.smoke:
        configs = [ConditionalGRUDecoderConfig(embedding_dim=16, hidden_dim=48, dropout=0.2, learning_rate=1e-3, batch_size=32, epochs=1)]

    experiment_results = []
    best_model = None
    best_result = None
    for config in configs:
        model, result = train_one_config(
            config=config,
            train_items=train_items,
            valid_items=valid_items,
            vocab_size=len(content_id_to_token),
            static_feature_dim=static_feature_dim,
            step_feature_dim=step_feature_dim,
            special_token_ids=special_token_ids,
            device=device,
        )
        test_loader = make_loader(test_items, config.batch_size, shuffle=False)
        test_labels, test_scores = predict_teacher_forcing(model, test_loader, device)
        result["teacher_forcing_test"] = metric_at_k(test_labels, test_scores, TOP_KS)
        result["greedy_route_test"] = route_generation_metrics(model, test_items, device, special_token_ids)
        experiment_results.append(result)
        if best_result is None or result["best_valid_recall@10"] > best_result["best_valid_recall@10"]:
            best_model = model
            best_result = result

    assert best_model is not None
    assert best_result is not None
    checkpoint = {
        "model_state_dict": best_model.state_dict(),
        "model_class": "ConditionalGRUCourseDecoder",
        "vocab_size": len(content_id_to_token),
        "static_feature_dim": static_feature_dim,
        "step_feature_dim": step_feature_dim,
        "config": best_result["config"],
        "pad_token_id": content_id_to_token[PAD_TOKEN],
        "unk_token_id": content_id_to_token[UNK_TOKEN],
        "start_token_id": content_id_to_token[START_TOKEN],
        "special_token_ids": special_token_ids,
        "static_feature_columns": static_feature_columns,
        "step_feature_columns": STEP_FEATURE_COLUMNS,
    }
    torch.save(checkpoint, artifact_dir / "best_conditional_gru_decoder.pt")
    with open(artifact_dir / "content_id_vocab.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "content_id_to_token": content_id_to_token,
                "token_to_content_id": token_to_content_id,
                "pad_token": PAD_TOKEN,
                "unk_token": UNK_TOKEN,
                "start_token": START_TOKEN,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    with open(artifact_dir / "static_feature_encoder.pkl", "wb") as fp:
        pickle.dump(static_encoder, fp)
    with open(artifact_dir / "step_feature_encoder.pkl", "wb") as fp:
        pickle.dump(step_encoder, fp)

    metrics = {
        "coverage": coverage,
        "usable_trips": len(model_input),
        "split_sizes": split_sizes,
        "sample_sizes": {"train": len(train_items), "valid": len(valid_items), "test": len(test_items)},
        "experiments": experiment_results,
        "best_config": best_result["config"],
    }
    with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    with open(artifact_dir / "train_config.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "seed": SEED,
                "min_sequence_len": MIN_SEQUENCE_LEN,
                "top_ks": TOP_KS,
                "input_path": str(args.input_path.resolve().relative_to(ROOT)),
                "sequence_path": str(args.sequence_path.resolve().relative_to(ROOT)),
                "best_config": best_result["config"],
                "static_feature_columns": static_feature_columns,
                "step_feature_columns": STEP_FEATURE_COLUMNS,
                "special_token_ids": special_token_ids,
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    write_report(artifact_dir / "experiment_report.md", metrics)
    pt_size = (artifact_dir / "best_conditional_gru_decoder.pt").stat().st_size
    print(json.dumps({"best_config": best_result["config"], "pt_bytes": pt_size, "best_test": best_result["teacher_forcing_test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
