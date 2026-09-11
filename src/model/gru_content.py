from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GRUContentConfig:
    embedding_dim: int = 16
    hidden_dim: int = 32
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 8
    weight_decay: float = 1e-5


class GRUContentRecommender(nn.Module):
    """GRU recommender that is friendly to ONNX export.

    The notebook version used packed sequences. For a single-layer unidirectional
    GRU with right padding, gathering the output at lengths - 1 gives the same
    sequence context while exporting more cleanly to ONNX Runtime.
    """

    def __init__(
        self,
        vocab_size: int,
        user_feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.place_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.user_projection = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, user_features: torch.Tensor, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.place_embedding(sequences)
        gru_output, _ = self.gru(embedded)
        last_idx = torch.clamp(lengths - 1, min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, gru_output.size(2))
        sequence_context = torch.gather(gru_output, dim=1, index=gather_idx).squeeze(1)
        user_context = self.user_projection(user_features)
        logits = self.output(torch.cat([sequence_context, user_context], dim=1))
        logits[:, 0] = -1e9
        logits[:, 1] = -1e9
        return logits
