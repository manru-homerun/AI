from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ConditionalGRUDecoderConfig:
    embedding_dim: int = 16
    hidden_dim: int = 48
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 8
    weight_decay: float = 1e-5


class ConditionalGRUCourseDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        static_feature_dim: int,
        step_feature_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        special_token_ids: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.special_token_ids = tuple(int(token_id) for token_id in special_token_ids)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.user_encoder = nn.Sequential(
            nn.Linear(static_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gru_cell = nn.GRUCell(embedding_dim + step_feature_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def initial_hidden(self, static_features: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(static_features)

    def decode_step(
        self,
        current_token: torch.Tensor,
        hidden_state: torch.Tensor,
        step_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.token_embedding(current_token)
        decoder_input = torch.cat([embedded, step_features], dim=1)
        next_hidden = self.gru_cell(decoder_input, hidden_state)
        logits = self.output(next_hidden)
        for token_id in self.special_token_ids:
            logits[:, token_id] = -1e9
        return logits, next_hidden

    def forward(
        self,
        static_features: torch.Tensor,
        previous_tokens: torch.Tensor,
        step_features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.initial_hidden(static_features)
        logits_by_step = []
        for step_idx in range(previous_tokens.size(1)):
            logits, hidden = self.decode_step(
                previous_tokens[:, step_idx],
                hidden,
                step_features[:, step_idx, :],
            )
            logits_by_step.append(logits)
        return torch.stack(logits_by_step, dim=1)


class CourseUserEncoderOnnx(nn.Module):
    def __init__(self, model: ConditionalGRUCourseDecoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        return self.model.initial_hidden(static_features)


class CourseDecoderStepOnnx(nn.Module):
    def __init__(self, model: ConditionalGRUCourseDecoder) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        current_token: torch.Tensor,
        hidden_state: torch.Tensor,
        step_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.decode_step(current_token, hidden_state, step_features)
