"""LSTM-based sentiment classifier."""

import torch
import torch.nn as nn

from preprocess import PAD_IDX


class SentimentLSTM(nn.Module):
    """Embedding -> LSTM -> Dropout -> Linear.

    The final hidden state of the LSTM summarises the whole review and is
    projected to two logits (negative / positive).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)            # (batch, seq_len, embed_dim)
        _, (hidden, _) = self.lstm(embedded)    # hidden: (1, batch, hidden_dim)
        last_hidden = self.dropout(hidden[-1])  # (batch, hidden_dim)
        return self.fc(last_hidden)             # (batch, output_dim)
