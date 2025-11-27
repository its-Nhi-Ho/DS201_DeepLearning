import torch
from torch import nn
from torch.nn import functional as F

class LSTMClassifier(nn.Module):
  def __init__(self, vocab_size: int, n_labels: int, hidden_size: int = 256, emb_dim: int = 256):
    super().__init__()

    self.embedding = nn.Embedding(
        num_embeddings = vocab_size,
        embedding_dim = emb_dim,
        padding_idx = 0
    )

    self.lstm = nn.LSTM(
        input_size = emb_dim,
        hidden_size = hidden_size,
        num_layers = 5,
        batch_first = True
    )

    self.fc = nn.Linear(hidden_size, n_labels)

  def forward(self, input_ids: torch.Tensor):
    x = self.embedding(input_ids)

    output, (h_n, c_n) = self.lstm(x)

    last_hidden = h_n[-1]

    logits = self.fc(last_hidden)
    return logits
