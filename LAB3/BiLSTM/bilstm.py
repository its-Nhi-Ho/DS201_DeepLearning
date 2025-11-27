import torch
from torch import nn
from torch.nn import functional as F
class BiLSTM(nn.Module):
  def __init__(self, vocab_size: int, n_labels:int, hidden_size: int = 256, emb_dim: int = 256):
    super(BiLSTM, self).__init__()

    self.embedding = nn.Embedding(
        num_embeddings = vocab_size,
        embedding_dim = emb_dim,
        padding_idx = 0
    )

    # Encodeer: 5 layer LSTM
    self.lstm = nn.LSTM(
        input_size = emb_dim,
        hidden_size = hidden_size,
        num_layers = 5,
        batch_first = True,
        bidirectional = True
    )

    self.fc = nn.Linear(hidden_size * 2, n_labels)


  def forward(self, input_ids: torch.Tensor, lengths):
      x = self.embedding(input_ids)
      # Nếu lengths không truyền vào thì tự tính
      if lengths is None:
          lengths = (input_ids != 0).sum(dim=1).cpu()
      packed = nn.utils.rnn.pack_padded_sequence(
          x, lengths=lengths, batch_first=True, enforce_sorted=False
      )
      # LSTM
      packed_output, _ = self.lstm(packed)

      # Unpack
      output, _ = nn.utils.rnn.pad_packed_sequence(
          packed_output, batch_first=True
      )

      # output: (batch, seq, hidden*2)
      # Final logits
      logits = self.fc(output)
      return logits
