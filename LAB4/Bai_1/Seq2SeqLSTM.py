import torch
from torch import nn
import random

class Seq2SeqLSTM(nn.Module):
    def __init__(self, d_model: int, n_encoder: int, n_decoder: int, dropout: float, vocab):
        super().__init__()
        self.vocab = vocab
        
        # 1. Embedding
        # Encoder embedding
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )
        
        # Decoder embedding (input size = 2*d_model để khớp với hidden state ghép từ encoder)
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=2 * d_model, 
            padding_idx=vocab.pad_idx
        )

        # 2. Encoder (Bidirectional = True)
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            batch_first=True,
            dropout=dropout if n_encoder > 1 else 0,
            bidirectional=True
        )

        # 3. Decoder (Bidirectional = False)
        # Hidden size = 2 * d_model (do ghép 2 chiều từ Encoder)
        self.decoder = nn.LSTM(
            input_size=2 * d_model,
            hidden_size=2 * d_model,
            num_layers=n_decoder,
            batch_first=True,
            dropout=dropout if n_decoder > 1 else 0,
            bidirectional=False
        )

        # 4. Output Head
        self.output_head = nn.Linear(
            in_features=2 * d_model,
            out_features=vocab.total_tgt_tokens
        )

    def _process_encoder_states(self, hidden, cell):
        # Lấy kích thước hiện tại
        num_layers_x2, batch_size, hidden_dim = hidden.shape
        num_layers = num_layers_x2 // 2
        
        # 1. Tách chiều forward và backward: (num_layers, 2, batch, hidden)
        hidden = hidden.view(num_layers, 2, batch_size, hidden_dim)
        cell = cell.view(num_layers, 2, batch_size, hidden_dim)
        
        # 2. Ghép (Concatenate) 2 chiều lại: (num_layers, batch, 2*hidden)
        # dim=2 tương ứng với chiều hidden_dim
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        return hidden, cell

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # Encoder
        embedded_src = self.src_embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        
        # Chuyển đổi hidden/cell state
        hidden, cell = self._process_encoder_states(hidden, cell)

        # Decoder
        embedded_tgt = self.tgt_embedding(tgt)
        decoder_outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
        
        # Output Projection
        logits = self.output_head(decoder_outputs)
        return logits

    def predict(self, src: torch.Tensor, max_len: int = 100):
        self.eval()
        device = src.device
    
        bs = src.shape[0] 

        # 1. Encoder Pass
        with torch.no_grad():
            embedded_src = self.src_embedding(src)
            _, (hidden, cell) = self.encoder(embedded_src)
            hidden, cell = self._process_encoder_states(hidden, cell)

        # 2. Chuẩn bị token đầu tiên <bos>
        # Giả sử self.vocab.bos_idx là index của <bos>
        input_token = torch.tensor([self.vocab.bos_idx] * bs, device=device).unsqueeze(1) # (Batch, 1)

        outputs = []
        
        # 3. Vòng lặp sinh từ (Autoregressive)
        for _ in range(max_len):
            embedded_input = self.tgt_embedding(input_token) # (Batch, 1, 2*Dim)
            
            # Decoder bước t
            decoder_output, (hidden, cell) = self.decoder(embedded_input, (hidden, cell))
            
            # Dự đoán từ tiếp theo
            logit = self.output_head(decoder_output.squeeze(1)) # (Batch, Vocab)
            pred_token = logit.argmax(dim=1) # (Batch)
            
            # Lưu kết quả
            outputs.append(pred_token.unsqueeze(1))
            
            # Input cho bước t+1 là output của bước t
            input_token = pred_token.unsqueeze(1)

        return torch.cat(outputs, dim=1)
