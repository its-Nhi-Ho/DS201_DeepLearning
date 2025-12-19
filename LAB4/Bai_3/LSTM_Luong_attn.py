import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Seq2seqLSTM_Luong_attn(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_encoder: int,
                 n_decoder: int,
                 dropout: int,
                 vocab
    ):
        super().__init__()

        self.vocab = vocab
        self.d_model = d_model
        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.decoder_dim = 2 * d_model # Kích thước hidden state của Decoder (2*d_model do Encoder Bi-LSTM)

        # 1. ENCODER (Bi-LSTM, Output: 2*d_model)
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.total_src_tokens,
            embedding_dim=d_model,
            padding_idx=vocab.pad_idx
        )
        self.encoder = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_encoder,
            batch_first=True,
            dropout=dropout,
            bidirectional=True # Encoder là Bidirectional
        )

        # 2. DECODER
        # Luong Chuẩn: tgt_embedding = d_model
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens,
            embedding_dim=d_model, # Kích thước d_model
            padding_idx=vocab.pad_idx
        )

        # Luong Chuẩn: Decoder Input size chỉ cần d_model
        self.decoder = nn.LSTM(
            input_size=d_model, # Kích thước d_model (chỉ nhận embedding)
            hidden_size=self.decoder_dim, # 2*d_model
            num_layers=n_decoder,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # 3. ATTENTION (Luong General Attention: score(s_t, h_j) = s_t^T * W_a * h_j)
        # self.attn_weights đóng vai trò là ma trận W_a
        # Input: h_j (2*d_model), Output: W_a * h_j (2*d_model)
        self.attn_weights = nn.Linear(
            in_features=self.decoder_dim,
            out_features=self.decoder_dim,
            bias=False
        )

        # 4. CONTEXT-AWARE OUTPUT PROJECTION (s_t_tilde)
        # Input: [h_t_last_layer (2*d_model) ; context_vector (2*d_model)] -> 4*d_model
        # Output: 2*d_model
        self.context_projection = nn.Linear(
            in_features=4*d_model,
            out_features=self.decoder_dim
        )

        # 5. OUTPUT HEAD
        self.output_head = nn.Linear(
            in_features=self.decoder_dim, # 2*d_model
            out_features=vocab.total_tgt_tokens
        )

    def aligning(self, query: torch.Tensor, k_v: torch.Tensor) -> torch.Tensor:
        '''
        Tính Context Vector c_t theo Luong General Attention
        query: Trạng thái ẩn lớp cuối cùng của Decoder h_t (bs, 2*d_model)
        k_v: Các Annotation Vectors của Encoder h_j (bs, len, 2*d_model)
        '''
        # 1. Tính W_a * h_j
        attn_key = self.attn_weights(k_v) # (bs, len, 2*d_model)

        # 2. Tính Alignment Score: score = s_t^T * (W_a * h_j)
        query = query.unsqueeze(1) # (bs, 1, 2*d_model)
        a = torch.bmm(query, attn_key.transpose(1, 2)) # (bs, 1, len)

        # 3. Chuẩn hóa bằng Softmax -> Trọng số Attention (alpha_t)
        a = a.squeeze(1) # (bs, len)
        a = F.softmax(a, dim=1).unsqueeze(2) # (bs, len, 1)

        # 4. Tính Context Vector (Tổng có trọng số)
        context_vector = (a * k_v).sum(dim=1) # (bs, 2*d_model)

        return context_vector

    def forward_step(
        self,
        input_ids: torch.Tensor, # (bs, 1)
        enc_outputs: torch.Tensor, # (bs, len, 2*d_model)
        dec_states: Tuple[torch.Tensor, torch.Tensor] # (h_state, c_state)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        h_state, c_state = dec_states

        # 1. Chạy Decoder LSTM (Luong: Input chỉ là embedding)
        embedded_input = self.tgt_embedding(input_ids) # (bs, 1, d_model)
        _, (new_h_state, new_c_state) = self.decoder(embedded_input, (h_state, c_state))

        # 2. Tính Context Vector (c_t) TỪ h_t MỚI (new_h_state)
        new_last_h_state = new_h_state[-1] # (bs, 2*d_model)
        context_vector = self.aligning(new_last_h_state, enc_outputs) # (bs, 2*d_model)

        # 3. Tính Context-aware Output (s_t_tilde)
        # S_t_tilde = tanh(W_c [h_t_last_layer ; c_t])
        combined_output = torch.tanh(
            self.context_projection(
                torch.cat([new_last_h_state, context_vector], dim=-1) # (bs, 4*d_model)
            )
        ) # (bs, 2*d_model)

        return new_h_state.contiguous(), new_c_state.contiguous(), combined_output

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.train()
        embedded_x = self.src_embedding(x)
        bs, _, _ = embedded_x.shape

        # 1. Encoder
        enc_outputs, (enc_hn, enc_cn) = self.encoder(embedded_x)
        enc_hidden_states = enc_outputs

        # 2. Khởi tạo Trạng thái Decoder
        h_0 = enc_hn[-2:].transpose(0, 1).reshape(bs, -1)
        c_0 = enc_cn[-2:].transpose(0, 1).reshape(bs, -1)

        dec_h_state = h_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
        dec_c_state = c_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
        dec_states = (dec_h_state, dec_c_state)

        _, tgt_len = y.shape
        logits = []

        # 3. Decoding Loop (Teacher Forcing)
        for ith in range(tgt_len):
            y_ith = y[:, ith].unsqueeze(-1) # (bs, 1)

            dec_h_state, dec_c_state, combined_output = self.forward_step(
                y_ith,
                enc_hidden_states,
                dec_states
            )
            dec_states = (dec_h_state, dec_c_state) # Cập nhật trạng thái

            logit = self.output_head(combined_output) # (bs, total_tgt_tokens)
            logits.append(logit.unsqueeze(1))

        logits = torch.cat(logits, dim=1) # (bs, tgt_len, total_tgt_tokens)

        return logits

    def predict(self, x: torch.Tensor, max_len: int = 50):
        self.eval()
        with torch.no_grad():
            embedded_x = self.src_embedding(x)
            bs, _, _ = embedded_x.shape

            # 1. Encoder
            enc_outputs, (enc_hn, enc_cn) = self.encoder(embedded_x)
            enc_hidden_states = enc_outputs

            # 2. Khởi tạo Trạng thái Decoder
            h_0 = enc_hn[-2:].transpose(0, 1).reshape(bs, -1)
            c_0 = enc_cn[-2:].transpose(0, 1).reshape(bs, -1)
            dec_h_state = h_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
            dec_c_state = c_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
            dec_states = (dec_h_state, dec_c_state)

            y_ith = torch.zeros(bs, ).fill_(self.vocab.bos_idx).long().to(x.device).unsqueeze(-1)

            mark_eos = torch.zeros(bs, dtype=torch.bool).to(x.device)
            outputs = []

            for _ in range(max_len):
                dec_h_state, dec_c_state, combined_output = self.forward_step(
                    y_ith,
                    enc_hidden_states,
                    dec_states
                )
                dec_states = (dec_h_state, dec_c_state)

                logit = self.output_head(combined_output)
                y_ith = logit.argmax(dim=-1).long().unsqueeze(-1)

                mark_eos = mark_eos | (y_ith.squeeze(-1) == self.vocab.eos_idx)

                if all(mark_eos.tolist()):
                    break

                outputs.append(y_ith.squeeze(-1))

            if outputs:
                outputs = torch.stack(outputs, dim=1)
            else:
                outputs = torch.empty((bs, 0), dtype=torch.long, device=x.device)

            return outputs
