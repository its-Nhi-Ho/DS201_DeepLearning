import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Seq2seqLSTM_Bahdanau_attn(nn.Module):
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
        # Sửa: Embedding dim giảm về d_model (sẽ ghép với context 2*d_model)
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.total_tgt_tokens, 
            embedding_dim=d_model, 
            padding_idx=vocab.pad_idx
        )
        
        # Sửa: Decoder Input size = d_model (embedding) + 2*d_model (context vector) = 3*d_model
        self.decoder = nn.LSTM(
            input_size=3*d_model, 
            hidden_size=self.decoder_dim, # 2*d_model
            num_layers=n_decoder, 
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # 3. ATTENTION
        # Dùng Additive/Concat Attention (Bahdanau et al.)
        # Input: [h_t-1_last_layer (2*d_model) ; h_j (2*d_model)] -> 4*d_model
        self.attn_weights = nn.Linear(
            in_features=4*d_model,
            out_features=1, # output 1 scalar
            bias=False
        )

        # 4. CONTEXT-AWARE OUTPUT PROJECTION (S_t)
        # Tính S_t_tilde = tanh(W_c [c_t ; h_t_last_layer])
        # Input: [context_vector (2*d_model) ; h_t_last_layer (2*d_model)] -> 4*d_model
        # Output: 2*d_model
        self.context_projection = nn.Linear(
            in_features=4*d_model, 
            out_features=self.decoder_dim
        )
        
        # 5. OUTPUT HEAD
        # Output head nhận S_t_tilde (2*d_model)
        self.output_head = nn.Linear(
            in_features=self.decoder_dim, # 2*d_model
            out_features=vocab.total_tgt_tokens
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

    def aligning(self, query: torch.Tensor, k_v: torch.Tensor) -> torch.Tensor:
        '''
        Tính Context Vector c_t
        query: Trạng thái ẩn lớp cuối cùng của Decoder h_t-1 (bs, 2*d_model)
        k_v: Các Annotation Vectors của Encoder h_j (bs, len, 2*d_model)
        '''
        bs, l, _ = k_v.shape
        # (1) Mở rộng query để khớp với chiều dài câu nguồn (l)
        query_expanded = query.unsqueeze(1).repeat(1, l, 1) # (bs, len, 2*d_model)

        # (2) Tính Alignment Score (Additive/Concat)
        a = self.attn_weights(
            torch.cat([query_expanded, k_v], dim=-1) # (bs, len, 4*d_model) -> (bs, len, 1)
        ) 
        
        # (3) Chuẩn hóa bằng Softmax -> Trọng số Attention (alpha_t)
        a = F.softmax(a, dim=1) # (bs, len, 1)

        # (4) Tính Context Vector (Tổng có trọng số)
        context_vector = (a * k_v).sum(dim=1) # (bs, 2*d_model)

        return context_vector

    def forward_step(
        self, 
        input_ids: torch.Tensor, # (bs, 1)
        enc_outputs: torch.Tensor, # (bs, len, 2*d_model)
        dec_states: Tuple[torch.Tensor, torch.Tensor] # (h_state, c_state)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        h_state, c_state = dec_states

        # 1. Tính Context Vector (c_t) từ h_t-1 (lớp cuối cùng)
        last_h_state = h_state[-1] # (bs, 2*d_model)
        context_vector = self.aligning(last_h_state, enc_outputs) # (bs, 2*d_model)

        # 2. Chuẩn bị Decoder Input (Embedded input + Context)
        embedded_input = self.tgt_embedding(input_ids).squeeze(1) # (bs, d_model)
        # Ghép embedded_input và context_vector -> (bs, 3*d_model)
        decoder_input = torch.cat([embedded_input, context_vector], dim=-1).unsqueeze(1) # (bs, 1, 3*d_model)

        # 3. Chạy Decoder LSTM
        _, (new_h_state, new_c_state) = self.decoder(decoder_input, (h_state, c_state))

        # 4. Tính Context-aware Output (s_t_tilde)
        new_last_h_state = new_h_state[-1] # (bs, 2*d_model)
        
        # S_t_tilde = tanh(W_c [c_t ; h_t_last_layer])
        combined_output = torch.tanh(
            self.context_projection(
                torch.cat([context_vector, new_last_h_state], dim=-1) # (bs, 4*d_model)
            )
        ) # (bs, 2*d_model)

        return new_h_state.contiguous(), new_c_state.contiguous(), combined_output

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor: # <<< FIXED: Chỉ trả về Logits
        self.train()
        embedded_x = self.src_embedding(x)
        bs, _, dim = embedded_x.shape
        
        # 1. Encoder (Đã sửa: Gọi 1 lần để lấy tất cả output)
        enc_outputs, (enc_hn, enc_cn) = self.encoder(embedded_x)
        enc_hidden_states = enc_outputs # Annotation Vectors (h_j)

        # 2. Khởi tạo Trạng thái Decoder 
        h_0 = enc_hn[-2:].transpose(0, 1).reshape(bs, -1) # (bs, 2*d_model)
        c_0 = enc_cn[-2:].transpose(0, 1).reshape(bs, -1) # (bs, 2*d_model)

        dec_h_state = h_0.unsqueeze(0).repeat(self.n_decoder, 1, 1) # (n_decoder, bs, 2*d_model)
        dec_c_state = c_0.unsqueeze(0).repeat(self.n_decoder, 1, 1) # (n_decoder, bs, 2*d_model)
        dec_states = (dec_h_state, dec_c_state)
        
        _, tgt_len = y.shape # y là decoder_input (tgt[:, :-1])
        logits = []

        # 3. Decoding Loop (Teacher Forcing)
        # Lặp tgt_len lần để tạo ra tgt_len logits
        for ith in range(tgt_len): # <<< FIXED: range(tgt_len)
            y_ith = y[:, ith].unsqueeze(-1) # (bs, 1) - Từ đầu vào (teacher-forced)

            dec_h_state, dec_c_state, combined_output = self.forward_step(
                y_ith, 
                enc_hidden_states, 
                dec_states
            )
            dec_states = (dec_h_state, dec_c_state) # Cập nhật trạng thái
            
            logit = self.output_head(combined_output) # (bs, total_tgt_tokens)
            logits.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1) # (bs, tgt_len, total_tgt_tokens)

        # 4. Trả về Logits 
        return logits # <<< FIXED: Trả về logits để train.py tính loss

    def predict(self, x: torch.Tensor, max_len: int = 50):
        self.eval()
        with torch.no_grad():
            embedded_x = self.src_embedding(x)
            bs, _, _ = embedded_x.shape
            
            # 1. Encoder 
            enc_outputs, (enc_hn, enc_cn) = self.encoder(embedded_x)
            enc_hidden_states = enc_outputs

            # 2. Khởi tạo Trạng thái Decoder (Giống trong forward)
            h_0 = enc_hn[-2:].transpose(0, 1).reshape(bs, -1)
            c_0 = enc_cn[-2:].transpose(0, 1).reshape(bs, -1)
            dec_h_state = h_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
            dec_c_state = c_0.unsqueeze(0).repeat(self.n_decoder, 1, 1)
            dec_states = (dec_h_state, dec_c_state)

            # Bắt đầu dịch với token <bos>
            y_ith = torch.zeros(bs, ).fill_(self.vocab.bos_idx).long().to(x.device).unsqueeze(-1) # (bs, 1)
            
            mark_eos = torch.zeros(bs, dtype=torch.bool).to(x.device)
            outputs = []
            
            # 3. Decoding Loop
            for _ in range(max_len):
                # forward_step trả về (new_h, new_c, combined_output)
                dec_h_state, dec_c_state, combined_output = self.forward_step(
                    y_ith, 
                    enc_hidden_states, 
                    dec_states
                )
                dec_states = (dec_h_state, dec_c_state) # Cập nhật trạng thái

                # Dự đoán token tiếp theo
                logit = self.output_head(combined_output)
                y_ith = logit.argmax(dim=-1).long().unsqueeze(-1) # (bs, 1)

                # Đánh dấu đã kết thúc
                mark_eos = mark_eos | (y_ith.squeeze(-1) == self.vocab.eos_idx)
                
                # Nếu tất cả các chuỗi trong batch đều đã kết thúc
                if all(mark_eos.tolist()):
                    break

                # Chỉ thêm các token chưa phải là <eos> vào outputs
                outputs.append(y_ith.squeeze(-1))

            if outputs:
                outputs = torch.stack(outputs, dim=1) # (bs, length)
            else:
                # Trả về tensor rỗng nếu không có gì được dịch (ví dụ, độ dài max_len quá ngắn)
                outputs = torch.empty((bs, 0), dtype=torch.long, device=x.device)

            return outputs
