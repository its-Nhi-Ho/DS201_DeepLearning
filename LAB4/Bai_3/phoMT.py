import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from vocab import Vocab

def collate_fn(batch: list[dict]) -> dict:
    # Lấy danh sách các tensor nguồn và đích từ batch
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return {
        "src": src_padded,
        "tgt": tgt_padded
    }

class phoMTDataset(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        # Load dữ liệu
        self.data = json.load(open(path, encoding="utf-8"))
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        item = self.data[index]

        # 1. Lấy text từ file JSON (Key trong file JSON là 'english' và 'vietnamese')
        src_text = item["english"]
        tgt_text = item["vietnamese"]

        # 2. Mã hóa
        encoded_src = self.vocab.encode_sentence(src_text, self.vocab.src_language)
        encoded_tgt = self.vocab.encode_sentence(tgt_text, self.vocab.tgt_language)

        # 3. Trả về dictionary với key chuẩn 'src' và 'tgt'
        return {
            "src": encoded_src,
            "tgt": encoded_tgt
        }
