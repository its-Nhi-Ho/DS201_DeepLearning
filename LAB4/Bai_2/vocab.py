import torch
import re
import json
import os
import string 

class Vocab:
  def __init__(self, path: str, src_language: str, tgt_language: str):
    self.initialize_special_tokens()
    self.src_language = src_language
    self.tgt_language = tgt_language
    self.make_vocab(path, src_language, tgt_language)

  def initialize_special_tokens(self) -> None:
    self.bos_token = "<bos>"
    self.eos_token = "<eos>"
    self.pad_token = "<pad>"
    self.unk_token = "<unk>"

    self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

    self.pad_idx = 0
    self.bos_idx = 1
    self.eos_idx = 2
    self.unk_idx = 3
    
    self.special_ids = [self.pad_idx, self.bos_idx, self.eos_idx, self.unk_idx]

  def make_vocab(self, path: str, src_language: str, tgt_language: str):
    json_files = os.listdir(path)
    src_words = set()
    tgt_words = set()

    print("Building vocabulary...")
    for json_file in json_files:
      file_path = os.path.join(path, json_file)
      # Chỉ đọc file json
      if not json_file.endswith('.json'): 
          continue
          
      data = json.load(open(file_path, encoding='utf-8'))
      for item in data:
        src_sentence = item[src_language]
        tgt_sentence = item[tgt_language]

        src_tokens = self.preprocess_sentence(src_sentence)
        tgt_tokens = self.preprocess_sentence(tgt_sentence)

        src_words.update(src_tokens)
        tgt_words.update(tgt_tokens)

    # Tạo src dictionary
    src_i2s = self.specials + list(src_words) # Dùng list thay vì tuple để dễ xử lý
    self.src_i2s = {i: tok for i, tok in enumerate(src_i2s)}
    self.src_s2i = {tok: i for i, tok in enumerate(src_i2s)}

    # Tạo tgt dictionary
    tgt_i2s = self.specials + list(tgt_words)
    self.tgt_i2s = {i: tok for i, tok in enumerate(tgt_i2s)}
    self.tgt_s2i = {tok: i for i, tok in enumerate(tgt_i2s)}
    
    print(f"Vocab created. Src tokens: {len(self.src_i2s)}, Tgt tokens: {len(self.tgt_i2s)}")

  @property
  def total_src_tokens(self) -> int:
    return len(self.src_i2s)

  @property
  def total_tgt_tokens(self) -> int:
    return len(self.tgt_i2s)

  def preprocess_sentence(self, sentence: str) -> list:
    # Xử lý: Xóa dấu câu -> chữ thường -> tách từ (split)
    translator = str.maketrans("", "", string.punctuation)
    sentence = sentence.lower()
    sentence = sentence.translate(translator)
    # Phải split để trả về list các từ, không phải list các ký tự
    return sentence.split() 

  def encode_sentence(self, sentence: str, language: str) -> torch.Tensor:
    tokens = self.preprocess_sentence(sentence)
    
    if language == self.src_language:
        s2i = self.src_s2i
    elif language == self.tgt_language:
        s2i = self.tgt_s2i
    else:
        raise ValueError(f"Language {language} not supported")
        
    vec = [s2i.get(token, self.unk_idx) for token in tokens] # Dùng .get để an toàn hơn
    
    vec = [self.bos_idx] + vec + [self.eos_idx]
    
    return torch.tensor(vec, dtype=torch.long)

  def decode_sentence(self, tensor: torch.Tensor, language: str) -> list[str]:
    if isinstance(tensor, torch.Tensor):
        sentence_ids = tensor.tolist()
    else:
        sentence_ids = tensor

    if language == self.src_language:
        i2s = self.src_i2s
    else:
        i2s = self.tgt_i2s
        
    words = []
    for idx in sentence_ids:
        # Bỏ qua các token đặc biệt khi decode để ra câu tự nhiên
        if idx not in self.special_ids:
            words.append(i2s.get(idx, self.unk_token))
            
    return " ".join(words)
