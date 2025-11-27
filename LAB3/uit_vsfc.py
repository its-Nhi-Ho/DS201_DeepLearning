import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import json
import string
import os

def collate_fn(items: dict) -> torch.Tensor:
  input_ids = [item["input_ids"] for item in items]
  max_len = max((input.shape[0] for input in input_ids))
  input_ids = [
     F.pad(
         input,
         pad = (0,max_len - input.shape[0]),
         mode = "constant",
         value = 0
     ).unsqueeze(0) for input in input_ids]

  label_ids = [item["label"].unsqueeze(0) for item in items]

  return {
      "input_ids": torch.cat(input_ids, dim = 0),
      "label": torch.cat(label_ids, dim = 0),
  }

class Vocab:
  def __init__(self, path: list[str]):
    all_words = set()
    labels = set()
    for filename in os.listdir(path):
      data = json.load(open(os.path.join(path,filename)))
      for item in data:
        sentence: str = item["sentence"]
        sentence = self.preprocess_sentence(sentence)

        words = sentence.split()
        all_words.update(words)
        labels.add(item["topic"])

    self.pad= "<p>"

    self.w2i = {
        word: idx for idx, word in enumerate(all_words, start = 1)
    }
    self.w2i[self.pad] = 0
    self.i2w = {
        idx: word for word, idx in self.w2i.items()
    }

    self.l2i = {
        label: idx for idx, label in enumerate(labels)
    }

    self.i2l = {
        idx: label for label, idx in self.l2i.items()
    }

  @property
  def n_labels(self) -> int:
    return len(self.l2i)

  @property
  def len(self) -> int:
    return len(self.w2i)

  def preprocess_sentence(self, sentence: str) -> str:
    translator = str.maketrans("","", string.punctuation)
    sentence = sentence.lower()
    sentence = sentence.translate(translator)

    return sentence

  def encode_sentence(self, sentence: str) -> torch.Tensor:
    sentence = self.preprocess_sentence(sentence)
    words = sentence.split()
    word_ids = [self.w2i[word] for word in words]

    return torch.Tensor(word_ids).long()

  def encode_label(self, label:str) -> torch.Tensor:
    label_idx = self.l2i[label]
    return torch.tensor(label_idx)

  def decode_label(self, label_ids: torch.Tensor) -> str:
    label_ids = label_ids.tolist()
    labels = [self.i2l[idx] for idx in label_ids]

    return labels

class UIT_VSFC(Dataset):
  def __init__(self, data_dir:str, vocab):
    super().__init__()
    self.vocab = vocab
    self.samples = []

    data = json.load(open(data_dir, "r", encoding="utf-8"))

    for item in data:
      sentence = item["sentence"]
      topic = item["topic"]

      input_ids = vocab.encode_sentence(sentence)
      label = vocab.encode_label(topic)

      self.samples.append({
            "input_ids": input_ids,
            "label": label
        })
  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx:int):
    return self.samples[idx]
