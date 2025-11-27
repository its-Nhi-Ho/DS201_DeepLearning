import torch
from typing import List, Tuple
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import numpy as np
import json
import os

def collate_fn(items):
    input_ids = [item["input_ids"] for item in items]
    label_ids = [item["label"] for item in items]
    lengths = torch.tensor([len(x) for x in input_ids], dtype=torch.long)
    max_len = max(t.size(0) for t in input_ids)

    input_ids = [
        F.pad(t, pad=(0, max_len - t.size(0)), mode="constant", value=0).unsqueeze(0)
        for t in input_ids
    ]

    label_ids = [
        F.pad(l, pad=(0, max_len - l.size(0)), mode="constant", value=-100).unsqueeze(0)
        for l in label_ids
    ]

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "label": torch.cat(label_ids, dim=0),
        "lengths": lengths
    }

class Vocab:
  def __init__(self, data_files):
    all_words = set()
    tags = set()

    for path in data_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                sent = json.loads(line.strip())
                tokens = sent["words"]
                labels = sent["tags"]

                all_words.update(tokens)
                tags.update(labels)


    self.pad = "<p>"
    all_words = sorted(all_words)
    self.w2i = {word: idx for idx,word in enumerate(all_words, start = 1)}
    self.w2i[self.pad] = 0

    self.i2w = {
        idx: word for word, idx in self.w2i.items()
    }
    tags = sorted(tags)
    self.l2i = {tag: idx for idx, tag in enumerate(tags)}

    self.i2l = {
        idx: tag for tag, idx in self.l2i.items()
    }

  @property
  def len(self):
    return len(self.w2i)

  @property
  def n_labels(self):
    return len(self.l2i)

  def encode_sentence(self,tokens):
    ids = [self.w2i.get(tok,0) for tok in tokens]
    return torch.tensor(ids).long()

  def encode_labels(self, labels):
    ids = [self.l2i[l] for l in labels]
    return torch.tensor(ids).long()
  def decode_labels(self, id_tensor):
    return [self.i2l[i] for i in id_tensor.tolist()]


class phoNERDataset(Dataset):
  def __init__(self, json_path, vocab:Vocab):
    super().__init__()
    self.vocab = vocab
    self.samples = []

    with open(json_path, encoding="utf-8") as f:
      for line in f:
          sent = json.loads(line.strip())
          tokens = sent["words"]
          labels = sent["tags"]

          x = vocab.encode_sentence(tokens)
          y = vocab.encode_labels(labels)

          self.samples.append({
              "input_ids": x,
              "label": y
          })

  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    return self.samples[idx]
