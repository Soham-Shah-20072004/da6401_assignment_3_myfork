"""
dataset.py — Multi30k loading, spaCy tokenization, vocabulary, batching
DA6401 Assignment 3: "Attention Is All You Need"

Source = German (de), Target = English (en).

Special tokens (indices fixed — pad_idx=1 matches model.make_*_mask defaults):
    <unk> = 0   <pad> = 1   <sos> = 2   <eos> = 3
"""

from collections import Counter
from typing import List

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import spacy
from datasets import load_dataset


UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = [UNK, PAD, SOS, EOS]


# ──────────────────────────────────────────────────────────────────────
#  Vocabulary
# ──────────────────────────────────────────────────────────────────────

class Vocab:
    """word ↔ index map. Built from the TRAIN split only (data isolation)."""

    def __init__(self, counter: Counter, min_freq: int = 2) -> None:
        self.itos: List[str] = list(SPECIALS)
        for tok, freq in counter.most_common():
            if freq >= min_freq:
                self.itos.append(tok)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    # autograder / evaluate_bleu compatibility
    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class Multi30kDataset(Dataset):
    """
    Loads one split of bentrevett/multi30k and produces (src_ids, tgt_ids)
    LongTensors, each wrapped with <sos> ... <eos>.

    Vocabs are built once on the train split and shared with val/test by
    passing them in (so val/test never leak vocabulary into training).
    """

    _nlp_de = None
    _nlp_en = None

    def __init__(self, split: str = "train", src_vocab=None, tgt_vocab=None,
                 min_freq: int = 2) -> None:
        self.split = split
        self.data = load_dataset("bentrevett/multi30k", split=split)

        if Multi30kDataset._nlp_de is None:
            Multi30kDataset._nlp_de = spacy.load("de_core_news_sm")
            Multi30kDataset._nlp_en = spacy.load("en_core_web_sm")

        # Tokenize (lowercase) once up front.
        self.src_tok = [self._tok_de(ex["de"]) for ex in self.data]
        self.tgt_tok = [self._tok_en(ex["en"]) for ex in self.data]

        if src_vocab is None or tgt_vocab is None:
            if split != "train":
                raise ValueError("val/test must receive vocabs built on train")
            self.src_vocab = Vocab(self._count(self.src_tok), min_freq)
            self.tgt_vocab = Vocab(self._count(self.tgt_tok), min_freq)
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

    # -- tokenizers --------------------------------------------------
    def _tok_de(self, s: str) -> List[str]:
        return [t.text for t in Multi30kDataset._nlp_de.tokenizer(s.lower())]

    def _tok_en(self, s: str) -> List[str]:
        return [t.text for t in Multi30kDataset._nlp_en.tokenizer(s.lower())]

    @staticmethod
    def _count(tokenized: List[List[str]]) -> Counter:
        c = Counter()
        for toks in tokenized:
            c.update(toks)
        return c

    # -- Dataset protocol -------------------------------------------
    def __len__(self) -> int:
        return len(self.src_tok)

    def __getitem__(self, i: int):
        src = [SOS_IDX] + self.src_vocab.encode(self.src_tok[i]) + [EOS_IDX]
        tgt = [SOS_IDX] + self.tgt_vocab.encode(self.tgt_tok[i]) + [EOS_IDX]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────
#  Collate — pad a batch to its longest sequence
# ──────────────────────────────────────────────────────────────────────

def collate_batch(batch):
    """batch: list of (src_ids, tgt_ids). Returns padded [B, Ls], [B, Lt]."""
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    return src, tgt
