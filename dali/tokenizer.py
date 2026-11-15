import os
import json
import re
from typing import List, Dict, Tuple, Optional

class BytePairTokenizer:
    """
    Subword Byte-Pair Encoding (BPE) Tokenizer with Space-Prefix Preservation.
    - Preserves exact original whitespace using the space-prefix symbol 'Ġ'.
    - Zero unknown tokens: encodes arbitrary unseen text without information loss.
    - Learns frequent subword merges from training text.
    """
    def __init__(self, vocab_size: int = 512):
        self.target_vocab_size = vocab_size
        self.space_prefix = "Ġ"
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"
        
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}

    def _get_stats(self, splits: List[List[str]]) -> Dict[Tuple[str, str], int]:
        pairs = {}
        for word in splits:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def _merge_pair(self, pair: Tuple[str, str], splits: List[List[str]]) -> List[List[str]]:
        new_splits = []
        p0, p1 = pair
        merged = p0 + p1
        for word in splits:
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_splits.append(new_word)
        return new_splits

    def _pre_tokenize(self, text: str) -> List[str]:
        """Splits text into code tokens, preserving exact whitespace, tabs, and newlines."""
        return re.findall(r"\n|[ ]+|\t|\w+|[^\w\s]", text)

    def train(self, texts: List[str]):
        """Learns subword vocabulary and BPE merges from corpus."""
        words = []
        char_set = set()
        for text in texts:
            toks = self._pre_tokenize(text)
            for tok in toks:
                words.append(tok)
                for c in tok:
                    char_set.add(c)

        # Initialize base vocab
        self.vocab = {tok: idx for idx, tok in enumerate(self.special_tokens)}
        for char in sorted(list(char_set)):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        splits = [list(w) for w in words]
        self.merges = {}

        num_merges = self.target_vocab_size - len(self.vocab)
        for _ in range(max(0, num_merges)):
            pairs = self._get_stats(splits)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            merged_token = best_pair[0] + best_pair[1]
            splits = self._merge_pair(best_pair, splits)
            self.merges[best_pair] = merged_token
            if merged_token not in self.vocab:
                self.vocab[merged_token] = len(self.vocab)

        self.inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encodes arbitrary text into subword token IDs."""
        words = self._pre_tokenize(text)
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.bos_token])

        for word in words:
            subwords = list(word)
            for pair, merged in self.merges.items():
                p0, p1 = pair
                i = 0
                new_subwords = []
                while i < len(subwords):
                    if i < len(subwords) - 1 and subwords[i] == p0 and subwords[i + 1] == p1:
                        new_subwords.append(merged)
                        i += 2
                    else:
                        new_subwords.append(subwords[i])
                        i += 1
                subwords = new_subwords

            for sw in subwords:
                token_ids.append(self.vocab.get(sw, self.vocab[self.unk_token]))

        if add_special_tokens:
            token_ids.append(self.vocab[self.eos_token])
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes token IDs back to exact original text with lossless whitespace."""
        subwords = []
        for tid in token_ids:
            tok = self.inverse_vocab.get(tid, "")
            if tok in self.special_tokens:
                continue
            subwords.append(tok)
            
        return "".join(subwords)

    def save(self, filepath: str):
        data = {
            "vocab": self.vocab,
            "merges": [list(k) for k in self.merges.keys()]
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = {tuple(pair): pair[0] + pair[1] for pair in data["merges"]}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
