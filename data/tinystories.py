import os
import random
import torch
from typing import List, Tuple, Dict
from dali.tokenizer import BytePairTokenizer

class BPEStoriesDataset:
    """Manages tokenized sequences and batch generation for BPE-encoded stories."""
    def __init__(self, corpus_path: str = "data/stories_corpus.txt", tokenizer_path: str = "checkpoints/bpe_tokenizer.json", seq_len: int = 64):
        self.seq_len = seq_len
        self.tokenizer = BytePairTokenizer()
        
        if os.path.exists(tokenizer_path):
            self.tokenizer.load(tokenizer_path)
        else:
            with open(corpus_path, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
            self.tokenizer.train(texts)
            self.tokenizer.save(tokenizer_path)

        with open(corpus_path, "r") as f:
            self.raw_stories = [line.strip() for line in f if line.strip()]

        self.tokenized_stories = [self.tokenizer.encode(s) for s in self.raw_stories]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_inputs = []
        batch_targets = []
        pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        for _ in range(batch_size):
            story = random.choice(self.tokenized_stories)
            if len(story) > self.seq_len:
                start = random.randint(0, len(story) - self.seq_len)
                chunk = story[start : start + self.seq_len]
            else:
                chunk = story + [pad_id] * (self.seq_len - len(story))

            inp = chunk[:-1]
            tgt = chunk[1:]
            batch_inputs.append(inp)
            batch_targets.append(tgt)

        return (
            torch.tensor(batch_inputs, dtype=torch.long, device=device),
            torch.tensor(batch_targets, dtype=torch.long, device=device)
        )
