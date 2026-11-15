import os
from dali.tokenizer import BytePairTokenizer

def train_code_tokenizer(
    corpus_path: str = "data/final_coding_corpus.txt",
    output_path: str = "checkpoints/code_bpe.json",
    vocab_size: int = 1024
):
    print("=" * 68)
    print("  STEP 4: TRAINING CODE-PRESERVING BPE TOKENIZER")
    print("=" * 68)

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BytePairTokenizer(vocab_size=vocab_size)
    tokenizer.train([text])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"✓ Trained BPE Vocabulary Size: {tokenizer.vocab_size} tokens")
    print(f"✓ Checkpoint Saved to:        {output_path}")
    print("=" * 68)
    return tokenizer

if __name__ == "__main__":
    train_code_tokenizer()
