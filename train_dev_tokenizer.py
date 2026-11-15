import os
from dali.tokenizer import BytePairTokenizer

def main():
    print("Training Dev & Python BPE Tokenizer...")
    with open("data/dev_corpus.txt", "r") as f:
        text = f.read()

    tokenizer = BytePairTokenizer(vocab_size=1024)
    tokenizer.train([text])

    os.makedirs("checkpoints", exist_ok=True)
    out_path = "checkpoints/dev_bpe.json"
    tokenizer.save(out_path)
    print(f"✓ Saved Dev Tokenizer with {tokenizer.vocab_size} tokens to {out_path}")

if __name__ == "__main__":
    main()
