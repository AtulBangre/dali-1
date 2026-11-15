import os
from dali.tokenizer import BytePairTokenizer

def main():
    print("=" * 65)
    print("      PHASE 5: DALI-1 BYTE-PAIR ENCODING (BPE) TOKENIZER")
    print("=" * 65)

    corpus_path = "data/stories_corpus.txt"
    with open(corpus_path, "r") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts)} story documents from {corpus_path}.")

    # Train BPE tokenizer with target vocabulary size 1024
    target_vocab = 1024
    tokenizer = BytePairTokenizer(vocab_size=target_vocab)
    print(f"Training BPE Tokenizer (Target Vocabulary Size: {target_vocab})...")
    tokenizer.train(texts)

    print(f"BPE Training Complete! Total Learned Vocabulary: {tokenizer.vocab_size} tokens.")
    print(f"Total Learned Merges: {len(tokenizer.merges)} subword rules.")

    # Save tokenizer
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/bpe_tokenizer.json"
    tokenizer.save(save_path)
    print(f"Saved tokenizer to: {save_path}")

    # Verification & Roundtrip Test
    print("\n--- Subword Encoding & Roundtrip Tests ---")
    test_sentences = [
        "Once upon a time, Lily found a red ball.",
        "Maya imagined visiting distant planets in her spaceship.",
        "The baby dragon Spark blew warm sparkling bubbles!"
    ]

    for s in test_sentences:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        print(f"\n[Original]:  {s}")
        print(f"[Tokens]:    {encoded[:12]}... (Total {len(encoded)} subwords)")
        print(f"[Decoded]:   {decoded}")

    # Handling Unseen / Out-of-Vocabulary Words
    unseen_sentence = "An adventurous astronaut discovered quantum crystals on Mars."
    unseen_encoded = tokenizer.encode(unseen_sentence)
    unseen_decoded = tokenizer.decode(unseen_encoded)
    print(f"\n[Unseen Sentence Test (Zero <unk>)]: {unseen_sentence}")
    print(f"[Subword Tokens]: {unseen_encoded}")
    print(f"[Decoded Output]: {unseen_decoded}")
    print("=" * 65)

if __name__ == "__main__":
    main()
