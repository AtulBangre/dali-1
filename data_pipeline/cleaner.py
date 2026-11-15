import os
import re
from typing import List, Set

def clean_and_merge_datasets(
    corpus_files: List[str] = ["data/coding_corpus.txt", "data/dev_corpus.txt"],
    output_file: str = "data/final_coding_corpus.txt"
) -> int:
    print("=" * 68)
    print("  STEP 3: DATA CLEANING, DEDUPLICATION & VALIDATION")
    print("=" * 68)

    all_blocks = []
    seen_hashes: Set[str] = set()
    cleaned_count = 0

    for file_path in corpus_files:
        if not os.path.exists(file_path):
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "<|end_block|>" in content:
            blocks = [b.strip() for b in content.split("<|end_block|>") if b.strip()]
        else:
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]

        for b in blocks:
            # Deduplicate based on full block content or user query
            normalized_key = re.sub(r"\s+", " ", b.lower())
            if normalized_key in seen_hashes:
                continue
            seen_hashes.add(normalized_key)

            # Ensure valid QA block format
            if ("<|user|>" in b and "<|assistant|>" in b) or ("User:" in b and "Assistant:" in b):
                all_blocks.append(b)
                cleaned_count += 1

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n\n<|end_block|>\n\n".join(all_blocks))

    print(f"✓ Total Unique High-Quality QA Blocks: {cleaned_count}")
    print(f"✓ Written Cleaned Training Dataset to: {output_file}")
    print("=" * 68)
    return cleaned_count

if __name__ == "__main__":
    clean_and_merge_datasets()
