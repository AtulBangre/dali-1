import os
import time
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.tokenizer import BytePairTokenizer
from dali.quantization import quantize_model_to_4bit

def compile_master_web_corpus(output_path: str = "knowledge_base/master_web_corpus.txt") -> int:
    sources = [
        "knowledge_base/00_html5_standards/curated_dataset.txt",
        "knowledge_base/01_tailwind_ui/curated_dataset.txt",
        "knowledge_base/01_tailwind_ui/full_utilities_reference.txt",
        "data/final_coding_corpus.txt"
    ]
    all_blocks = []
    seen = set()

    for s in sources:
        if not os.path.exists(s):
            continue
        with open(s, "r", encoding="utf-8") as f:
            content = f.read()
        blocks = [b.strip() for b in content.split("<|end_block|>") if b.strip()]
        for b in blocks:
            norm = b[:150].lower()
            if norm not in seen:
                seen.add(norm)
                all_blocks.append(b)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n<|end_block|>\n\n".join(all_blocks))

    print(f"✓ Compiled Master Web Corpus with {len(all_blocks)} high-signal official blocks into {output_path}")
    return len(all_blocks)

def train_master_web_model(
    base_checkpoint: str = "checkpoints/dali1_coder_fp32.pt",
    master_corpus: str = "knowledge_base/master_web_corpus.txt",
    tokenizer_path: str = "checkpoints/code_bpe.json",
    output_fp32: str = "checkpoints/dali1_web_master_fp32.pt",
    output_int4: str = "checkpoints/dali1_coder_int4.pt",
    num_steps: int = 350,
    seq_len: int = 256,
    lr: float = 8e-4
):
    print("=" * 72)
    print("  TRAINING FULL-FLEDGED MASTER WEB DEVELOPMENT & UI MODEL (DALI-1)")
    print("=" * 72)

    device = torch.device("cpu")
    compile_master_web_corpus(master_corpus)

    tokenizer = BytePairTokenizer()
    tokenizer.load(tokenizer_path)

    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = DALI1Model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    print("✓ Successfully loaded base 14.03M DALI-1 weights into memory.")

    with open(master_corpus, "r", encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("<|end_block|>") if b.strip()]

    tokenized = [tokenizer.encode(b, add_special_tokens=True) for b in blocks]
    pad_id = tokenizer.vocab[tokenizer.pad_token]
    batch_size = 6

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=5e-5)

    print(f"Starting Master Curriculum Training ({num_steps} steps, Batch Size {batch_size})...")
    start_time = time.time()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        batch_inputs = []
        batch_targets = []

        for _ in range(batch_size):
            sample = tokenized[torch.randint(0, len(tokenized), (1,)).item()]
            if len(sample) > seq_len:
                chunk = sample[:seq_len]
            else:
                chunk = sample + [pad_id] * (seq_len - len(sample))
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])

        inp = torch.tensor(batch_inputs, dtype=torch.long, device=device)
        tgt = torch.tensor(batch_targets, dtype=torch.long, device=device)

        outputs = model(inp, targets=tgt)
        loss_lm = F.cross_entropy(
            outputs["logits"].view(-1, config.vocab_size),
            tgt.view(-1),
            ignore_index=pad_id
        )

        loss = loss_lm + config.budget_penalty_weight * outputs["avg_steps"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 70 == 0 or step == 1:
            ppl = math.exp(min(loss_lm.item(), 20.0))
            print(f"Step {step:03d}/{num_steps:03d} | Loss: {loss_lm.item():.4f} | Perplexity: {ppl:.2f}")

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"✓ Master Training Finished in {total_time:.2f} seconds!")

    # Save FP32 Model Checkpoint
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": tokenizer_path
    }, output_fp32)
    print(f"✓ Saved Full Master FP32 Checkpoint: {output_fp32}")

    # Quantize to 4-bit INT4
    print("\nQuantizing Full Master Model to 4-Bit INT4...")
    quant_model, comp_ratio = quantize_model_to_4bit(model)
    torch.save({
        "config": config,
        "quantized_state_dict": quant_model.state_dict(),
        "tokenizer_path": tokenizer_path,
        "is_4bit": True
    }, output_int4)
    print(f"✓ Saved 4-Bit INT4 Model ({comp_ratio:.1f}x compression): {output_int4}")
    print("=" * 72)

if __name__ == "__main__":
    train_master_web_model()
