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

def train_fastapi_postgres_module(
    base_checkpoint: str = "checkpoints/dali1_coder_mern_fp32.pt",
    curated_dataset: str = "knowledge_base/04_fastapi_postgres/curated_dataset.txt",
    output_fp32: str = "checkpoints/dali1_coder_fastapi_fp32.pt",
    output_int4: str = "checkpoints/dali1_coder_int4.pt",
    num_steps: int = 150,
    seq_len: int = 256,
    lr: float = 4e-4
):
    print("=" * 72)
    print("  KNOWLEDGE BASE 04: TRAINING FASTAPI & ASYNC POSTGRESQL ENGINE")
    print("=" * 72)

    device = torch.device("cpu")
    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    tokenizer = BytePairTokenizer()
    tokenizer.load(ckpt["tokenizer_path"])

    model = DALI1Model(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    print("✓ Successfully loaded base 14.03M DALI-1 weights into memory.")

    with open(curated_dataset, "r", encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("<|end_block|>") if b.strip()]

    tokenized = [tokenizer.encode(b, add_special_tokens=True) for b in blocks]
    print(f"✓ Loaded {len(blocks)} comprehensive FastAPI & PostgreSQL instruction blocks.")

    pad_id = tokenizer.vocab[tokenizer.pad_token]
    batch_size = 4

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=5e-5)

    print(f"Starting FastAPI & PostgreSQL Continual Training ({num_steps} steps)...")
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

        if step % 30 == 0 or step == 1:
            ppl = math.exp(min(loss_lm.item(), 20.0))
            print(f"Step {step:03d}/{num_steps:03d} | Loss: {loss_lm.item():.4f} | Perplexity: {ppl:.2f}")

    total_time = time.time() - start_time
    print("-" * 72)
    print(f"✓ FastAPI & PostgreSQL Training Finished in {total_time:.2f} seconds!")

    # Save FP32 Model Checkpoint
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": ckpt["tokenizer_path"]
    }, output_fp32)
    print(f"✓ Saved Updated FP32 Checkpoint: {output_fp32}")

    # Quantize to 4-bit INT4
    print("\nQuantizing Updated Model to 4-Bit INT4...")
    quant_model, comp_ratio = quantize_model_to_4bit(model)
    torch.save({
        "config": config,
        "quantized_state_dict": quant_model.state_dict(),
        "tokenizer_path": ckpt["tokenizer_path"],
        "is_4bit": True
    }, output_int4)
    print(f"✓ Saved 4-Bit INT4 Model ({comp_ratio:.1f}x compression): {output_int4}")
    print("=" * 72)

if __name__ == "__main__":
    train_fastapi_postgres_module()
