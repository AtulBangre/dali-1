import os
import time
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.losses import DALILoss
from data.tinystories import BPEStoriesDataset
from generate import generate_story

def get_target_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("🚀 Hardware Acceleration: Apple Silicon GPU (MPS) Enabled!")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("🚀 Hardware Acceleration: NVIDIA CUDA Enabled!")
        return torch.device("cuda")
    else:
        print("⚙️ Hardware: Multi-Core CPU Mode Enabled.")
        return torch.device("cpu")

def main():
    print("=" * 68)
    print("  PHASE 6: DALI-1 SCALED MULTI-STORY PRE-TRAINING SPRINT")
    print("=" * 68)

    device = get_target_device()

    dataset = BPEStoriesDataset(
        corpus_path="data/stories_corpus.txt",
        tokenizer_path="checkpoints/bpe_tokenizer.json",
        seq_len=64
    )
    tokenizer = dataset.tokenizer
    
    print(f"BPE Subword Vocabulary Size: {dataset.vocab_size} tokens")
    print(f"Total Stories In Corpus:      {len(dataset.raw_stories)} rich narrative documents")
    
    # Model configuration
    config = DALIConfig(
        vocab_size=dataset.vocab_size,
        hidden_dim=128,
        state_dim=128,
        memory_slots=32,
        memory_dim=64,
        num_adaptive_layers=2,
        dropout=0.05
    )

    model = DALI1Model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DALI-1 Trainable Parameters: {total_params:,}")
    print("-" * 68)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    num_steps = 500
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    loss_fn = DALILoss(config).to(device)

    batch_size = 16
    pad_id = tokenizer.vocab[tokenizer.pad_token]

    print(f"Starting Pre-Training Sprint ({num_steps} steps, batch_size={batch_size})...")
    start_time = time.time()
    total_tokens_processed = 0

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        input_ids, targets = dataset.get_batch(batch_size, device)
        outputs = model(input_ids, targets=targets)
        
        logits = outputs["logits"]
        loss_lm = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            targets.view(-1),
            ignore_index=pad_id
        )

        loss = loss_lm + config.budget_penalty_weight * outputs["avg_steps"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_tokens_processed += (input_ids != pad_id).sum().item()

        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            tps = total_tokens_processed / max(elapsed, 1e-5)
            ppl = math.exp(min(loss_lm.item(), 20.0))
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:03d}/{num_steps:03d} | "
                f"Loss: {loss_lm.item():.4f} | "
                f"PPL: {ppl:<6.2f} | "
                f"Speed: {tps:,.0f} tok/s | "
                f"LR: {current_lr:.5f}"
            )

    train_time = time.time() - start_time
    print("-" * 68)
    print(f"Pre-Training successfully finished in {train_time:.2f} seconds!")
    print(f"Average Processing Throughput: {total_tokens_processed / train_time:,.0f} tokens/second")
    print("=" * 68)

    # Save Scaled Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/dali1_scaled_stories.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": "checkpoints/bpe_tokenizer.json",
        "steps": num_steps,
        "final_loss": loss_lm.item()
    }, ckpt_path)
    print(f"Saved scaled checkpoint to: {ckpt_path}")

    # Generation Demonstration Across Diverse Topics
    print("\n" + "=" * 68)
    print("   DALI-1 AUTOREGRESSIVE GENERATION ACROSS DIVERSE TOPICS")
    print("=" * 68)
    
    prompts = [
        "Once upon a time, a clever little fox named Rusty",
        "Once upon a time, an old lighthouse keeper named Arthur",
        "Once upon a time, a robotic toy named Gizmo",
        "Once upon a time, a young astronomer named Kevin"
    ]

    for p in prompts:
        story = generate_story(model, tokenizer, prompt=p, max_new_tokens=45, temperature=0.6)
        print(f"\n[Prompt]: \"{p}\"")
        print(f"[Generated Story]:\n{story}\n")
    print("=" * 68)

if __name__ == "__main__":
    main()
