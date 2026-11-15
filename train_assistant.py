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
from dali.tokenizer import BytePairTokenizer
from dali.quantization import quantize_model_to_4bit

class AssistantDataset:
    def __init__(self, corpus_path: str = "data/assistant_corpus.txt", tokenizer_path: str = "checkpoints/assistant_bpe.json", seq_len: int = 64):
        self.seq_len = seq_len
        self.tokenizer = BytePairTokenizer()
        self.tokenizer.load(tokenizer_path)

        with open(corpus_path, "r") as f:
            content = f.read()
            
        # Split into User-Assistant dialogue blocks
        pairs = content.split("\n\n")
        self.samples = [p.strip() for p in pairs if p.strip()]
        self.tokenized = [self.tokenizer.encode(p) for p in self.samples]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_chunked_batch(self, batch_size: int, device: torch.device):
        batch_inputs = []
        batch_targets = []
        pad_id = self.tokenizer.vocab[self.tokenizer.pad_token]

        for _ in range(batch_size):
            sample = self.tokenized[torch.randint(0, len(self.tokenized), (1,)).item()]
            if len(sample) > self.seq_len:
                chunk = sample[:self.seq_len]
            else:
                chunk = sample + [pad_id] * (self.seq_len - len(sample))

            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])

        return (
            torch.tensor(batch_inputs, dtype=torch.long, device=device),
            torch.tensor(batch_targets, dtype=torch.long, device=device)
        )

def main():
    print("=" * 68)
    print("  PHASE 8: CHUNKED STATE TRAINING & 4-BIT NANO-ASSISTANT (CODE & Q&A)")
    print("=" * 68)

    device = torch.device("cpu")
    dataset = AssistantDataset()
    tokenizer = dataset.tokenizer

    print(f"Assistant Vocabulary Size: {dataset.vocab_size} tokens")
    print(f"Total Instruction Pairs:   {len(dataset.samples)} Python & World Fact pairs")
    
    # 4M Parameter Config
    config = DALIConfig(
        vocab_size=dataset.vocab_size,
        hidden_dim=256,
        state_dim=256,
        memory_slots=64,
        memory_dim=128,
        chunk_size=8,
        num_adaptive_layers=2,
        dropout=0.05
    )

    model = DALI1Model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DALI-1 Trainable Parameters: {total_params:,} (~1.5M params)")
    print("-" * 68)

    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    num_steps = 400
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-4)
    loss_fn = DALILoss(config).to(device)

    batch_size = 16
    pad_id = tokenizer.vocab[tokenizer.pad_token]

    print(f"Starting Chunked State Instruction Training ({num_steps} steps)...")
    start_time = time.time()

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        input_ids, targets = dataset.get_chunked_batch(batch_size, device)
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

        if step % 100 == 0 or step == 1:
            ppl = math.exp(min(loss_lm.item(), 20.0))
            print(f"Step {step:03d}/{num_steps:03d} | LM Loss: {loss_lm.item():.4f} | Perplexity: {ppl:.2f}")

    train_time = time.time() - start_time
    print("-" * 68)
    print(f"Instruction Training Finished in {train_time:.2f} seconds!")

    # 4. Save FP32 Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    fp32_ckpt = "checkpoints/dali1_assistant_fp32.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": "checkpoints/assistant_bpe.json"
    }, fp32_ckpt)
    print(f"Saved full precision checkpoint to: {fp32_ckpt}")

    # 5. Convert to 4-bit INT4 Quantization!
    print("\nApplying 4-bit INT4 Quantization (87.5% memory compression)...")
    quant_model, compression = quantize_model_to_4bit(model)
    print(f"Model successfully compressed by: {compression:.1f}x into 4-bit INT4 format!")

    int4_ckpt = "checkpoints/dali1_assistant_int4.pt"
    torch.save({
        "config": config,
        "quantized_state_dict": quant_model.state_dict(),
        "tokenizer_path": "checkpoints/assistant_bpe.json",
        "is_4bit": True
    }, int4_ckpt)
    print(f"Saved 4-bit INT4 Assistant Checkpoint to: {int4_ckpt}")
    print("=" * 68)

if __name__ == "__main__":
    main()
