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

class CoderDataset:
    def __init__(self, corpus_path: str = "data/final_coding_corpus.txt", tokenizer_path: str = "checkpoints/code_bpe.json", seq_len: int = 256):
        self.seq_len = seq_len
        self.tokenizer = BytePairTokenizer()
        self.tokenizer.load(tokenizer_path)

        with open(corpus_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "<|end_block|>" in content:
            pairs = content.split("<|end_block|>")
        else:
            pairs = content.split("\n\n")

        self.samples = [p.strip() for p in pairs if p.strip()]
        self.tokenized = [self.tokenizer.encode(p) for p in self.samples]

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    def get_batch(self, batch_size: int, device: torch.device):
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
    print("=" * 70)
    print("  STEP 6 & 7: TRAINING DALI-1 WEB & CODE DEBUGGING ASSISTANT")
    print("=" * 70)

    device = torch.device("cpu")
    dataset = CoderDataset()
    tokenizer = dataset.tokenizer

    print(f"✓ Vocabulary Size:           {dataset.vocab_size} specialized code tokens")
    print(f"✓ Total Training Curated QA: {len(dataset.samples)} full-stack & debugging blocks")

    config = DALIConfig(
        vocab_size=dataset.vocab_size,
        hidden_dim=640,
        state_dim=640,
        memory_slots=128,
        memory_dim=256,
        chunk_size=8,
        num_adaptive_layers=6,
        tie_word_embeddings=False,
        dropout=0.05
    )

    model = DALI1Model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ DALI-1 Model Size:         {total_params:,} parameters ({total_params/1e6:.2f}M)")
    print("-" * 70)

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=0.01)
    num_steps = 450
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=5e-5)

    batch_size = 8
    pad_id = tokenizer.vocab[tokenizer.pad_token]

    print(f"Starting Training ({num_steps} steps, Batch Size {batch_size})...")
    start_time = time.time()

    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        inputs, targets = dataset.get_batch(batch_size, device)
        outputs = model(inputs, targets=targets)

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

        if step % 70 == 0 or step == 1:
            ppl = math.exp(min(loss_lm.item(), 20.0))
            print(f"Step {step:03d}/{num_steps:03d} | Loss: {loss_lm.item():.4f} | Perplexity: {ppl:.2f} | Adaptive Depth: {outputs['avg_steps'].item():.2f}")

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"✓ Training Completed in {total_time:.2f} seconds!")

    # Save FP32 Model Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    fp32_path = "checkpoints/dali1_coder_fp32.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": "checkpoints/code_bpe.json"
    }, fp32_path)
    print(f"✓ Saved FP32 Checkpoint: {fp32_path}")

    # Step 7: 4-Bit INT4 Quantization
    print("\nQuantizing Model to 4-bit INT4...")
    quant_model, comp_ratio = quantize_model_to_4bit(model)
    int4_path = "checkpoints/dali1_coder_int4.pt"
    torch.save({
        "config": config,
        "quantized_state_dict": quant_model.state_dict(),
        "tokenizer_path": "checkpoints/code_bpe.json",
        "is_4bit": True
    }, int4_path)
    print(f"✓ Step 7 Complete: Compressed by {comp_ratio:.1f}x! Saved 4-bit INT4 Checkpoint: {int4_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
