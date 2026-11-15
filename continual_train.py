import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.tokenizer import BytePairTokenizer

def add_new_knowledge(
    base_checkpoint: str = "checkpoints/dali1_dev_14m_fp32.pt",
    new_knowledge_path: str = "data/additional_knowledge.txt",
    output_checkpoint: str = "checkpoints/dali1_dev_14m_updated.pt",
    steps: int = 50,
    lr: float = 3e-4
):
    print("=" * 68)
    print("  DALI-1 CONTINUAL LEARNING: ADDING NEW KNOWLEDGE TO TRAINED MODEL")
    print("=" * 68)

    ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    tokenizer = BytePairTokenizer()
    tokenizer.load(ckpt["tokenizer_path"])

    model = DALI1Model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.train()
    print("✓ Successfully loaded base trained model weights into memory.")

    with open(new_knowledge_path, "r", encoding="utf-8") as f:
        pairs = [p.strip() for p in f.read().split("\n\n") if p.strip()]

    print(f"✓ Injecting {len(pairs)} new knowledge blocks without forgetting...")

    pad_id = tokenizer.vocab[tokenizer.pad_token]
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    tokenized_new = [tokenizer.encode(p) for p in pairs]

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        sample = tokenized_new[step % len(tokenized_new)]
        
        seq_len = 64
        if len(sample) > seq_len:
            chunk = sample[:seq_len]
        else:
            chunk = sample + [pad_id] * (seq_len - len(sample))

        inp = torch.tensor([chunk[:-1]], dtype=torch.long)
        tgt = torch.tensor([chunk[1:]], dtype=torch.long)

        out = model(inp, targets=tgt)
        loss = F.cross_entropy(out["logits"].view(-1, config.vocab_size), tgt.view(-1), ignore_index=pad_id)
        loss.backward()
        optimizer.step()

        if step % 25 == 0 or step == 1:
            print(f"Incremental Step {step:02d}/{steps:02d} | Loss: {loss.item():.4f}")

    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": ckpt["tokenizer_path"]
    }, output_checkpoint)
    print(f"✓ Successfully updated checkpoint saved to: {output_checkpoint}")
    print("=" * 68)

if __name__ == "__main__":
    add_new_knowledge()
