import os
import time
import torch
import torch.optim as optim

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.losses import DALILoss
from data.synthetic_memory import SyntheticFactMemoryDataset

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")

def main():
    device = get_device()
    
    # 1. Configuration (Small ~2M to 10M prototype for quick local experimentation)
    config = DALIConfig(
        vocab_size=1000,
        hidden_dim=128,
        state_dim=128,
        memory_slots=32,
        memory_dim=64,
        memory_top_k=2,
        num_adaptive_layers=2,
        dropout=0.1
    )

    print("Initializing DALI-1 Model...")
    model = DALI1Model(config).to(device)
    loss_fn = DALILoss(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    # 2. Data Generator
    dataset = SyntheticFactMemoryDataset(vocab_size=config.vocab_size, seq_len=64)

    # 3. Training Loop
    num_steps = 100
    batch_size = 16
    print(f"\nStarting Training Sprint ({num_steps} steps, batch_size={batch_size})...")

    start_time = time.time()
    for step in range(1, num_steps + 1):
        model.train()
        optimizer.zero_grad()

        input_ids, targets = dataset.get_batch(batch_size, device)

        # Forward pass
        outputs = model(input_ids, targets=targets)
        loss_dict = loss_fn(
            logits=outputs["logits"],
            targets=targets,
            avg_steps=outputs["avg_steps"],
            surprises=outputs["surprises"]
        )

        loss = loss_dict["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 20 == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                f"Step {step:03d}/{num_steps:03d} | "
                f"Total Loss: {loss.item():.4f} | "
                f"LM Loss: {loss_dict['loss_lm'].item():.4f} | "
                f"Avg Blocks: {outputs['avg_steps'].item():.2f} | "
                f"Elapsed: {elapsed:.2f}s"
            )

    # 4. Save Checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/dali1_prototype.pt"
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, checkpoint_path)
    print(f"\nModel checkpoint saved at: {checkpoint_path}")

if __name__ == "__main__":
    main()
