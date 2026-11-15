import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.tokenizer import BytePairTokenizer
from dali.quantization import Int4Linear, quantize_model_to_4bit
from dali.scale_1b import DALI1BConfig

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ Active Compute: NVIDIA CUDA GPU ({gpu_name}) | Total VRAM: {vram_gb:.2f} GB")
    else:
        device = torch.device("cpu")
        print("⚠ CUDA not detected. Running on CPU.")
    return device

def prepare_1b_model(vocab_size: int = 1024, max_seq_len: int = 1024):
    """
    Constructs DALI-1 1-Billion Parameter Model for Free Google Colab T4 (16GB VRAM).
    Uses sub-quadratic bounded memory architecture with gradient-friendly dimensions.
    """
    config = DALIConfig(
        vocab_size=vocab_size,
        hidden_dim=2560,
        state_dim=2560,
        memory_slots=512,
        memory_dim=1024,
        chunk_size=64,
        num_adaptive_layers=24,  # Scaled for 16GB VRAM Colab T4 headroom
        early_exit_confidence=0.92,
        budget_penalty_weight=0.005,
        max_seq_len=max_seq_len,
        dropout=0.05,
        tie_word_embeddings=True
    )
    model = DALI1Model(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 72)
    print(f"  DALI-1 1-BILLION (1B) SCALE MODEL INITIALIZED")
    print("=" * 72)
    print(f"Total Parameters:      {total_params:,} (~{total_params/1e6:.1f}M)")
    print(f"Trainable Parameters:  {trainable_params:,}")
    print(f"FP16 Weights in VRAM:  ~{(total_params * 2) / (1024**3):.2f} GB")
    print(f"Target 4-bit INT4:     ~{(total_params * 0.5) / (1024**3):.2f} GB (~{(total_params * 0.5)/(1024**2):.0f} MB)")
    print("=" * 72)
    return model, config

def train_1b_colab(
    data_path: str = "knowledge_base/master_web_corpus.txt",
    tokenizer_path: str = "checkpoints/code_bpe.json",
    output_dir: str = "checkpoints",
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    lr: float = 3e-4
):
    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Tokenizer
    print("✓ Loading Code-Preserving Tokenizer...")
    tokenizer = BytePairTokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        return

    # 2. Initialize 1B Architecture
    model, config = prepare_1b_model(vocab_size=tokenizer.vocab_size)
    model.to(device)

    # 3. Load Training Dataset
    if not os.path.exists(data_path):
        # Fallback to any master corpus
        for alt in ["FULL_STACK_MASTER_CORPUS.txt", "data/final_coding_corpus.txt"]:
            if os.path.exists(alt):
                data_path = alt
                break

    print(f"✓ Loading Training Dataset from '{data_path}'...")
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    tokens = tokenizer.encode(text, add_special_tokens=True)
    print(f"✓ Total Tokens: {len(tokens):,}")

    seq_len = 256
    chunks = [tokens[i:i+seq_len] for i in range(0, len(tokens)-seq_len, seq_len)]
    if not chunks:
        print("Error: Dataset too short for seq_len")
        return

    # 4. Optimizer & Mixed Precision Scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    total_steps = len(chunks) // batch_size

    print(f"✓ Training Sprint: {epochs} Epochs | Batch Size: {batch_size} | Accumulation: {grad_accum_steps}")
    print("-" * 72)

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        optimizer.zero_grad()

        for i in range(0, len(chunks) - batch_size + 1, batch_size):
            batch_chunks = chunks[i:i+batch_size]
            input_ids = torch.tensor(batch_chunks, dtype=torch.long, device=device)
            target_ids = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            # Mixed Precision Forward Pass (AMP)
            with autocast(dtype=torch.float16):
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
                loss = loss / grad_accum_steps

            # Backward Pass with Scaler
            scaler.scale(loss).backward()

            if (steps + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loss_val = loss.item() * grad_accum_steps
            epoch_loss += loss_val
            steps += 1

            if steps % 10 == 0 or steps == total_steps:
                elapsed = time.time() - start_time
                vram_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Step {steps:03d}/{total_steps:03d} | Loss: {loss_val:.4f} | VRAM: {vram_used:.2f}GB | Elapsed: {elapsed:.1f}s")

    # 5. Save FP16 Checkpoint
    fp16_path = os.path.join(output_dir, "dali1_1b_fp16.pt")
    print(f"\n✓ Saving Full-Precision FP16 Checkpoint to '{fp16_path}'...")
    torch.save({
        "config": config,
        "model_state_dict": model.state_dict(),
        "tokenizer_path": tokenizer_path
    }, fp16_path)

    # 6. Quantize to Native 4-bit INT4
    print("✓ Quantizing 1B Model to Native 4-Bit INT4 (<550MB RAM)...")
    quant_model, comp_ratio = quantize_model_to_4bit(model)
    int4_ckpt = {
        "config": config,
        "quantized_state_dict": quant_model.state_dict(),
        "tokenizer_path": tokenizer_path,
        "is_4bit": True
    }
    int4_path = os.path.join(output_dir, "dali1_1b_int4.pt")
    torch.save(int4_ckpt, int4_path)

    int4_size_mb = os.path.getsize(int4_path) / (1024**2)
    print("=" * 72)
    print("🎉 1-BILLION (1B) MODEL TRAINING & QUANTIZATION COMPLETE!")
    print(f"📁 4-Bit INT4 Checkpoint:  {int4_path}")
    print(f"💾 File Size on Disk:      {int4_size_mb:.1f} MB (Runs on Mac with <550MB RAM!)")
    print("=" * 72)

    # 7. Check if Google Drive is available to copy
    drive_dir = "/content/drive/MyDrive/dali_checkpoints"
    if os.path.exists("/content/drive/MyDrive"):
        os.makedirs(drive_dir, exist_ok=True)
        import shutil
        dest = os.path.join(drive_dir, "dali1_1b_int4.pt")
        shutil.copy2(int4_path, dest)
        print(f"✓ Backup Copied to Google Drive: '{dest}'")

if __name__ == "__main__":
    train_1b_colab()
