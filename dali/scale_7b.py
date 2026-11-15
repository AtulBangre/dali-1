"""
DALI-1: 7-Billion Parameter Architecture Specification & Scaling Blueprint
Defines parameter shapes, layer dimensions, memory bounds, and cluster training requirements.
"""

from dataclasses import dataclass

@dataclass
class DALI7BConfig:
    """Full 7-Billion Parameter Architecture Blueprint."""
    model_name: str = "DALI-1-7B"
    vocab_size: int = 32000
    hidden_dim: int = 4096
    state_dim: int = 4096
    
    # Layer depth
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    
    # Bounded Memory Bank (7B Scale)
    memory_slots: int = 256
    memory_dim: int = 1024
    memory_top_k: int = 4
    
    # Adaptive Compute
    num_adaptive_layers: int = 4
    max_seq_len: int = 4096
    dropout: float = 0.05
    tie_word_embeddings: bool = False

def calculate_7b_memory_stats(config: DALI7BConfig = DALI7BConfig()):
    """Calculates exact parameter counts and memory requirements across precision modes."""
    # Approximate parameter count:
    # Embeddings: vocab_size * hidden_dim
    emb_params = config.vocab_size * config.hidden_dim
    # Per-layer params (recurrent state + memory + feedforward): ~4 * hidden_dim^2 * 3
    layer_params = config.num_layers * (12 * (config.hidden_dim ** 2))
    # Memory bank & projections
    mem_params = config.memory_slots * config.memory_dim + 4 * (config.hidden_dim * config.memory_dim)
    # Output head
    head_params = config.hidden_dim * config.vocab_size
    
    total_params = emb_params + layer_params + mem_params + head_params

    # Precision calculations
    bytes_fp32 = total_params * 4
    bytes_fp16 = total_params * 2
    bytes_int4 = total_params * 0.5  # 4 bits = 0.5 byte

    # Training VRAM (Weights + Gradients + AdamW states)
    training_vram_fp32 = (total_params * 4) + (total_params * 4) + (total_params * 8)
    training_vram_fp16 = (total_params * 2) + (total_params * 2) + (total_params * 6)

    return {
        "total_parameters": total_params,
        "inference_fp32_gb": bytes_fp32 / (1024 ** 3),
        "inference_fp16_gb": bytes_fp16 / (1024 ** 3),
        "inference_int4_gb": bytes_int4 / (1024 ** 3),
        "training_vram_fp32_gb": training_vram_fp32 / (1024 ** 3),
        "training_vram_fp16_gb": training_vram_fp16 / (1024 ** 3)
    }

if __name__ == "__main__":
    stats = calculate_7b_memory_stats()
    print("=" * 65)
    print("      DALI-1 7-BILLION PARAMETER SCALING BLUEPRINT")
    print("=" * 65)
    print(f"Total Model Parameters:          {stats['total_parameters']:,} (~7.1 Billion)")
    print("-" * 65)
    print("INFERENCE MEMORY REQUIREMENTS:")
    print(f"  - 32-bit (FP32) Inference:     {stats['inference_fp32_gb']:.2f} GB RAM")
    print(f"  - 16-bit (FP16/BF16) Inference:{stats['inference_fp16_gb']:.2f} GB RAM")
    print(f"  - 4-bit (INT4) Inference:      {stats['inference_int4_gb']:.2f} GB RAM  <-- (FITS ON 16GB MAC!)")
    print("-" * 65)
    print("FULL PRE-TRAINING VRAM REQUIREMENTS (From Scratch):")
    print(f"  - FP32 Full Training:          {stats['training_vram_fp32_gb']:.2f} GB VRAM")
    print(f"  - FP16/BF16 Mixed Training:    {stats['training_vram_fp16_gb']:.2f} GB VRAM  (Requires 1-2x A100/H100 80GB)")
    print("=" * 65)
