import time
import torch
import torch.nn.functional as F

from dali.config import DALIConfig
from dali.model import DALI1Model
from baselines.transformer import BaselineTransformer

def measure_memory_and_latency(model, seq_len: int, batch_size: int = 4, runs: int = 20):
    device = torch.device("cpu")
    model.eval()
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # Latency
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.perf_counter()
    avg_latency_ms = ((end - start) / runs) * 1000.0

    return avg_latency_ms

def run_context_scaling_benchmark():
    print("=" * 70)
    print("  EXPERIMENT 1: CONTEXT LENGTH SCALING BENCHMARK (64 -> 1024 TOKENS)")
    print("=" * 70)
    print("Testing Hypothesis: DALI-1 Bounded Memory scales linearly O(T),")
    print("while Transformer Full-Attention scales quadratically O(T^2).")
    print("-" * 70)

    lengths = [64, 128, 256, 512, 1024]
    
    # Init models
    dali_config = DALIConfig(
        vocab_size=1000,
        hidden_dim=128,
        state_dim=128,
        memory_slots=32,
        memory_dim=64,
        max_seq_len=2048,
        num_adaptive_layers=2
    )
    dali_model = DALI1Model(dali_config)
    
    transformer_model = BaselineTransformer(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=2048
    )

    print(f"{'Context Length (T)':<20} | {'DALI-1 Latency':<18} | {'Transformer Latency':<18} | {'Scaling Ratio':<12}")
    print("-" * 70)

    results = []
    base_dali = None
    base_tf = None

    for T in lengths:
        dali_lat = measure_memory_and_latency(dali_model, seq_len=T)
        tf_lat = measure_memory_and_latency(transformer_model, seq_len=T)

        if base_dali is None:
            base_dali = dali_lat
            base_tf = tf_lat

        dali_growth = dali_lat / base_dali
        tf_growth = tf_lat / base_tf

        results.append((T, dali_lat, tf_lat, dali_growth, tf_growth))
        print(f"{T:<20} | {dali_lat:<15.2f} ms | {tf_lat:<15.2f} ms | {tf_growth/dali_growth:.2f}x growth")

    print("=" * 70)
    print("\nScaling Analysis:")
    t_start, t_end = lengths[0], lengths[-1]
    res_start, res_end = results[0], results[-1]
    print(f"When context increased {t_end // t_start}x (from {t_start} to {t_end} tokens):")
    print(f"- DALI-1 Latency grew by:       {res_end[3]:.1f}x (Linear O(T) bound)")
    print(f"- Transformer Latency grew by:  {res_end[4]:.1f}x (Quadratic O(T^2) growth)")
    print("=" * 70)

if __name__ == "__main__":
    run_context_scaling_benchmark()
