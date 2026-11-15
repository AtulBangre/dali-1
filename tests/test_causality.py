import torch
from dali.config import DALIConfig
from dali.model import DALI1Model

def test_causal_no_future_leakage():
    """
    Gate 1 Milestone Verification:
    Ensure that token predictions at time t are NOT affected by tokens at time > t.
    """
    config = DALIConfig(
        vocab_size=100,
        hidden_dim=64,
        state_dim=64,
        memory_slots=16,
        memory_dim=32,
        num_adaptive_layers=1
    )
    model = DALI1Model(config)
    model.eval()

    batch_size = 2
    seq_len = 16

    # Base sequence
    torch.manual_seed(42)
    seq_a = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Modified sequence: identical up to index 7, but completely different from index 8 onwards
    seq_b = seq_a.clone()
    seq_b[:, 8:] = torch.randint(0, config.vocab_size, (batch_size, seq_len - 8))

    with torch.no_grad():
        out_a = model(seq_a)
        out_b = model(seq_b)

    logits_a = out_a["logits"]
    logits_b = out_b["logits"]

    # Positions 0 to 7 must produce IDENTICAL logits
    diff_prefix = torch.max(torch.abs(logits_a[:, :8, :] - logits_b[:, :8, :])).item()
    print(f"Max prefix difference (must be 0.0): {diff_prefix:.8f}")

    assert diff_prefix < 1e-5, f"Causality Violation! Future tokens leaked into past predictions: diff={diff_prefix}"
    print("✅ Gate 1 PASSED: Causality verified! No future token leakage.")

if __name__ == "__main__":
    test_causal_no_future_leakage()
