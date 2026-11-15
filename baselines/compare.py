import time
import math
import torch
import torch.nn.functional as F
import torch.optim as optim

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.losses import DALILoss
from baselines.transformer import BaselineTransformer
from data.synthetic_memory import SyntheticFactMemoryDataset

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_inference_latency(model: torch.nn.Module, device: torch.device, seq_len: int = 64, runs: int = 50) -> float:
    model.eval()
    dummy_input = torch.randint(0, 1000, (1, seq_len), device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)
    end = time.perf_counter()
    return ((end - start) / runs) * 1000.0

def evaluate_fact_recall(model, dataset, device, eval_batches=40, batch_size=16):
    """
    Evaluates True Fact Recall:
    Checks if model outputs the exact attribute when queried for an entity.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, targets, ans_indices = dataset.get_batch(batch_size, device)
            logits = model(inputs)["logits"]
            preds = torch.argmax(logits, dim=-1)
            
            for b in range(batch_size):
                idx = ans_indices[b].item()
                if idx >= 0:
                    total += 1
                    if preds[b, idx].item() == targets[b, idx].item():
                        correct += 1
    return (correct / total) * 100.0 if total > 0 else 0.0

def run_comparative_experiment():
    device = torch.device("cpu")
    print("=" * 65)
    print("  DALI-1 vs BASELINE TRANSFORMER: 5-METRIC COMPREHENSIVE BENCHMARK")
    print("=" * 65)

    vocab_size = 1000
    hidden_dim = 128
    seq_len = 64
    batch_size = 16
    training_steps = 150

    dali_config = DALIConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        state_dim=hidden_dim,
        memory_slots=32,
        memory_dim=64,
        memory_top_k=2,
        num_adaptive_layers=2,
        dropout=0.05
    )
    dali_model = DALI1Model(dali_config).to(device)
    dali_loss_fn = DALILoss(dali_config).to(device)
    dali_optim = optim.AdamW(dali_model.parameters(), lr=2e-3, weight_decay=0.01)

    transformer_model = BaselineTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_heads=4,
        max_seq_len=1024,
        dropout=0.05
    ).to(device)
    transformer_optim = optim.AdamW(transformer_model.parameters(), lr=2e-3, weight_decay=0.01)

    print(f"DALI-1 Parameters:             {count_parameters(dali_model):,}")
    print(f"Baseline Transformer Params:   {count_parameters(transformer_model):,}")
    print("-" * 65)

    torch.manual_seed(42)
    dataset = SyntheticFactMemoryDataset(vocab_size=vocab_size, seq_len=seq_len)

    # 1. Train DALI-1
    print(f"\n[1/2] Training DALI-1 ({training_steps} steps)...")
    start_dali = time.time()
    dali_final_loss = 0.0
    for step in range(1, training_steps + 1):
        dali_model.train()
        dali_optim.zero_grad()
        inputs, targets, ans_indices = dataset.get_batch(batch_size, device)
        
        outputs = dali_model(inputs, targets=targets)
        logits = outputs["logits"]
        
        loss_lm = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none').view(batch_size, -1)
        qa_mask = torch.zeros_like(loss_lm)
        for b in range(batch_size):
            idx = ans_indices[b].item()
            if idx >= 0:
                qa_mask[b, idx] = 10.0
                
        loss = (loss_lm * (1.0 + qa_mask)).mean() + dali_config.budget_penalty_weight * outputs["avg_steps"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dali_model.parameters(), max_norm=1.0)
        dali_optim.step()
        dali_final_loss = loss.item()

        if step % 50 == 0:
            print(f"  DALI-1 Step {step:03d} | Loss: {dali_final_loss:.4f}")
    dali_train_time = time.time() - start_dali

    # 2. Train Transformer Baseline
    print(f"\n[2/2] Training Baseline Transformer ({training_steps} steps)...")
    start_tf = time.time()
    tf_final_loss = 0.0
    for step in range(1, training_steps + 1):
        transformer_model.train()
        transformer_optim.zero_grad()
        inputs, targets, ans_indices = dataset.get_batch(batch_size, device)
        
        outputs = transformer_model(inputs, targets=targets)
        logits = outputs["logits"]
        
        loss_lm = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), reduction='none').view(batch_size, -1)
        qa_mask = torch.zeros_like(loss_lm)
        for b in range(batch_size):
            idx = ans_indices[b].item()
            if idx >= 0:
                qa_mask[b, idx] = 10.0
                
        loss = (loss_lm * (1.0 + qa_mask)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), max_norm=1.0)
        transformer_optim.step()
        tf_final_loss = loss.item()

        if step % 50 == 0:
            print(f"  Transformer Step {step:03d} | Loss: {tf_final_loss:.4f}")
    tf_train_time = time.time() - start_tf

    # 3. Benchmark Metrics
    print("\nMeasuring True Fact Recall QA Accuracy...")
    dali_qa_acc = evaluate_fact_recall(dali_model, dataset, device)
    tf_qa_acc = evaluate_fact_recall(transformer_model, dataset, device)

    print("Measuring Inference Latency (seq_len=64)...")
    dali_latency = benchmark_inference_latency(dali_model, device, seq_len=seq_len)
    tf_latency = benchmark_inference_latency(transformer_model, device, seq_len=seq_len)

    print("\n" + "=" * 65)
    print("                 BENCHMARK SUMMARY RESULTS (ALL 5 METRICS)")
    print("=" * 65)
    print(f"{'Metric':<32} | {'DALI-1':<13} | {'Transformer':<13}")
    print("-" * 65)
    print(f"{'Trainable Parameters':<32} | {count_parameters(dali_model):<13,d} | {count_parameters(transformer_model):<13,d}")
    print(f"{'Training Time (150 steps)':<32} | {dali_train_time:<11.2f}s | {tf_train_time:<11.2f}s")
    print(f"{'Final Training Loss':<32} | {dali_final_loss:<13.4f} | {tf_final_loss:<13.4f}")
    print(f"{'True Fact Recall QA Accuracy':<32} | {dali_qa_acc:<12.2f}% | {tf_qa_acc:<12.2f}%")
    print(f"{'Inference Latency (per seq)':<32} | {dali_latency:<10.2f}ms | {tf_latency:<10.2f}ms")
    print("=" * 65)

if __name__ == "__main__":
    run_comparative_experiment()
