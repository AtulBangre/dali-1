import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dali.config import DALIConfig
from dali.model import DALI1Model
from data.synthetic_memory import SyntheticFactMemoryDataset

class StateOnlyDALI(nn.Module):
    """
    Ablation Variant A: State-only model (No bounded memory bank).
    Compresses everything into a single compact recurrent state without external retrieval.
    """
    def __init__(self, config: DALIConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.w_state = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)
        self.w_gate = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size)

    def forward(self, input_ids):
        B, T = input_ids.shape
        device = input_ids.device
        e = self.embedding(input_ids)
        state = torch.zeros(B, self.config.state_dim, device=device)
        all_logits = []
        for t in range(T):
            e_t = e[:, t, :]
            concat_in = torch.cat([e_t, state], dim=-1)
            cand_s = torch.tanh(self.w_state(concat_in))
            alpha = torch.sigmoid(self.w_gate(concat_in))
            state = (1.0 - alpha) * state + alpha * cand_s
            logits_t = self.lm_head(state)
            all_logits.append(logits_t)
        return {"logits": torch.stack(all_logits, dim=1)}

def train_and_eval(model, dataset, seed: int, steps: int = 150):
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device("cpu")
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)

    # Train
    for _ in range(steps):
        model.train()
        optimizer.zero_grad()
        inputs, targets, ans_indices = dataset.get_batch(16, device)
        logits = model(inputs)["logits"]
        loss_lm = F.cross_entropy(logits.view(-1, 1000), targets.view(-1), reduction='none').view(16, -1)
        qa_mask = torch.zeros_like(loss_lm)
        for b in range(16):
            idx = ans_indices[b].item()
            if idx >= 0:
                qa_mask[b, idx] = 10.0
        loss = (loss_lm * (1.0 + qa_mask)).mean()
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(30):
            inputs, targets, ans_indices = dataset.get_batch(16, device)
            preds = torch.argmax(model(inputs)["logits"], dim=-1)
            for b in range(16):
                idx = ans_indices[b].item()
                if idx >= 0:
                    total += 1
                    if preds[b, idx].item() == targets[b, idx].item():
                        correct += 1
    return (correct / total) * 100.0 if total > 0 else 0.0

def run_ablation_and_stability_test():
    print("=" * 70)
    print("  EXPERIMENT 2: COMPONENT ABLATION & MULTI-SEED REPRODUCIBILITY")
    print("=" * 70)
    print("Testing Hypothesis: Bounded Memory is strictly necessary to prevent")
    print("loss of exact facts across distractor sequences.")
    print("-" * 70)

    seeds = [42, 123, 999]
    dataset = SyntheticFactMemoryDataset(vocab_size=1000, seq_len=64)
    config = DALIConfig(vocab_size=1000, hidden_dim=128, state_dim=128, memory_slots=32, memory_dim=64, num_adaptive_layers=2)

    print(f"{'Random Seed':<15} | {'State-Only (No Memory)':<24} | {'Full DALI-1 (With Memory)':<24}")
    print("-" * 70)

    state_scores = []
    full_scores = []

    for seed in seeds:
        state_model = StateOnlyDALI(config)
        full_model = DALI1Model(config)

        score_state = train_and_eval(state_model, dataset, seed=seed)
        score_full = train_and_eval(full_model, dataset, seed=seed)

        state_scores.append(score_state)
        full_scores.append(score_full)

        print(f"Seed {seed:<10} | {score_state:<21.2f} % | {score_full:<21.2f} %")

    avg_state = sum(state_scores) / len(state_scores)
    avg_full = sum(full_scores) / len(full_scores)

    print("=" * 70)
    print(f"{'Mean Score':<15} | {avg_state:<21.2f} % | {avg_full:<21.2f} %")
    print("-" * 70)
    print(f"Scientific Finding: External Selective Memory provides a +{avg_full - avg_state:.1f}%")
    print("absolute accuracy improvement over State-only recurrence!")
    print("=" * 70)

if __name__ == "__main__":
    run_ablation_and_stability_test()
