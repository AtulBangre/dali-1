import torch
import math
from dali.config import DALIConfig
from dali.model import DALI1Model
from data.synthetic_memory import SyntheticFactMemoryDataset

def evaluate_checkpoint(checkpoint_path: str = "checkpoints/dali1_prototype.pt"):
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint["config"]
    model = DALI1Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = SyntheticFactMemoryDataset(vocab_size=config.vocab_size, seq_len=64)
    eval_batches = 20
    batch_size = 16

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    total_steps_executed = 0.0

    print("Evaluating on synthetic fact recall benchmark...")
    with torch.no_grad():
        for _ in range(eval_batches):
            input_ids, targets = dataset.get_batch(batch_size, device)
            outputs = model(input_ids, targets=targets)
            
            logits = outputs["logits"]
            # Cross entropy per token
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += targets.numel()

            # Accuracy
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += (preds == targets).sum().item()
            total_steps_executed += outputs["avg_steps"].item()

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    accuracy = (correct_tokens / total_tokens) * 100.0
    avg_blocks = total_steps_executed / eval_batches

    print("\n--- Evaluation Results ---")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Perplexity (PPL): {perplexity:.2f}")
    print(f"Next-Token Accuracy: {accuracy:.2f}%")
    print(f"Average Adaptive Blocks Used: {avg_blocks:.2f} / {config.num_adaptive_layers}")

if __name__ == "__main__":
    evaluate_checkpoint()
