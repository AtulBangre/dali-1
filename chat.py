import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.quantization import Int4Linear
from dali.tokenizer import BytePairTokenizer

def convert_model_to_int4(module: torch.nn.Module):
    """Replaces nn.Linear layers with Int4Linear for state dict loading."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and child.in_features % 2 == 0:
            setattr(module, name, Int4Linear(child.in_features, child.out_features, bias=child.bias is not None))
        else:
            convert_model_to_int4(child)

def load_assistant(checkpoint_path: str = "checkpoints/dali1_coder_int4.pt"):
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "checkpoints/dali1_dev_14m_int4.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    
    tokenizer = BytePairTokenizer()
    tokenizer.load(ckpt["tokenizer_path"])

    model = DALI1Model(config)
    if ckpt.get("is_4bit", False):
        convert_model_to_int4(model)
        model.load_state_dict(ckpt["quantized_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()
    return model, tokenizer, config

def generate_response(
    model: DALI1Model,
    tokenizer: BytePairTokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.3,
    top_k: int = 3,
    repetition_penalty: float = 1.15
) -> str:
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    
    generated = list(tokens)
    eos_id = tokenizer.vocab.get(tokenizer.eos_token, -1)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_inputs = torch.tensor([generated[-160:]], dtype=torch.long)
            outputs = model(curr_inputs)
            logits = outputs["logits"][0, -1, :]

            # Repetition penalty
            for token_id in set(generated):
                logits[token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-4)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == eos_id:
                break

            generated.append(next_token)

    new_tokens = generated[len(tokens):]
    return tokenizer.decode(new_tokens).strip()

def main():
    print("=" * 68)
    print("  DALI-1 WEB & FULL-STACK CODE DEBUGGING ASSISTANT (4-BIT INT4)")
    print("  Type your prompt or error message. Type 'exit' to quit.")
    print("=" * 68)

    model, tokenizer, config = load_assistant()
    print("✓ 4-bit INT4 model loaded into RAM (~9.0 MB footprint)")
    print("-" * 68)

    # Check CLI arguments for one-shot prompt
    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
        prompt_formatted = f"<|user|>\n{user_prompt}\n<|assistant|>"
        response = generate_response(model, tokenizer, prompt_formatted)
        print(f"User: {user_prompt}\n")
        print(f"DALI-1:\n{response}")
        return

    # Interactive REPL
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting DALI-1 Chat. Goodbye!")
                break

            formatted_prompt = f"<|user|>\n{user_input}\n<|assistant|>"
            response = generate_response(model, tokenizer, formatted_prompt)
            print(f"\nDALI-1:\n{response}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting DALI-1 Chat.")
            break

if __name__ == "__main__":
    main()
