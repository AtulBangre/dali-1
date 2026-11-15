import os
import sys
import torch
import torch.nn.functional as F
from typing import Optional

from dali.config import DALIConfig
from dali.model import DALI1Model
from dali.tokenizer import BytePairTokenizer

def generate_story(
    model: DALI1Model,
    tokenizer: BytePairTokenizer,
    prompt: str = "Once upon a time, Lily",
    max_new_tokens: int = 45,
    temperature: float = 0.6,
    top_k: int = 8,
    repetition_penalty: float = 1.2,
    device: Optional[torch.device] = None
) -> str:
    """
    Autoregressively generates text using DALI-1's predictive state & memory.
    Supports Subword Byte-Pair Encoding and exact spacing reconstruction.
    """
    if device is None:
        device = next(model.parameters()).device
        
    model.eval()
    
    # Encode prompt (without EOS)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = [tokenizer.vocab[tokenizer.bos_token]] + prompt_tokens
    generated = list(input_ids)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            outputs = model(curr_tensor)
            
            # Take logits for the last token position
            next_token_logits = outputs["logits"][:, -1, :].squeeze(0).clone()
            
            # Apply repetition penalty
            if repetition_penalty > 1.0:
                for token_id in set(generated):
                    if next_token_logits[token_id] > 0:
                        next_token_logits[token_id] /= repetition_penalty
                    else:
                        next_token_logits[token_id] *= repetition_penalty
                        
            # Temperature scaling
            next_token_logits = next_token_logits / max(temperature, 1e-5)
            
            # Top-k filtering
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < values[-1]] = float('-inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if EOS generated
            if next_token_id == tokenizer.vocab[tokenizer.eos_token]:
                break
                
            generated.append(next_token_id)

    return tokenizer.decode(generated)

def load_trained_model(
    checkpoint_path: str = "checkpoints/dali1_scaled_stories.pt",
    tokenizer_path: str = "checkpoints/bpe_tokenizer.json"
):
    device = torch.device("cpu")
    if not os.path.exists(checkpoint_path):
        if os.path.exists("checkpoints/dali1_bpe_stories.pt"):
            checkpoint_path = "checkpoints/dali1_bpe_stories.pt"
        else:
            print(f"Error: Checkpoint not found! Please run train_stories.py first.")
            sys.exit(1)

    tokenizer = BytePairTokenizer()
    tokenizer.load(tokenizer_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = DALI1Model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    ckpt_path = "checkpoints/dali1_scaled_stories.pt"
    tok_path = "checkpoints/bpe_tokenizer.json"
    
    print("=" * 65)
    print("      DALI-1 SUBWORD BPE AUTOREGRESSIVE GENERATION")
    print("=" * 65)
    print(f"Loading trained BPE model from: {ckpt_path}...")
    model, tokenizer = load_trained_model(ckpt_path, tok_path)
    print("Trained model & Subword BPE Tokenizer loaded successfully!\n")

    if len(sys.argv) > 1:
        user_prompt = " ".join(sys.argv[1:])
    else:
        user_prompt = "Once upon a time, Lily"

    print(f"Prompt: \"{user_prompt}\"")
    print("-" * 65)
    output_text = generate_story(
        model,
        tokenizer,
        prompt=user_prompt,
        max_new_tokens=45,
        temperature=0.6,
        top_k=8,
        repetition_penalty=1.2
    )
    print("Generated Story:")
    print(output_text)
    print("=" * 65)
