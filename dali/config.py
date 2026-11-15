from dataclasses import dataclass
from typing import Optional

@dataclass
class DALIConfig:
    vocab_size: int = 4096
    hidden_dim: int = 256
    state_dim: int = 256
    
    # Bounded Memory settings
    memory_slots: int = 64
    memory_dim: int = 128
    memory_top_k: int = 2
    memory_write_threshold: float = 0.5
    chunk_size: int = 8  # Chunk-level memory updates for high throughput and reduced write churn
    
    # Adaptive Compute settings
    num_base_layers: int = 2
    num_adaptive_layers: int = 2
    early_exit_confidence: float = 0.85
    budget_penalty_weight: float = 0.01
    
    # General
    max_seq_len: int = 1024
    dropout: float = 0.1
    tie_word_embeddings: bool = True
