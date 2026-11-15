import torch
import torch.nn as nn
from typing import Optional, Tuple
from .config import DALIConfig

class PredictiveStateEngine(nn.Module):
    """
    Compact predictive state engine with input-dependent dynamic gating.
    Computes candidate state update and balances retention vs overwrite.
    """
    def __init__(self, config: DALIConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.state_dim = config.state_dim
        self.memory_dim = config.memory_dim

        # Candidate state projection: takes [e_t; s_{t-1}; r_t]
        input_dim = config.hidden_dim + config.state_dim + config.memory_dim
        self.w_candidate = nn.Linear(input_dim, config.state_dim)

        # Gate projection: determines update strength alpha_t in (0, 1)
        gate_input_dim = config.hidden_dim + config.state_dim
        self.w_gate = nn.Linear(gate_input_dim, config.state_dim)
        
        self.layer_norm = nn.LayerNorm(config.state_dim)
        self.dropout = nn.Dropout(config.dropout)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize compact state to zeros."""
        return torch.zeros(batch_size, self.state_dim, device=device)

    def forward(
        self,
        token_emb: torch.Tensor,
        prev_state: torch.Tensor,
        retrieved_memory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step-wise causal state update:
        Args:
            token_emb: (batch_size, hidden_dim)
            prev_state: (batch_size, state_dim)
            retrieved_memory: (batch_size, memory_dim)
        Returns:
            new_state: (batch_size, state_dim)
            gate: (batch_size, state_dim)
        """
        # Concatenate inputs for candidate state
        concat_u = torch.cat([token_emb, prev_state, retrieved_memory], dim=-1)
        candidate_u = torch.tanh(self.w_candidate(concat_u))

        # Dynamic retention / overwrite gate
        concat_gate = torch.cat([token_emb, prev_state], dim=-1)
        gate_alpha = torch.sigmoid(self.w_gate(concat_gate))

        # Gated state update
        new_state = (1.0 - gate_alpha) * prev_state + gate_alpha * candidate_u
        new_state = self.layer_norm(new_state)
        new_state = self.dropout(new_state)

        return new_state, gate_alpha
