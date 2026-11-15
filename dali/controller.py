import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .config import DALIConfig

class AdaptiveRefinementBlock(nn.Module):
    """Refinement block used for conditional computation."""
    def __init__(self, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))

class AdaptiveComputeController(nn.Module):
    """
    Consistent learned controller for adaptive refinement and early exit.
    Maintains numerical representation parity between training and inference.
    """
    def __init__(self, config: DALIConfig):
        super().__init__()
        self.config = config
        self.num_layers = config.num_adaptive_layers
        self.blocks = nn.ModuleList([
            AdaptiveRefinementBlock(config.hidden_dim, config.dropout)
            for _ in range(self.num_layers)
        ])
        self.router = nn.Linear(config.hidden_dim, 1)

    def forward(
        self,
        hidden: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden.size(0)
        device = hidden.device
        
        current_h = hidden
        steps_executed = torch.zeros(batch_size, device=device)

        for block in self.blocks:
            current_h = block(current_h)
            steps_executed += 1.0
            
            # During inference, exit early if confident
            if not training:
                halt_prob = torch.sigmoid(self.router(current_h)).squeeze(-1)
                if (halt_prob >= self.config.early_exit_confidence).all():
                    break

        return current_h, steps_executed
