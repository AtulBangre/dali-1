import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from .config import DALIConfig

class DALILoss(nn.Module):
    """
    Multi-objective loss function for DALI-1:
    L_total = L_LM + lambda_budget * L_budget + lambda_stab * L_stab
    """
    def __init__(self, config: DALIConfig, stability_weight: float = 0.001):
        super().__init__()
        self.config = config
        self.budget_weight = config.budget_penalty_weight
        self.stability_weight = stability_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        avg_steps: torch.Tensor,
        surprises: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            avg_steps: scalar tensor of steps taken
            surprises: (batch_size, seq_len)
        Returns:
            dict of loss values
        """
        # 1. Standard Causal Language Modeling Loss
        loss_lm = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        # 2. Budget Loss: penalize extra refinement steps to enforce efficiency
        loss_budget = avg_steps

        # 3. Stability Loss: penalize extreme variance in surprise to encourage smooth state updates
        loss_stab = torch.var(surprises)

        # Total Loss
        total_loss = loss_lm + (self.budget_weight * loss_budget) + (self.stability_weight * loss_stab)

        return {
            "loss": total_loss,
            "loss_lm": loss_lm,
            "loss_budget": loss_budget,
            "loss_stab": loss_stab
        }
