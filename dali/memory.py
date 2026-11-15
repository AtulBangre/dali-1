import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
from .config import DALIConfig

class BoundedMemoryBank(nn.Module):
    """
    Bounded Key-Value Memory Bank with top-k attention retrieval,
    causal surprise-gated writing, and slot management.
    """
    def __init__(self, config: DALIConfig):
        super().__init__()
        self.config = config
        self.num_slots = config.memory_slots
        self.memory_dim = config.memory_dim
        self.hidden_dim = config.hidden_dim
        self.state_dim = config.state_dim
        self.top_k = min(config.memory_top_k, config.memory_slots)

        # Query projection
        self.w_query = nn.Linear(config.hidden_dim + config.state_dim, config.memory_dim)
        
        # Write candidate projections
        self.w_key_cand = nn.Linear(config.hidden_dim + config.state_dim, config.memory_dim)
        self.w_val_cand = nn.Linear(config.hidden_dim + config.state_dim, config.memory_dim)
        
        # Write gate projection: predicts whether to write based on [surprise; uncertainty; emb]
        self.w_write_gate = nn.Linear(2 + config.hidden_dim, 1)

    def init_memory(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """Initialize empty memory bank for a batch."""
        return {
            "keys": torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device),
            "values": torch.zeros(batch_size, self.num_slots, self.memory_dim, device=device),
            "age": torch.zeros(batch_size, self.num_slots, device=device),
            "usage": torch.zeros(batch_size, self.num_slots, device=device)
        }

    def retrieve(
        self,
        token_emb: torch.Tensor,
        prev_state: torch.Tensor,
        memory: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query memory using top-k attention.
        Args:
            token_emb: (batch_size, hidden_dim)
            prev_state: (batch_size, state_dim)
            memory: dict holding keys and values
        Returns:
            retrieved_context: (batch_size, memory_dim)
            attn_weights: (batch_size, top_k)
        """
        batch_size = token_emb.size(0)
        q_input = torch.cat([token_emb, prev_state], dim=-1)
        query = self.w_query(q_input).unsqueeze(1)  # (B, 1, memory_dim)

        keys = memory["keys"]      # (B, num_slots, memory_dim)
        values = memory["values"]  # (B, num_slots, memory_dim)

        # Scaled dot-product attention
        scale = self.memory_dim ** -0.5
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) * scale  # (B, num_slots)

        # Top-k selection
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)  # (B, top_k)

        # Gather top-k values
        # topk_indices: (B, top_k) -> expand for gathering (B, top_k, memory_dim)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, self.memory_dim)
        topk_values = torch.gather(values, 1, idx_expanded)  # (B, top_k, memory_dim)

        # Weighted sum: (B, 1, top_k) x (B, top_k, memory_dim) -> (B, memory_dim)
        retrieved = torch.bmm(topk_weights.unsqueeze(1), topk_values).squeeze(1)

        # Update usage and age (non-inplace)
        new_age = memory["age"] + 1.0
        new_usage = memory["usage"].clone()
        new_usage = new_usage.scatter_add(1, topk_indices, topk_weights.detach())
        memory["age"] = new_age
        memory["usage"] = new_usage

        return retrieved, topk_weights

    def update_memory(
        self,
        token_emb: torch.Tensor,
        state: torch.Tensor,
        surprise: torch.Tensor,
        uncertainty: torch.Tensor,
        memory: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Causal write policy: writes candidate key/value into the memory bank
        if surprise and uncertainty exceed adaptive threshold.
        """
        batch_size = token_emb.size(0)
        
        # Write gate score in [0, 1]
        gate_input = torch.cat([surprise.unsqueeze(-1), uncertainty.unsqueeze(-1), token_emb], dim=-1)
        write_prob = torch.sigmoid(self.w_write_gate(gate_input)).squeeze(-1)  # (B,)

        cand_input = torch.cat([token_emb, state], dim=-1)
        k_cand = F.normalize(self.w_key_cand(cand_input), dim=-1)
        v_cand = self.w_val_cand(cand_input)

        # Eviction policy: pick oldest or least recently used slot
        # Priority = age / (usage + 1e-4)
        eviction_priority = memory["age"] / (memory["usage"] + 1e-4)
        target_slot = torch.argmax(eviction_priority, dim=-1)  # (B,)

        # Vectorized, non-in-place memory update to preserve autograd graph
        write_mask = (write_prob > self.config.memory_write_threshold).float().view(batch_size, 1, 1)
        slot_one_hot = F.one_hot(target_slot, num_classes=self.num_slots).float().unsqueeze(-1)  # (B, S, 1)
        update_weight = slot_one_hot * write_mask * write_prob.view(batch_size, 1, 1)  # (B, S, 1)

        k_cand_expanded = k_cand.unsqueeze(1)  # (B, 1, memory_dim)
        v_cand_expanded = v_cand.unsqueeze(1)  # (B, 1, memory_dim)

        # Functional non-inplace update
        memory["keys"] = memory["keys"] * (1.0 - update_weight) + update_weight * k_cand_expanded
        memory["values"] = memory["values"] * (1.0 - update_weight) + update_weight * v_cand_expanded
        
        # Reset age and usage for updated slots (non-inplace)
        reset_mask = (slot_one_hot * write_mask).squeeze(-1)  # (B, S)
        memory["age"] = memory["age"] * (1.0 - reset_mask)
        memory["usage"] = memory["usage"] * (1.0 - reset_mask)

        return write_prob
