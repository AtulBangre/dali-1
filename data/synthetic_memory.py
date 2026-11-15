import random
import torch
from typing import List, Tuple, Dict

class SyntheticFactMemoryDataset:
    """
    Generates synthetic fact-recall sequences with distractors to isolate
    and benchmark memory retrieval capabilities (as proposed in Appendix B).
    """
    def __init__(self, vocab_size: int = 1000, num_samples: int = 1000, seq_len: int = 64):
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        # Reserved token IDs
        self.pad_id = 0
        self.query_marker_id = 1
        self.answer_marker_id = 2
        
        # Entity and Attribute pools
        self.entities = list(range(10, 50))       # 40 distinct entities
        self.attributes = list(range(50, 100))    # 50 distinct attributes
        self.distractors = list(range(100, vocab_size))

    def generate_sample(self) -> Tuple[List[int], List[int], int]:
        """
        Creates a sequence:
        [Fact 1: E1, A1] ... [Distractors] ... [Fact N: EN, AN] ... [Distractors] ... [Query: 1, E_target, 2] -> [Answer: A_target]
        Returns:
            input_ids, targets, answer_index
        """
        num_facts = random.randint(2, 4)
        chosen_entities = random.sample(self.entities, num_facts)
        chosen_attributes = random.sample(self.attributes, num_facts)
        fact_dict = dict(zip(chosen_entities, chosen_attributes))

        tokens = []
        for entity, attr in fact_dict.items():
            tokens.extend([entity, attr])
            # Add 3 to 8 distractor tokens between facts
            num_dist = random.randint(3, 8)
            tokens.extend(random.choices(self.distractors, k=num_dist))

        # Target entity to recall (e.g. from the earliest facts)
        target_entity = chosen_entities[0]
        expected_answer = fact_dict[target_entity]

        # Query structure: [1, target_entity, 2, expected_answer]
        tokens.extend([self.query_marker_id, target_entity, self.answer_marker_id, expected_answer])

        # Pad to seq_len
        if len(tokens) > self.seq_len:
            tokens = tokens[:self.seq_len]
        else:
            tokens = tokens + [self.pad_id] * (self.seq_len - len(tokens))

        input_ids = tokens[:-1]
        targets = tokens[1:]

        # Find exact index of answer_marker_id (where the model must predict expected_answer)
        try:
            answer_idx = input_ids.index(self.answer_marker_id)
        except ValueError:
            answer_idx = -1

        return input_ids, targets, answer_idx

    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_inputs = []
        batch_targets = []
        batch_ans_indices = []
        for _ in range(batch_size):
            inp, tgt, ans_idx = self.generate_sample()
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            batch_ans_indices.append(ans_idx)

        return (
            torch.tensor(batch_inputs, dtype=torch.long, device=device),
            torch.tensor(batch_targets, dtype=torch.long, device=device),
            torch.tensor(batch_ans_indices, dtype=torch.long, device=device)
        )
