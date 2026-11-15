from dataclasses import dataclass
from dali.config import DALIConfig

@dataclass
class DALI1BConfig:
    """
    DALI-1 1-Billion (1B) Parameter Architectural Specification.
    Designed for consumer-hardware local deployment (MacBook 16GB RAM) via 4-bit INT4 quantization.
    """
    vocab_size: int = 32768
    hidden_dim: int = 2560
    state_dim: int = 2560
    memory_slots: int = 512
    memory_dim: int = 1024
    chunk_size: int = 64
    num_adaptive_layers: int = 36
    early_exit_confidence: float = 0.90
    budget_penalty_weight: float = 0.005
    max_seq_len: int = 4096
    dropout: float = 0.05
    tie_word_embeddings: bool = True

    def calculate_hardware_budget(self) -> dict:
        # Approximate parameter counts
        embed_params = self.vocab_size * self.hidden_dim
        recurrent_params = 4 * (self.hidden_dim * self.state_dim + self.state_dim * self.state_dim)
        memory_params = 3 * (self.state_dim * self.memory_dim)
        adaptive_layer_params = self.num_adaptive_layers * (4 * self.hidden_dim * self.hidden_dim)
        total_params = embed_params + recurrent_params + memory_params + adaptive_layer_params

        # Memory footprints
        fp32_ram_gb = (total_params * 4) / (1024 ** 3)
        fp16_ram_gb = (total_params * 2) / (1024 ** 3)
        int4_ram_gb = (total_params * 0.5) / (1024 ** 3)

        return {
            "total_parameters": total_params,
            "parameter_millions": total_params / 1e6,
            "fp32_weights_gb": round(fp32_ram_gb, 2),
            "fp16_weights_gb": round(fp16_ram_gb, 2),
            "int4_quantized_ram_gb": round(int4_ram_gb, 2),
            "mac_16gb_ram_utilization_pct": round((int4_ram_gb / 16.0) * 100, 1)
        }

if __name__ == "__main__":
    cfg = DALI1BConfig()
    budget = cfg.calculate_hardware_budget()
    print("=" * 68)
    print("  DALI-1 1-BILLION (1B) PARAMETER SCALE BLUEPRINT")
    print("=" * 68)
    print(f"Total Parameters:          {budget['total_parameters']:,} (~{budget['parameter_millions']:.1f}M)")
    print(f"FP32 Precision Size:       {budget['fp32_weights_gb']} GB")
    print(f"FP16 Half-Precision Size:  {budget['fp16_weights_gb']} GB")
    print(f"Native 4-Bit INT4 Size:    {budget['int4_quantized_ram_gb']} GB (~{budget['int4_quantized_ram_gb']*1024:.0f} MB)")
    print(f"Mac 16GB RAM Footprint:    {budget['mac_16gb_ram_utilization_pct']}% of total RAM")
    print("=" * 68)
