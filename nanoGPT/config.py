from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_embed: int = 768
    n_head: int = 12
    bias: bool = False
    dropout: float = 0.1
