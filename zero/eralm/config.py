from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EraLMConfig:
    vocab: int = 4096
    d: int = 256
    layers: int = 4
    n_heads: int = 8
    alpha: float = 0.003
    theta: float = 0.9
    amp: bool = False
    compile: bool = False
    grad_ckpt: bool = False
    device: str = "cpu"
