"""EraLM Configuration dataclass."""

from dataclasses import dataclass


@dataclass
class EraLMConfig:
    vocab: int = 4096
    d: int = 256
    layers: int = 4
    n_heads: int = 4
    alpha: float = 0.003
    max_eras: int = 32
    n_min: int = 64
    window_size: int = 32
    smoothing_a: float = 0.2
    amp: bool = False
    compile: bool = False
    grad_ckpt: bool = False
    lr: float = 3e-3
    weight_decay: float = 0.1
