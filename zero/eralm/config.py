"""Configuration for EraLM (Era Language Model)."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EraLMConfig:
    vocab: int = 4096
    d: int = 256
    layers: int = 4
    heads: Optional[int] = None
    max_seq: int = 257
    alpha: float = 0.003
    amp: bool = False
    compile: bool = False
    grad_ckpt: bool = False
    device: str = "cpu"
