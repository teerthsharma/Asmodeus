"""EraLM configuration. Every field has a default; `EraLMConfig(**json)` round-trips through checkpoints."""
from dataclasses import asdict, dataclass


@dataclass
class EraLMConfig:
    vocab: int = 4096
    # TT network (the softmax that is smart alone)
    d: int = 384
    layers: int = 6
    expand: float = 1.5
    conv: int = 4
    # era bank (ridge eras on the TT's features)
    bank: bool = True            # eras + decider on; False = TT alone
    n_max: int = 16              # era slots
    ridge: float = 1.0           # prior precision lam
    merge_rule: str = "anytime"  # "anytime" (L39d, merge_delta) or "bf1" (E0c reference)
    merge_delta: float = 0.05
    # decider (TT) over experts [eras..., TT alone, cache]
    decider: str = "MOM"         # SOFT | MOM | HARD | TEMP | GBAYES | MOM+GB
    alpha: float = 0.003         # share rate; G2 guaranteed for MOM when alpha <= 0.0078 (Lean, L38/L41)
    decider_p: float = None      # HARD threshold / TEMP temperature / GBAYES eta
    cache: int = 512             # Polya cache over the last `cache` tokens (0: off); base measure = TT
    cache_a: float = 4.0         # cache concentration on the TT base
    # engineering
    amp: bool = True             # bf16 autocast on CUDA
    compile: bool = False        # torch.compile the TT network
    grad_ckpt: bool = False      # activation checkpointing per block

    def to_dict(self):
        return asdict(self)
