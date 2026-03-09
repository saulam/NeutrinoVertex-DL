
from dataclasses import dataclass
from typing import Tuple
import torch
import math

@dataclass
class FitConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    
    #vertex activity image size(for the transformer)
    va_size: int = 7
    #image size(for the generator)
    img_size: int = 5
    
    # tracks
    N_max: int = 8

    # loss weights
    lam_a: float = 1e-3          # sparsity on amplitudes
    gamma_repel: float = 1e-2    # direction repulsion (optional)
    repel_sigma_deg: float = 10.0

    gamma_drift: float = 1e-2    # energy drift (optional)

    # optimization
    adam_lr: float = 5e-2
    adam_steps: int = 100
    lbfgs_steps: int = 50
    use_lbfgs: bool = False
    # pruning / merging
    prune_every: int = 200
    amp_prune_frac: float = 0.03
    amp_prune_min: float = 0.0
    merge_angle_deg: float = 5.0

    # number of samples for template
    n_sample_for_template: int = 64

    # background
    background_mode: str = "none"  # "scalar" or "map" or "none"

    # likelihood
    data_term: str = "KL_combined"  # "poisson" or "robust_l2" or "KL_combined"

    @property
    def repel_sigma_rad(self) -> float:
        return math.radians(self.repel_sigma_deg)

    @property
    def merge_angle_rad(self) -> float:
        return math.radians(self.merge_angle_deg)
