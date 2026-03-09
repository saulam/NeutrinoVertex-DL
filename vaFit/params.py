from typing import Dict, Optional
import torch
import torch.nn as nn

from fit_utils import inv_softplus, softplus

class FitParams(nn.Module):
    def __init__(
        self,
        n_max: int,
        init_dir: torch.Tensor,
        init_ekin: torch.Tensor,
        init_vtx: torch.Tensor,
        init_weights: Optional[torch.Tensor],
        background_mode: str,
        va_size: int,
        device: str,
        dtype: torch.dtype,
        more_particles: bool = False,
        E_mean: float = 0.0,
        E_std: float = 1.0,
    ):
        super().__init__()
        self.n_max = n_max
        self.init_dir = init_dir.to(device, dtype)
        self.background_mode = background_mode
        self.va_size = va_size
        self.more_particles = more_particles
        # direction parameters
        dir = torch.randn(n_max, 3, device=device, dtype=dtype)
        dir /= (dir.norm(dim=-1, keepdim=True) + 1e-6)
        dir[:init_dir.shape[0]] = self.init_dir

        self.dir = nn.Parameter(dir)

        self.E_mean = E_mean
        self.E_std = E_std

        # energy parameters
        ekin = torch.randn(n_max, 1, device=device, dtype=dtype)
        ekin[:init_ekin.shape[0]] = init_ekin.to(device, dtype)
        self.ekin = nn.Parameter(ekin)

        # vertex parameters
        self.vtx = nn.Parameter(init_vtx.to(device, dtype))
       
        # track weights if more_particles is True
        if more_particles:
            if init_weights is None:
                w_init = torch.full((n_max,), -3.0, device=device, dtype=dtype)
            else:
                w_init = torch.full((n_max,), -3.0, device=device, dtype=dtype)
                w_init[:init_weights.shape[0]] = inv_softplus(init_weights.to(device, dtype).clamp_min(1e-6))
            self.w = nn.Parameter(w_init)
        else:
            self.w = None

        if self.background_mode == "scalar":
            self.background = nn.Parameter(torch.tensor(0.0, device=device, dtype=dtype))
        elif self.background_mode == "map":
            self.background = nn.Parameter(torch.randn(self.va_size*self.va_size*self.va_size, device=device, dtype=dtype))
        elif self.background_mode == "none":
            self.background = None
        else:
            raise ValueError(f"Invalid background mode: {self.background_mode}")

    def constrained(self) -> Dict[str, torch.Tensor]:
        if self.more_particles:
            weights = softplus(self.w)
        else:
            weights = None
        b = self.background if self.background is not None else None

        dir = self.dir / (self.dir.norm(dim=-1, keepdim=True) + 1e-8)

        ekin = softplus(self.ekin * self.E_std + self.E_mean)
        ekin = (ekin - self.E_mean) / self.E_std

        return {
            "dir": dir,
            "E": ekin,
            "vtx": self.vtx,
            "weights": weights,
            "b": b,
        }
        