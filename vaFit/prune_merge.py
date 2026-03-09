from typing import List
import torch

from config import FitConfig
from fit_utils import angle_between

@torch.no_grad()
def prune_tracks(params, cfg: FitConfig) -> List[int]:
    """
    Prune tracks from the fit parameters.
    """
    c = params.constrained()
    weights = c['weights']
    active = (weights > cfg.amp_prune_min).nonzero(as_tuple=False).squeeze(-1).tolist()
    if len(active) == 0:
        return []

    med = torch.median(weights[active])
    thr = max(cfg.amp_prune_min, (cfg.amp_prune_frac * med).item())
    return (weights > thr).nonzero(as_tuple=False).squeeze(-1).tolist()

@torch.no_grad()
def merge_close_tracks(params, active: List[int], cfg: FitConfig) -> List[int]:
    """
    Merge close tracks from the fit parameters.
    """
    if len(active) <= 1:
        return active
    c = params.constrained()
    dir = c['dir'][active]
    weights = c['weights'][active]

    keep = True*len(active)
    for i in range(len(active)):
        if not keep[i]:
            continue
        for j in range(i+1, len(active)):
            if not keep[j]:
                continue
            ang = angle_between(dir[i], dir[j]).item()
            if ang < cfg.merge_angle_rad:
                if weights[i] > weights[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break
                
    return [idx for idx, k in zip(active, keep) if k]