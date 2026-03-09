import torch
import torch.nn.functional as F
import numpy as np

def softplus(x: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    return F.softplus(x, beta, threshold)

def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.exp(x.clamp_min(1e-6)) - 1.0)

def angle_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.acos(torch.sum(a * b, dim=-1).clamp(-0.9999, 0.9999))

def normalize_parameters(param, args, metadata, particle_type = "proton_contained"):
    
    #source_center_vec = np.array([0.01,0.02,-192.87]);

    # convert torch tensor to numpy array
    param[:,3] -= metadata['statistics']['per_tree'][particle_type]['true_iniekin']['mean']
    param[:,3] /= metadata['statistics']['per_tree'][particle_type]['true_iniekin']['std']
    param[:,0:3] /= (args.cube_size * 1.5)
    
    return param