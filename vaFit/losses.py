import torch
import torch.nn.functional as F

def poisson_nll(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-6)
    return F.poisson_nll_loss(x_recon, x, log_input=False, full=True)

def KL_divergence(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # softmax the x_recon and x
    x_recon = F.softmax(x_recon, dim=-1)
    x = F.softmax(x, dim=-1)
    # compute the KL divergence
    return F.kl_div(x_recon, x, log_target=True, reduction='batchmean').sum()

def charbonnier(residual: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(residual**2 + eps**2)

def robust_l2(x_recon: torch.Tensor, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return charbonnier(x_recon - x, eps=eps).sum()

def data_loss(x: torch.Tensor, x_recon: torch.Tensor, data_term: str) -> torch.Tensor:
    if data_term == "poisson":
        return poisson_nll(x_recon, x)
    elif data_term == "robust_l2":
        return robust_l2(x_recon, x)
    elif data_term == "KL_combined":
        return KL_divergence(x_recon, x) + robust_l2(x_recon, x)
    else:
        raise ValueError(f"Invalid data term: {data_term}")