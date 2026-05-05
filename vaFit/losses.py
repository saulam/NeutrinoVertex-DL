import torch
import torch.nn.functional as F

def poisson_nll(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-6)
    return F.poisson_nll_loss(x_recon, x, log_input=False, full=True)

def KL_divergence(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # softmax the x_recon and x
    log_p = F.log_softmax(x_recon, dim=-1)
    q = F.softmax(x, dim=-1)
    # compute the KL divergence
    return F.kl_div(log_p, q, reduction='batchmean')

def charbonnier(residual: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(residual**2 + eps**2)

def robust_l2(x_recon: torch.Tensor, x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    
    return charbonnier(x_recon - x, eps=eps).sum()/x.shape[-1]

def capped_log_loss(x_recon, x, cap=500, lambda_over=0.2, eps=1e-6):
    # non-negative domain for charges
    x = x.clamp_min(0.0)
    x_recon = x_recon.clamp_min(0.0)

    max_cap = max(x.max()*1.5, cap)
    # smooth cap (gradient-friendly)
    x_recon_cap = max_cap * torch.tanh(x_recon / max_cap)

    # log-space residual
    r = torch.log1p(x_recon_cap + eps) - torch.log1p(x + eps)
    fit = torch.sqrt(r * r + 1e-6).mean()

    # asymmetric penalty for overshoot
    over = torch.relu(x_recon_cap - x)
    over_pen = (over * over).mean()

    return fit + lambda_over * over_pen
def weighted_poisson_loss(x_recon, x, tau=0.5, w_bg=0.05, eps=0.5):
    # x and x_recon must be non-negative, same charge scale
    x = x.clamp_min(0.0)
    rate = x_recon.clamp_min(eps)

    hit_mask = (x > tau).to(x.dtype)
    voxel_w = w_bg + (1.0 - w_bg) * hit_mask
    print("hit_mask: ", hit_mask)
    print("voxel_w: ", voxel_w)
    print("rate: ", rate)
    print("x: ", x)
    # per-voxel NLL
    nll = rate - x * torch.log(rate)
    print("nll: ", nll)
    return (voxel_w * nll).sum(dim=-1).mean()
def data_loss(x: torch.Tensor, x_recon: torch.Tensor, data_term: str) -> torch.Tensor:
    print("data_term: ", data_term)
    if data_term == "poisson":
        return poisson_nll(x_recon, x)
    elif data_term == "robust_l2":
        return robust_l2(x_recon, x)
    elif data_term == "KL_combined":
        return KL_divergence(x_recon, x) + robust_l2(x_recon, x)
    elif data_term == "weighted_poisson":
        return weighted_poisson_loss(x_recon, x)
    elif data_term == "capped_log":
        return capped_log_loss(x_recon, x)
    else:
        raise ValueError(f"Invalid data term: {data_term}")