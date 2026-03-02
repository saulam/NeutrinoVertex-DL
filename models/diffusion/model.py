import math
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_group_norm(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    a: (T,)
    t: (B,)
    returns: (B, 1, 1, 1, 1) broadcastable to x_shape
    """
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule from Nichol & Dhariwal.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

class ChargeTransform:
    """
    Recommended for sparse, nonnegative charge images.

    Example:
        tfm = ChargeTransform(scale=1.0)
        x_model = tfm.encode(x_raw)
        x_raw_recon = tfm.decode(x_model)

    If your charge values have a large dynamic range, tune `scale`.
    Often a dataset-level robust scale works well.
    """
    def __init__(self, scale: float = 1.0):
        self.scale = float(scale)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=0.0)
        return torch.log1p(x / self.scale)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid tiny negative values before expm1
        x = torch.clamp(x, min=0.0)
        return self.scale * torch.expm1(x)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) integer timesteps
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        if half_dim == 0:
            raise ValueError("time embedding dim must be >= 2")

        if half_dim == 1:
            freqs = torch.ones(1, device=t.device, dtype=torch.float32)
        else:
            freqs = torch.exp(
                torch.arange(half_dim, device=t.device, dtype=torch.float32)
                * (-math.log(10000.0) / (half_dim - 1))
            )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # If dim is odd, pad one extra zero
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))

        return emb
    

class TimeMLP(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(t_emb)
    

class KinematicsEncoder(nn.Module):
    def __init__(self, in_dim: int = 7, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, kin: torch.Tensor) -> torch.Tensor:
        """
        kin: (B, 7) normalized [inix, iniy, iniz, E, dx, dy, dz]
        """
        return self.net(kin)

class FiLM3D(nn.Module):
    """
    Feature-wise Linear Modulation:
    produce scale/shift and apply them.
    """
    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.to_scale_shift(cond)  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma[:, :, None, None, None]
        beta = beta[:, :, None, None, None]
        return x * (1 + gamma) + beta


# ----------------------------
# Fixed-resolution residual network
# ----------------------------

class ResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _make_group_norm(channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )

        self.norm2 = _make_group_norm(channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )

        self.film = FiLM3D(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.film(h, cond)
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return x + h


class UNet3DConditioned(nn.Module):
    """
    Kept same class name for compatibility, but now this is a fixed-resolution
    residual 3D network instead of a down/up U-Net.

    Better suited to 7x7x7 sparse track-like images.
    """
    def __init__(
        self,
        cond_dim: int,
        base_channels: int = 32,
        img_size: int = 7,
        num_blocks: int = 6,
        dilations: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size

        if dilations is None:
            # Small amount of dilation gives larger receptive field
            dilations = (1, 1, 2, 1, 2, 1)

        if len(dilations) != num_blocks:
            raise ValueError("len(dilations) must match num_blocks")

        c = base_channels
        self.in_conv = nn.Conv3d(1, c, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList(
            [ResBlock3D(c, cond_dim, dilation=d, dropout=dropout) for d in dilations]
        )

        self.out_norm = _make_group_norm(c)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_conv(x)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_conv(self.out_act(self.out_norm(h)))
        return h


# ----------------------------
# Diffusion model
# ----------------------------

class ConditionalDiffusionModel(nn.Module):
    def __init__(
        self,
        time_dim: int = 128,
        cond_dim: int = 128,
        base_channels: int = 32,
        num_timesteps: int = 1000,
        img_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.img_size = img_size

        # Better default schedule than linear
        betas = cosine_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1.0),
        )

        # embeddings
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim, cond_dim)
        self.kin_encoder = KinematicsEncoder(
            in_dim=7, hidden_dim=cond_dim, out_dim=cond_dim
        )

        # Better fusion than plain sum
        self.cond_fuse = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # fixed-resolution 3D network
        self.unet = UNet3DConditioned(
            cond_dim=cond_dim,
            base_channels=base_channels,
            img_size=img_size,
            num_blocks=6,
            dilations=(1, 1, 2, 1, 2, 1),
            dropout=dropout,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, kin: torch.Tensor) -> torch.Tensor:
        """
        x_t: (B, 1, img_size, img_size, img_size)
        t:   (B,) integer timesteps in [0, T-1]
        kin: (B, 7)
        returns: predicted noise eps
        """
        t_emb = self.time_embed(t)
        t_cond = self.time_mlp(t_emb)
        kin_cond = self.kin_encoder(kin)
        cond = self.cond_fuse(torch.cat([t_cond, kin_cond], dim=-1))
        eps_pred = self.unet(x_t, cond)
        return eps_pred

    # ----------------------------
    # Forward diffusion utilities
    # ----------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab = _extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_recip_ab = _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_ab = _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps

    # ----------------------------
    # Training loss
    # ----------------------------

    def loss(
        self,
        x0: torch.Tensor,
        kin: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        foreground_weight: float = 4.0,
        aux_x0_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        x0 should preferably already be in transformed space, e.g. log1p(charge / scale).

        Returns:
            total_loss, metrics_dict
        """
        B = x0.shape[0]
        device = x0.device

        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = self.q_sample(x0, t, noise)
        eps_pred = self.forward(x_t, t, kin)

        # Foreground-aware weighting: focus more on nonzero voxels
        foreground = (x0 > 0).float()
        weights = 1.0 + foreground_weight * foreground

        eps_sq = (eps_pred - noise) ** 2
        eps_loss = (eps_sq * weights).sum() / weights.sum().clamp_min(1.0)

        x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)
        x0_l1 = F.smooth_l1_loss(x0_pred, x0, reduction="none")
        x0_loss = (x0_l1 * weights).sum() / weights.sum().clamp_min(1.0)

        total = eps_loss + aux_x0_weight * x0_loss

        metrics = {
            "loss": total.detach(),
            "eps_loss": eps_loss.detach(),
            "x0_loss": x0_loss.detach(),
        }
        return total, metrics

    # ----------------------------
    # Sampling
    # ----------------------------

    def _make_sampling_timesteps(
        self,
        num_sampling_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a decreasing sequence of timesteps from T-1 to 0.
        If num_sampling_steps < self.num_timesteps, this is proper respacing.
        """
        num_sampling_steps = int(num_sampling_steps)
        if num_sampling_steps >= self.num_timesteps:
            return torch.arange(
                self.num_timesteps - 1, -1, -1, device=device, dtype=torch.long
            )

        ts = torch.linspace(
            self.num_timesteps - 1, 0, num_sampling_steps, device=device
        ).round().long()

        # Remove duplicates while preserving order
        unique_ts = [ts[0]]
        for i in range(1, len(ts)):
            if ts[i].item() != unique_ts[-1].item():
                unique_ts.append(ts[i])

        if unique_ts[-1].item() != 0:
            unique_ts.append(torch.tensor(0, device=device, dtype=torch.long))

        return torch.stack(unique_ts)

    @torch.no_grad()
    def sample(
        self,
        kin: torch.Tensor,
        num_samples: Optional[int] = None,
        device: Optional[str] = None,
        num_timesteps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        clamp_x0: bool = False,
    ) -> torch.Tensor:
        """
        DDIM-style sampling with optional timestep respacing.

        Args:
            kin: (B, 7) normalized kinematics
            num_samples: usually None; if None uses kin.shape[0]
            num_timesteps: number of sampling steps (can be < training T)
            eta: 0.0 = deterministic DDIM, >0 adds noise
            clamp_x0: if True, clamps x0 estimate to >= 0 in model space

        Returns:
            x0 sample in model space
        """
        self.eval()

        if device is None:
            device = kin.device
        else:
            device = torch.device(device)

        kin = kin.to(device)

        if num_samples is None:
            num_samples = kin.shape[0]

        if kin.shape[0] != num_samples:
            raise ValueError("kin.shape[0] must match num_samples")

        if noise is None:
            x = torch.randn(
                num_samples, 1, self.img_size, self.img_size, self.img_size, device=device
            )
        else:
            x = noise.to(device)

        n_steps = self.num_timesteps if num_timesteps is None else num_timesteps
        timesteps = self._make_sampling_timesteps(n_steps, device=device)

        for i, t_scalar in enumerate(timesteps):
            t = torch.full((num_samples,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.forward(x, t, kin)
            x0 = self.predict_x0_from_eps(x, t, eps)

            if clamp_x0:
                x0 = torch.clamp(x0, min=0.0)

            # Final step: return x0
            if i == len(timesteps) - 1:
                x = x0
                break

            next_t_scalar = timesteps[i + 1]

            alpha_bar_t = self.alphas_cumprod[t_scalar]
            alpha_bar_next = self.alphas_cumprod[next_t_scalar]

            if eta > 0.0:
                sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t)) * \
                        torch.sqrt(1 - alpha_bar_t / alpha_bar_next)
                z = torch.randn_like(x)
            else:
                sigma = torch.tensor(0.0, device=device, dtype=x.dtype)
                z = torch.zeros_like(x)

            c = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma ** 2, min=0.0))
            x = torch.sqrt(alpha_bar_next) * x0 + c * eps + sigma * z

        return x

    def differentiable_sample(
        self,
        kin: torch.Tensor,
        num_timesteps: Optional[int] = None,
        noise: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        clamp_x0: bool = False,
    ) -> torch.Tensor:
        """
        Differentiable DDIM-style sampler.

        kin:   (B, 7), gradients allowed
        noise: optional starting x_T; if None sampled once
        eta:   keep 0.0 if you want deterministic gradients
        """
        self.eval()

        B = kin.size(0)
        device = kin.device

        if noise is None:
            x = torch.randn(B, 1, self.img_size, self.img_size, self.img_size, device=device)
        else:
            x = noise

        n_steps = self.num_timesteps if num_timesteps is None else num_timesteps
        timesteps = self._make_sampling_timesteps(n_steps, device=device)

        for i, t_scalar in enumerate(timesteps):
            t = torch.full((B,), int(t_scalar.item()), device=device, dtype=torch.long)
            eps = self.forward(x, t, kin)
            x0 = self.predict_x0_from_eps(x, t, eps)

            if clamp_x0:
                x0 = torch.clamp(x0, min=0.0)

            if i == len(timesteps) - 1:
                x = x0
                break

            next_t_scalar = timesteps[i + 1]

            alpha_bar_t = self.alphas_cumprod[t_scalar]
            alpha_bar_next = self.alphas_cumprod[next_t_scalar]

            if eta > 0.0:
                sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar_t)) * \
                        torch.sqrt(1 - alpha_bar_t / alpha_bar_next)
                z = torch.randn_like(x)
            else:
                sigma = torch.tensor(0.0, device=device, dtype=x.dtype)
                z = torch.zeros_like(x)

            c = torch.sqrt(torch.clamp(1 - alpha_bar_next - sigma ** 2, min=0.0))
            x = torch.sqrt(alpha_bar_next) * x0 + c * eps + sigma * z

        return x
    