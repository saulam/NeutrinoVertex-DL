import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) integer timesteps
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        # [half_dim]
        freqs = torch.exp(
            torch.arange(half_dim, device=t.device) * (-math.log(10000.0) / (half_dim - 1))
        )
        # (B, 1) * (1, half_dim) -> (B, half_dim)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb  # (B, dim)
    

class TimeMLP(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t_emb):
        return self.mlp(t_emb)

class KinematicsEncoder(nn.Module):
    def __init__(self, in_dim=7, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, kin):
        """
        kin: (B, 7) normalized [inix, iniy, iniz, E, dx, dy, dz]
        """
        return self.net(kin)


class FiLM3D(nn.Module):
    """
    Feature-wise Linear Modulation:
    given feature map f (B, C, D, H, W) and cond (B, cond_dim),
    produce scale/shift and apply them.
    """
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x, cond):
        # cond: (B, cond_dim)
        gamma_beta = self.to_scale_shift(cond)  # (B, 2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)  # each (B, C)
        # reshape to (B, C, 1, 1, 1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=4, num_channels=out_ch)
        self.act = nn.SiLU()
        self.film = FiLM3D(cond_dim, out_ch)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond)
        x = self.act(x)
        return x


class UNet3DConditioned(nn.Module):
    """
    Simple U-Net for configurable size (default 7x7x7):
    - down: 1 -> 32 -> 64
    - up:   64 -> 32 -> 1
    """
    def __init__(self, cond_dim, base_channels=32, img_size=7):
        super().__init__()
        self.img_size = img_size
        c = base_channels
        self.down1 = ConvBlock3D(1, c, cond_dim)
        self.down2 = ConvBlock3D(c, 2*c, cond_dim)

        self.pool = nn.MaxPool3d(2, stride=2, padding=0)  # will go 5^3 -> 2^3 approx (we’ll pad)

        self.mid = ConvBlock3D(2*c, 2*c, cond_dim)

        self.up1 = nn.ConvTranspose3d(2*c, c, kernel_size=2, stride=2)
        self.conv_up1 = ConvBlock3D(2*c, c, cond_dim)  # concat skip

        self.conv_out = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x, cond):
        # x: (B, 1, img_size, img_size, img_size)
        # pad to even dims for pooling if needed
        B, C, D, H, W = x.shape
        pad_d = (0, (D % 2))  # (left, right)
        pad_h = (0, (H % 2))
        pad_w = (0, (W % 2))
        x = F.pad(x, pad_w + pad_h + pad_d)  # PyTorch pad: (W_left, W_right, H_left, H_right, D_left, D_right)

        d1 = self.down1(x, cond)      # (B, c, 5/6?, 5/6?, 5/6?)
        d2_in = self.pool(d1)         # downsample
        d2 = self.down2(d2_in, cond)  # (B, 2c, ~2, ~2, ~2)

        m = self.mid(d2, cond)

        u1 = self.up1(m)              # (B, c, ~4, ~4, ~4)
        # crop/align d1 to u1 if size differs (for 5x5x5 it might):
        if d1.shape[2:] != u1.shape[2:]:
            d1 = F.interpolate(d1, size=u1.shape[2:], mode='trilinear', align_corners=False)

        u1_cat = torch.cat([u1, d1], dim=1)  # (B, 2c, ...)
        u1 = self.conv_up1(u1_cat, cond)

        out = self.conv_out(u1)  # (B, 1, D, H, W)
        # Crop back to img_size x img_size x img_size
        out = out[:, :, :self.img_size, :self.img_size, :self.img_size]
        return out


class ConditionalDiffusionModel(nn.Module):
    def __init__(self, 
                 time_dim=128, 
                 cond_dim=128, 
                 base_channels=32, 
                 num_timesteps=1000,
                 img_size=7):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.img_size = img_size

        # betas / alphas schedule (simple linear schedule as an example)
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # embeddings
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = TimeMLP(time_dim, cond_dim)
        self.kin_encoder = KinematicsEncoder(in_dim=7, hidden_dim=cond_dim, out_dim=cond_dim)

        # U-Net
        self.unet = UNet3DConditioned(cond_dim=cond_dim, base_channels=base_channels, img_size=img_size)

    def forward(self, x_t, t, kin):
        """
        x_t: (B, 1, img_size, img_size, img_size) noisy sample
        t:   (B,) integer timesteps in [0, T-1]
        kin: (B, 7) kinematics
        returns: predicted noise ε̂
        """
        t_emb = self.time_embed(t)         # (B, time_dim)
        t_cond = self.time_mlp(t_emb)      # (B, cond_dim)
        kin_cond = self.kin_encoder(kin)   # (B, cond_dim)
        cond = t_cond + kin_cond           # combine (you could also concat and linear)

        eps_pred = self.unet(x_t, cond)    # (B, 1, 5, 5, 5)
        return eps_pred

    @torch.no_grad()
    def sample(self, kin, num_samples=1, device="cuda", num_timesteps=None):
        """
        kin: (num_samples, 7) tensor of kinematics (already normalized)
        num_timesteps: if provided, use fewer sampling steps than training (e.g., 500 instead of 1000)
        """
        self.eval()
        T = num_timesteps if num_timesteps is not None else self.num_timesteps
        kin = kin.to(device)

        x_t = torch.randn(num_samples, 1, self.img_size, self.img_size, self.img_size, device=device)

        for t_step in reversed(range(T)):
            t = torch.full((num_samples,), t_step, device=device, dtype=torch.long)
            eps_theta = self.forward(x_t, t, kin)  # (B,1,5,5,5)

            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alphas_cumprod[t_step]
            beta_t = self.betas[t_step]

            # DDPM sampling step (simplified)
            if t_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = torch.zeros_like(x_t)

            # x_{t-1}
            x_t = (
                1 / torch.sqrt(alpha_t) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
                ) + torch.sqrt(beta_t) * noise
            )

        return x_t  # (num_samples, 1, 5, 5, 5) approximate samples of x0
    

    def differentiable_sample(self, kin, num_timesteps=None, noise=None):
        """
        kin: (B, 4), requires_grad=True allowed
        num_steps: optionally fewer than self.num_timesteps
        noise: optional (B, 1, 5, 5, 5) tensor; if None, sampled once
        """
        self.eval()
        T = self.num_timesteps if num_timesteps is None else num_timesteps
        B = kin.size(0)
        device = kin.device

        if noise is None:
            x_t = torch.randn(B, 1, self.img_size, self.img_size, self.img_size, device=device)
        else:
            x_t = noise

        for t_step in reversed(range(T)):
            t = torch.full((B,), t_step, device=device, dtype=torch.long)
            eps_theta = self.forward(x_t, t, kin)

            alpha_t = self.alphas[t_step]
            alpha_bar_t = self.alphas_cumprod[t_step]
            beta_t = self.betas[t_step]
            noise_step = torch.zeros_like(x_t)

            x_t = (
                1 / torch.sqrt(alpha_t) * (
                    x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta
                ) + torch.sqrt(beta_t) * noise_step
            )

        return x_t 
