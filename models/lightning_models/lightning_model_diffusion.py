"""
Project: "Fast generative proton showers with conditional diffusion"
Author: Dr. Saul Alonso-Monsalve
Description: PyTorch Lightning model for a conditional DDPM-style diffusion model.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LightningModelDiffusion(pl.LightningModule):
    def __init__(
        self,
        diffusion_model,
        lr: float = 1e-4,
        wd: float = 0.0,
        energy_loss_weight: float = 0.0,
    ):
        """
        Args:
            diffusion_model: an instance of ConditionalDiffusionModel (or compatible).
            lr: learning rate.
            wd: weight decay.
            energy_loss_weight: optional weight for an energy conservation penalty
                                (0.0 disables it).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["diffusion_model"])
        self.model = diffusion_model
        self.lr = lr
        self.wd = wd
        self.energy_loss_weight = energy_loss_weight

    def forward(self, x_t, t, labels):
        """
        Forward pass through the denoiser network.

        Args:
            x_t (Tensor): noisy voxel grid at timestep t, shape (B, 1, 5, 5, 5).
            t   (Tensor): integer timesteps, shape (B,).
            labels (Tensor): kinematics (inix, iniy, iniz, E, dx, dy, dz), shape (B, 7).

        Returns:
            Tensor: predicted noise ε̂ with same shape as x_t.
        """
        return self.model(x_t, t, labels)

    @torch.no_grad()
    def sample(self, labels, num_samples=None, device=None, num_timesteps=None):
        """
        Convenience wrapper to generate samples given kinematics.

        Args:
            labels (Tensor): kinematics (E, px, py, pz), shape (B, 4).
            num_samples (int, optional): if provided, uses this many samples
                                         and expects labels to be broadcastable
                                         (e.g. single label).
                                         If None, uses labels.shape[0].
            device: device to use; if None, use self.device.
            num_timesteps (int, optional): if provided, use fewer sampling steps
                                          (e.g., 500 instead of 1000).

        Returns:
            Tensor: generated voxel grids, shape (B, 1, 5, 5, 5).
        """
        if device is None:
            device = self.device

        labels = labels.to(device)

        if num_samples is None:
            num_samples = labels.size(0)
        else:
            # if user passes a single label and num_samples > 1, tile it
            if labels.size(0) == 1 and num_samples > 1:
                labels = labels.repeat(num_samples, 1)

        return self.model.sample(kin=labels, num_samples=num_samples, device=device, num_timesteps=num_timesteps)


    def training_step(self, batch, batch_idx):
        """
        One diffusion training step (denoising score matching).

        Args:
            batch: tuple (x0, labels)
                x0:     clean voxel grids, shape (B, 1, 5, 5, 5)
                labels: kinematics (inix, iniy, iniz, E, dx, dy, dz), shape (B, 7)
        """
        x0, labels = batch
        x0 = x0.to(self.device)
        labels = labels.to(self.device)

        B = x0.size(0)
        T = self.model.num_timesteps

        # Sample timesteps uniformly for each example
        t = torch.randint(low=0, high=T, size=(B,), device=self.device, dtype=torch.long)

        # Sample noise ε ~ N(0, I)
        eps = torch.randn_like(x0)

        # Get ᾱ_t for each sample (shape (B, 1, 1, 1, 1))
        alpha_bar_t = self.model.alphas_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Forward diffusion: x_t = sqrt(ᾱ_t) * x0 + sqrt(1-ᾱ_t) * ε
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps

        # Predict noise ε̂
        eps_pred = self(x_t, t, labels)

        # Main DDPM loss: MSE between ε and ε̂
        loss_mse = F.mse_loss(eps_pred, eps)

        loss = loss_mse

        # Optional: energy conservation penalty (using x0 prediction)
        if self.energy_loss_weight > 0.0:
            # Recover x0 estimate from x_t and ε̂:
            # x0_hat = (x_t - sqrt(1-ᾱ_t) * ε̂) / sqrt(ᾱ_t)
            x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

            # Sum over voxels -> total energy per event
            # shape: (B,)
            e_true = x0.view(B, -1).sum(dim=1)
            e_pred = x0_hat.view(B, -1).sum(dim=1)

            energy_loss = F.mse_loss(e_pred, e_true)
            loss = loss + self.energy_loss_weight * energy_loss

            self.log("train_energy_loss", energy_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_mse_loss", loss_mse, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - compute loss on validation set.
        Same computation as training but without gradient updates.
        """
        x0, labels = batch
        x0 = x0.to(self.device)
        labels = labels.to(self.device)

        B = x0.size(0)
        T = self.model.num_timesteps

        # Sample timesteps uniformly
        t = torch.randint(low=0, high=T, size=(B,), device=self.device, dtype=torch.long)

        # Sample noise
        eps = torch.randn_like(x0)

        # Get alpha values
        alpha_bar_t = self.model.alphas_cumprod[t].view(B, 1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

        # Forward diffusion
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * eps

        # Predict noise
        eps_pred = self(x_t, t, labels)

        # Compute loss
        loss_mse = F.mse_loss(eps_pred, eps)
        loss = loss_mse

        # Optional: energy conservation penalty
        if self.energy_loss_weight > 0.0:
            x0_hat = (x_t - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t
            e_true = x0.view(B, -1).sum(dim=1)
            e_pred = x0_hat.view(B, -1).sum(dim=1)
            energy_loss = F.mse_loss(e_pred, e_true)
            loss = loss + self.energy_loss_weight * energy_loss
            self.log("val_energy_loss", energy_loss, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        # Logging
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_mse_loss", loss_mse, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """
        Configure single optimizer for the diffusion model (no separate critic/generator).
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        return optimizer
