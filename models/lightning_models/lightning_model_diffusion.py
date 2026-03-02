import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional


class LightningModelDiffusion(pl.LightningModule):
    def __init__(
        self,
        diffusion_model,
        lr: float = 1e-4,
        wd: float = 0.0,
        energy_loss_weight: float = 0.0,
        foreground_weight: float = 4.0,
        aux_x0_weight: float = 0.1,
        charge_transform: Optional[object] = None,
        clamp_x0_for_energy: bool = True,
        sample_eta: float = 0.0,
        sample_clamp_x0: bool = True,
    ):
        """
        Args:
            diffusion_model:
                An instance of the new ConditionalDiffusionModel.

            lr:
                Learning rate.

            wd:
                Weight decay.

            energy_loss_weight:
                Optional weight for an energy conservation penalty.
                If > 0, compares total deposited charge in x0_pred vs x0_true.

            foreground_weight:
                Extra weight on voxels where x0 > 0 in model space.
                Used for epsilon loss and auxiliary x0 loss.

            aux_x0_weight:
                Weight of auxiliary x0 reconstruction loss.

            charge_transform:
                Optional transform object with:
                    encode(x) -> model-space tensor
                    decode(x) -> raw-space tensor
                For example: ChargeTransform(scale=1.0)

                If None, inputs are assumed already in model space.

            clamp_x0_for_energy:
                If True, clamp predicted x0 to >= 0 before decoding / energy sum.

            sample_eta:
                Default eta passed to DDIM-style sampling.
                0.0 = deterministic.

            sample_clamp_x0:
                Whether to clamp x0 >= 0 during sampling.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["diffusion_model", "charge_transform"])

        self.model = diffusion_model
        self.lr = lr
        self.wd = wd

        self.energy_loss_weight = energy_loss_weight
        self.foreground_weight = foreground_weight
        self.aux_x0_weight = aux_x0_weight

        self.charge_transform = charge_transform
        self.clamp_x0_for_energy = clamp_x0_for_energy

        self.sample_eta = sample_eta
        self.sample_clamp_x0 = sample_clamp_x0

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert raw charge space -> model space.
        If no transform is provided, returns x unchanged.
        """
        if self.charge_transform is None:
            return x
        return self.charge_transform.encode(x)

    def _decode_x(self, x: torch.Tensor, clamp_nonnegative: bool = True) -> torch.Tensor:
        """
        Convert model space -> raw charge space.
        If no transform is provided, returns x unchanged (optionally clamped).
        """
        if clamp_nonnegative:
            x = torch.clamp(x, min=0.0)

        if self.charge_transform is None:
            return x
        return self.charge_transform.decode(x)

    def forward(self, x_t, t, labels):
        """
        Forward pass through the denoiser.

        Args:
            x_t: noisy voxel grid, shape (B, 1, img_size, img_size, img_size)
            t: integer timesteps, shape (B,)
            labels: kinematics, shape (B, 7)

        Returns:
            predicted noise eps_hat with same shape as x_t
        """
        return self.model(x_t, t, labels)

    @torch.no_grad()
    def sample(
        self,
        labels,
        num_samples: Optional[int] = None,
        device=None,
        num_timesteps: Optional[int] = None,
        eta: Optional[float] = None,
        decode_output: bool = True,
        clamp_x0: Optional[bool] = None,
    ):
        """
        Convenience wrapper to generate samples given kinematics.

        Args:
            labels:
                Kinematics, shape (B, 7)

            num_samples:
                If provided and labels has shape (1, 7), tile labels.

            device:
                Device to use; defaults to self.device.

            num_timesteps:
                Number of DDIM-style inference steps.

            eta:
                DDIM eta. 0.0 = deterministic.

            decode_output:
                If True and a charge_transform is provided, decode samples
                back to raw charge space.

            clamp_x0:
                Whether to clamp x0 >= 0 during sampling.

        Returns:
            Tensor of shape (B, 1, img_size, img_size, img_size)
        """
        if device is None:
            device = self.device

        if eta is None:
            eta = self.sample_eta

        if clamp_x0 is None:
            clamp_x0 = self.sample_clamp_x0

        labels = labels.to(device)

        if num_samples is None:
            num_samples = labels.size(0)
        else:
            if labels.size(0) == 1 and num_samples > 1:
                labels = labels.repeat(num_samples, 1)

        x = self.model.sample(
            kin=labels,
            num_samples=num_samples,
            device=device,
            num_timesteps=num_timesteps,
            eta=eta,
            clamp_x0=clamp_x0,
        )

        if decode_output:
            x = self._decode_x(x, clamp_nonnegative=clamp_x0)

        return x

    def _shared_step(self, batch, stage: str):
        """
        Shared logic for training / validation.

        Expected batch:
            x0:     raw charge space if charge_transform is provided,
                    otherwise already model space
            labels: kinematics, shape (B, 7)
        """
        x0, labels = batch
        x0 = x0.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Model-space target
        x0_model = self._encode_x(x0)

        B = x0_model.size(0)
        T = self.model.num_timesteps

        t = torch.randint(
            low=0,
            high=T,
            size=(B,),
            device=self.device,
            dtype=torch.long,
        )

        eps = torch.randn_like(x0_model)

        # Forward diffusion
        x_t = self.model.q_sample(x0_model, t, eps)

        # Predict noise
        eps_pred = self(x_t, t, labels)

        # Foreground-aware weighting
        foreground = (x0_model > (-1.0 + 1e-6)).float()
        weights = 1.0 + self.foreground_weight * foreground
        norm = weights.sum().clamp_min(1.0)

        # Main epsilon loss
        eps_loss = (((eps_pred - eps) ** 2) * weights).sum() / norm

        # Auxiliary x0 reconstruction loss
        x0_pred_model = self.model.predict_x0_from_eps(x_t, t, eps_pred)
        x0_loss = (F.smooth_l1_loss(x0_pred_model, x0_model, reduction="none") * weights).sum() / norm

        loss = eps_loss + self.aux_x0_weight * x0_loss

        # Optional energy conservation penalty in raw charge space
        if self.energy_loss_weight > 0.0:
            x0_pred_raw = self._decode_x(
                x0_pred_model,
                clamp_nonnegative=self.clamp_x0_for_energy,
            )

            # If charge_transform is None, this assumes x0 itself is already raw charge.
            e_true = x0.view(B, -1).sum(dim=1)
            e_pred = x0_pred_raw.view(B, -1).sum(dim=1)

            energy_loss = F.mse_loss(e_pred, e_true)
            loss = loss + self.energy_loss_weight * energy_loss

            self.log(
                f"{stage}_energy_loss",
                energy_loss,
                prog_bar=False,
                on_step=(stage == "train"),
                on_epoch=True,
                sync_dist=True,
            )

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_eps_loss",
            eps_loss,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_x0_loss",
            x0_loss,
            prog_bar=False,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
        )
        return optimizer