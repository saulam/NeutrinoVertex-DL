"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch Lightning model for the first configuration of the decomposing transformer.
"""

import torch
import pytorch_lightning as pl
from utils import create_mask, CustomLambdaLR, CombinedScheduler


# Define the PyTorch Lightning model
class LightningModelTransformerConf1(pl.LightningModule):
    def __init__(self, model, loss_fn1, loss_fn2, loss_fn3, loss_fn4, args):
        """
        Initialise the Lightning Model for the first configuration of the decomposing transformer.

        Args:
            model: The transformer model.
            loss_fn1: Loss function 1 (vertex position).
            loss_fn2: Loss function 2 (ekins).
            loss_fn3: Loss function 3 (dirs).
            loss_fn4: Loss function 3 (keep iterating).
            pad_value: Padding value for tokens.
            warmup_steps: Number of warmup steps for learning rate.
            cosine_annealing_steps: Number of steps for cosine annealing of learning rate.
            lr: Initial learning rate.
            beta1: Beta1 hyperparameter for the optimiser.
            beta2: Beta2 hyperparameter for the optimiser.
            weight_decay: Weight decay (L2 regularization) applied to the model parameters.
            lr_decay: Learning rate decay after each cycle of the cosine annealing scheduler.
        """
        super().__init__()

        self.model = model
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.loss_fn3 = loss_fn3
        self.loss_fn4 = loss_fn4
        self.pad_value = args.pad_value
        self.warmup_steps = args.warmup_steps
        self.start_cosine_step = args.start_cosine_step
        self.cosine_annealing_steps = args.scheduler_steps
        self.lr = args.lr
        self.betas = (args.beta1, args.beta2)
        self.weight_decay = args.weight_decay
        self.eps = args.eps

    def on_train_start(self):
        """
        Callback method called when the training starts.
        This method fixes the parameter groups for training.

        Note: This method is used to address an issue related to parameter groups in PyTorch Lightning.
        """
        self.optimizers().param_groups = self.optimizers()._optimizer.param_groups

    def common_step(self, batch):
        """
        Common processing steps for training and validation.

        Args:
            batch: Batch of data.

        Returns:
            loss1: Loss for vertex position.
            loss2: Loss for ekin.
            loss2: Loss for dir.
            loss3: Loss for keep iterating.
            loss: Total loss.
            batch_size: Batch size.
            lr: Current learning rate.
        """
        # Unpack values from the batch
        hits, exit_muon, vtx_true, ekin_true, dir_true, keep_iter_true = batch

        # Slice 'ekin_true' and 'dir_true' to exclude the last item of each sequence
        ekin_true_input = ekin_true[:-1, :]
        dir_true_input = dir_true[:-1, :]

        # Create masks for source and target sequences
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(hits, ekin_true_input, self.pad_value,
                                                                             self.device)

        # Pass the inputs and masks to the model to get predictions
        vtx_pred, ekin_pred, dir_pred, keep_iter_pred = self.model(hits, exit_muon, ekin_true_input, dir_true_input, src_mask, tgt_mask,
                                                                   src_padding_mask, tgt_padding_mask)

        # Mask out all the padding tokens
        padding_mask = (ekin_true[:, :, 0] != self.pad_value)
        ekin_true = ekin_true[padding_mask]
        ekin_pred = ekin_pred[padding_mask]
        dir_true = dir_true[padding_mask]
        dir_pred = dir_pred[padding_mask]
        keep_iter_true = keep_iter_true[padding_mask]
        keep_iter_pred = keep_iter_pred[padding_mask].squeeze()

        # Calculate loss
        loss1 = self.loss_fn1(vtx_pred, vtx_true)
        loss2 = self.loss_fn2(ekin_pred, ekin_true)
        loss3 = self.loss_fn3(dir_pred, dir_true)
        loss4 = self.loss_fn4(keep_iter_pred, keep_iter_true)
        loss = loss1 + loss2 + loss3 + loss4

        # Retrieve current learning rate
        lr = self.optimizers().param_groups[0]['lr']

        return loss1, loss2, loss3, loss4, loss, hits.shape[1], lr

    def training_step(self, batch, batch_idx):
        """
        Training step for a batch of data.

        Args:
            batch: Batch of training data.
            batch_idx: Index of the batch.

        Returns:
            loss: Training loss for the batch.
        """
        loss1, loss2, loss3, loss4, loss, batch_size, lr = self.common_step(batch)

        self.log("train_loss1", loss1.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("train_loss2", loss2.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("train_loss3", loss3.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("train_loss4", loss4.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("train_loss", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)
        self.log("lr", lr, batch_size=batch_size, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for a batch of data.

        Args:
            batch: Batch of validation data.
            batch_idx: Index of the batch.

        Returns:
            loss: Validation loss for the batch.
        """
        loss1, loss2, loss3, loss4, loss, batch_size, lr = self.common_step(batch)

        self.log("val_loss1", loss1.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("val_loss2", loss2.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("val_loss3", loss3.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("val_loss4", loss4.item(), batch_size=batch_size, prog_bar=False, sync_dist=True)
        self.log("val_loss", loss.item(), batch_size=batch_size, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """
        Configure and initialise the optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and learning rate scheduler.
        """
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if (
                (p.ndim == 1)
                or name.endswith(".bias")
            ):
                no_decay.append(p)
            else:
                decay.append(p)
                
        optimizer = torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
        )

        if self.warmup_steps==0 and self.cosine_annealing_steps==0:
            return optimizer

        if self.warmup_steps == 0:
            warmup_scheduler = None
        else:
            # Warm-up scheduler
            warmup_scheduler = CustomLambdaLR(optimizer, self.warmup_steps)
 
        if self.cosine_annealing_steps == 0:
            cosine_scheduler = None
        else:
            # Cosine annealing scheduler
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.cosine_annealing_steps,
                eta_min=0.,
            )

        # Combine both schedulers
        combined_scheduler = CombinedScheduler(
            optimizer=optimizer,
            scheduler1=warmup_scheduler,
            scheduler2=cosine_scheduler,
            warmup_steps=self.warmup_steps,
            start_cosine_step=self.start_cosine_step,
            lr_decay=1.0
        )

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': combined_scheduler, 'interval': 'step'}}


    def lr_scheduler_step(self, scheduler, *args):
        """
        Perform a learning rate scheduler step.

        Args:
            scheduler: Learning rate scheduler.
        """
        scheduler.step()
