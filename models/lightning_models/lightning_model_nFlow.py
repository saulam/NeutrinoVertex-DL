"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch Lightning model for the Conditional Normalizing Flow.
"""

import torch
import pytorch_lightning as pl
from nflows.flows.base import Flow
from nflows import transforms, distributions


class LightningModelCNF(pl.LightningModule):
    def __init__(
        self,
        img_shape = (5, 5, 5),
        label_size = 7,
        hidden_features = 128,
        num_blocks_in_MADE = 2,
        num_transformers = 8,

        lr: float = 0.00005,
        wd: float = 0.,
    ):
        super().__init__()
        self.img_shape = img_shape
        self.data_dim = img_shape[0] * img_shape[1] * img_shape[2]
        self.label_size = label_size
        self.hidden_features = hidden_features
        self.num_transformers = num_transformers   
        self.num_blocks_in_MADE = num_blocks_in_MADE
        self.lr = lr
        self.wd = wd

        self.save_hyperparameters()
        
    
        """
        Build the flow.
        """
        transform_list = []
        for _ in range(self.num_transformers):
            transform_list.append(
                #transforms.MaskedAffineAutoregressiveTransform(
                #    features=self.data_dim,
                #    hidden_features=self.hidden_features,
                #    context_features=self.label_size,
                #    num_blocks=self.num_blocks_in_MADE,
                #    use_residual_blocks=False,
                #    activation=torch.nn.ReLU(),
                #    dropout_probability=0.0,
                #    use_batch_norm=False,
                #)
                 transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                     features=self.data_dim,
                     hidden_features=self.hidden_features,
                     context_features=self.label_size,
                     num_bins=10,
                     tails="linear",
                     tail_bound=1.0,
                     num_blocks=self.num_blocks_in_MADE,
                     use_residual_blocks=True,
                     random_mask=False,
                     activation=torch.nn.ReLU(),
                     dropout_probability=0.0,
                     use_batch_norm=False,
                     )
                )
            transform_list.append(
                transforms.BatchNorm(features=self.data_dim),
            )
            transform_list.append(
                transforms.RandomPermutation(features=self.data_dim),
            )

        

        transform = transforms.CompositeTransform(transform_list)
        distribution = distributions.StandardNormal(shape = [self.data_dim])

        self.nflow = Flow(
            transform=transform,
            distribution=distribution,
        )

    def forward(self, x, y):
        return self.nflow.log_prob(x, context=y)
    
    def training_step(self, batch, batch_idx):
        x, labels = batch
        log_px = self.nflow.log_prob(inputs=x, context=labels)
        loss = -log_px.mean()
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        log_px = self.nflow.log_prob(inputs=x, context=labels)
        loss = -log_px.mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer

    @torch.no_grad()
    def sample(self, labels):
        """
        labels: (B, label_size)
        returns: (B, data_dim)
        """
        B = labels.size(0)
        samples = self.nflow.sample(num_samples=B, context=labels)
        return samples
