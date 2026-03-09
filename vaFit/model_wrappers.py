import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import argparse
import numpy as np
from typing import Dict
from models import TransformerConf3, LightningModelCNF

from fit_utils import normalize_parameters
from nflows.utils import torchutils


class TrackGenerator:
    def __init__(self, model: LightningModelCNF, args: argparse.Namespace, metadata: dict):
        self.model = model
        self.args = args
        self.metadata = metadata
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def render(self,
        labels: torch.Tensor,
        z: torch.Tensor = None,
        n_sample_for_template: int = 128,
    ) -> torch.Tensor:
        """
        Render a particle image from the given labels and noise.
        """
        # print(labels.shape)
        # print(labels)
        #parameters = normalize_parameters(labels[0], self.args, self.metadata, self.args.particle)
        parameters = labels
        max_charge = np.log(self.metadata['statistics']['per_tree'][self.args.particle]['recon_charge']['max'] + 1)
        min_charge = np.log(1)
        print("TrackGenerator parameters shape: ", parameters.shape)
        print("TrackGenerator parameters: ", parameters)
        # if self.args.particle == "muon" or self.args.particle == "proton_exiting":
        #     parameters = normalize_parameters(parameters, self.args, self.metadata, self.args.particle)
        #     print("TrackGenerator parameters after normalization: ", parameters)
        parameters = parameters.to(self.model.device)
        generated_charge = []
        
        sample_param = torchutils.repeat_rows(parameters, num_reps=n_sample_for_template)
        def inverse_only(z,ctx):
            out, _ = self.model.nflow._transform.inverse(z, context=ctx)
            return out
        generated_voxels, _ = self.model.nflow._transform.inverse(z, context=sample_param)
        #generated_voxels = checkpoint(inverse_only, z, sample_param, use_reentrant=False)
        print("generated_voxels shape: ", generated_voxels.shape)
        generated_voxels = (generated_voxels + 1) / 2
        generated_voxels *= (max_charge - min_charge)
        generated_voxels += min_charge

        generated_voxels = torch.exp(generated_voxels) - 1
        
        print("generated_voxels: ", generated_voxels)
        print("generated_voxels min: ", generated_voxels.min())
        print("generated_voxels max: ", generated_voxels.max())
        print("generated_voxels mean: ", generated_voxels.mean())
        # average over the sample dimension
        generated_voxels = torchutils.split_leading_dim(generated_voxels, shape=[-1, n_sample_for_template])
        generated_voxels = generated_voxels.mean(dim=1)

        print("TrackGenerator generated_voxels shape: ", generated_voxels.shape)

        return generated_voxels

class VertexTransformer:
    def __init__(self, model: TransformerConf3):
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        
    def predict(self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the labels from the given image.
        """
        raise NotImplementedError("Subclass must implement this method")
