import torch

from typing import Dict

from model_wrappers import TrackGenerator, VertexTransformer
from fit import fit_vertex_event

class FlowGenerator(TrackGenerator):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def render(self,
        labels: torch.Tensor
    ) -> torch.Tensor:
        return self.flow.sample(labels, num_samples=1)

class TransformerDecoder(VertexTransformer):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def predict(self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.predict(x)


class VertexFitter:
    def __init__(self, model_loader, cfg):
        self.model_loader = model_loader
        self.cfg = cfg


    def fit(self,
        x: torch.Tensor,batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        dataset = self.model_loader.load_sample_data(batch_idx)
        # keep only the first event
        dataset = {k: torch.tensor(v[0], device=self.cfg.device, dtype=self.cfg.dtype) for k, v in dataset.items()}
        return fit_vertex_event(x, self.model_loader, self.cfg, dataset)

