import torch
import numpy as np
from typing import Dict, List

from model_wrappers import TrackGenerator, VertexTransformer
from fit import fit_vertex_event, fit_single_track

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
        dataset: Dict[str, List[np.ndarray]],n_events: int,
        fit_type: str = "event", particle: str = "proton_contained"
    ) -> Dict[str, torch.Tensor]:
        x = torch.tensor(dataset['hits'][:n_events], device=self.cfg.device, dtype=self.cfg.dtype)
        # keep only the first event
        dataset = {k: v[:n_events] for k, v in dataset.items()}

        if fit_type == "event":
            return fit_vertex_event(x, self.model_loader, self.cfg, dataset)
        elif fit_type == "single_track":
            return fit_single_track(x, self.model_loader, self.cfg, dataset, particle)

