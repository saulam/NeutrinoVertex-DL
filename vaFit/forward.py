from typing import Dict, List, Optional
import torch

from model_loader import GlobalModelLoader

def render_sum(
    model_loader: GlobalModelLoader,
    c: Dict[str, torch.Tensor],
    active: List[int],
    img_size: int,
    exit_particle: Optional[torch.Tensor] = None,
    n_exit_protons: Optional[int] = None,
    z: Optional[torch.Tensor] = None,
    repeats: Optional[torch.Tensor] = None,
    n_sample_for_template: int = 128,
) -> torch.Tensor:
    if exit_particle is None:
        # render the contained particles
        ini_pos = c['vtx']
        if repeats is not None:
            ini_pos = torch.repeat_interleave(ini_pos, repeats=repeats, dim=0)                      # (3,)
        #idx = torch.tensor([active[i] for i in active], device=ini_pos.device)  # (B,)

        E = c['E'].view(-1, 1)            # (B,1)  (adjust if your E shape differs)
        d = c['dir'].view(-1, 3)          # (B,3)

        labels = torch.cat([ini_pos, E, d], dim=-1)  # (B,7)
        pred_x = model_loader.generator_model["proton_contained"].render(labels, z=z, n_sample_for_template=n_sample_for_template)
    else:
        print(exit_particle[0,7])
        print("exit_particle: ", exit_particle.shape)
        # render the exiting particle
        ini_pos = c['vtx']
        E = exit_particle[:, 0:1]
        d = exit_particle[:, 4:7]
    
        particle_type = "proton_exiting"
        if exit_particle[0, 7] == 1:
            particle_type = "muon"
        output = []
        
        if particle_type == "muon":
            label = torch.cat([ini_pos, E, d], dim=-1)
            output.append(model_loader.generator_model[particle_type].render(label, z=z, n_sample_for_template=n_sample_for_template))
        elif particle_type == "proton_exiting":
            # get vtx[i] where n_exit_protons[i] > 0
            ini_pos = c['vtx'][n_exit_protons > 0]
            label = torch.cat([ini_pos, E, d], dim=-1)
            output.append(model_loader.generator_model[particle_type].render(label, z=z, n_sample_for_template=n_sample_for_template))
        pred_x = torch.cat(output, dim=0)
        # for i in range(exit_particle.shape[0]):
        #     particle_ind = int(exit_particle[i, 7])
        #     E = exit_particle[i, 0:1]
        #     d = exit_particle[i, 4:7]
        #     label = torch.cat([ini_pos[i], E, d], dim=-1)
        #     if len(label.shape) == 1:
        #         label = label.unsqueeze(0)
        #     if particle_ind == 1:
        #         particle_type = "muon"
        #         output_mu.append(model_loader.generator_model[particle_type].render(label, z=z, n_sample_for_template=n_sample_for_template))
        #         print("output_mu shape: ", output_mu[-1].shape)
        #     elif particle_ind == 0:
        #         particle_type = "proton_exiting"
        #         output_proton.append(model_loader.generator_model[particle_type].render(label, z=z, n_sample_for_template=n_sample_for_template))
        #     else:
        #         raise ValueError(f"Invalid particle index: {particle_ind}")
        # pred_x = torch.cat(output_mu + output_proton, dim=0)
    return pred_x
