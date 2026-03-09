from typing import Dict, List, Optional
import torch
import numpy as np

from model_loader import GlobalModelLoader
from config import FitConfig
from params import FitParams
from forward import render_sum
from losses import data_loss
from fit_utils import angle_between
from prune_merge import prune_tracks, merge_close_tracks

def repel_loss(dir: torch.Tensor, weights: Optional[torch.Tensor], active: List[int], cfg: FitConfig) -> torch.Tensor:
    if cfg.gamma_repel <=0 or len(active) <= 1:
        return torch.tensor(0.0, device=dir.device, dtype=dir.dtype)
    
    dir = dir[active]
    if weights is not None:
        weights = weights[active]
    else:
        weights = torch.ones(len(active), device=dir.device, dtype=dir.dtype)

    sigma_rad = cfg.repel_sigma_rad

    loss = torch.tensor(0.0, device=dir.device, dtype=dir.dtype)
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            ang = angle_between(dir[i], dir[j])
            loss += weights[i] * weights[j] * (torch.exp(-0.5 * (ang / sigma_rad)**2))
    return cfg.gamma_repel * loss

def drift_loss(ekin: torch.Tensor, ekin_pred: torch.Tensor, active: List[int], cfg: FitConfig) -> torch.Tensor:
    if cfg.gamma_drift <=0 or len(active) <= 1:
        return torch.tensor(0.0, device=ekin.device, dtype=ekin.dtype)
    
    ekin = ekin[active]
    ekin_pred = ekin_pred[active]
    return cfg.gamma_drift * (ekin - ekin_pred).norm(dim=-1).mean()

def fit_vertex_event(
    x: torch.Tensor,
    model_loader: GlobalModelLoader,
    cfg: FitConfig,
    dataset: Dict[str, torch.Tensor],
    more_particles: bool = False,
    n_more_particles: int = 0,
) -> Dict[str, torch.Tensor]:
    
    device, dtype = cfg.device, cfg.dtype
    x = x.to(device, dtype)
    va_size = cfg.va_size
    img_size = cfg.img_size

    n_sample_for_template = cfg.n_sample_for_template

    # take this value to int
    N_pred = int(dataset['N_pred'])
    ekin_pred = dataset['ekin_pred']
    dir_pred = dataset['dir_pred']
    vtx_pred = dataset['vtx_pred']
    # For now we fix the kinematics of the exiting particle
    exit_particle = dataset['exit_particle']
    exit_muon_info = exit_particle[exit_particle[:, 7] == 1].to(device, dtype)
    exit_proton_info = exit_particle[exit_particle[:, 7] == 0].to(device, dtype)

    N_max = min(N_pred+n_more_particles, cfg.N_max) if more_particles else min(N_pred, cfg.N_max)

    # Set fitting parameters
    params = FitParams(
        n_max=N_max,
        init_dir=dir_pred,
        init_ekin=ekin_pred,
        init_vtx=vtx_pred,
        init_weights=None,
        background_mode=cfg.background_mode,
        va_size=va_size,
        device=device,
        dtype=dtype,
        more_particles=more_particles,
        E_mean=model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['mean'],
        E_std=model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['std'],
    ).to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(params.parameters(), lr=cfg.adam_lr)
    active = list(range(N_max))
    if len(exit_muon_info) > 0:
        z_muon = model_loader.generator_model["muon"].model.nflow._distribution.sample(len(exit_muon_info)*n_sample_for_template).to(device, dtype).detach()    
    else:
        z_muon = None
    if len(exit_proton_info) > 0:
        z_proton = model_loader.generator_model["proton_contained"].model.nflow._distribution.sample(len(exit_proton_info)*n_sample_for_template).to(device, dtype).detach()
    else:
        z_proton = None
    
    z_proton_contained = model_loader.generator_model["proton_contained"].model.nflow._distribution.sample(len(active)*n_sample_for_template).to(device, dtype).detach()
    for step in range(cfg.adam_steps):
        optimizer.zero_grad(set_to_none=True)

        c = params.constrained()

        if z_muon is not None:
            exit_muon = render_sum(model_loader, c, [0], img_size, exit_muon_info, z=z_muon, n_sample_for_template=n_sample_for_template)
            
        if z_proton is not None:
            exit_proton = render_sum(model_loader, c, [0], img_size, exit_proton_info, z=z_proton, n_sample_for_template=n_sample_for_template)

        x_recon = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template)
        

        idx = torch.tensor([active[i] for i in active], device=x_recon.device)
        # x_recon is a tensor of shape (len(active), img_size*img_size*img_size)
        # no need to reshape x_recon, it is already in the correct shape
        # idx is to find the corresponding weights for each track in x_recon
        if c['weights'] is not None:
            weights = c['weights'][idx]
        else:
            weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
        print("weights shape: ", weights.shape)
        x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
        if z_muon is not None:
            x_sum += exit_muon[0]
        if z_proton is not None:
            x_sum += exit_proton[0]

        print("x_sum shape: ", x_sum.shape)
        print("x shape: ", x.shape)

        d_loss = data_loss(x, x_sum, cfg.data_term)
        
        print("dir: ", c['dir'])
        print("ekin: ", c['E'])
        print("vtx: ", c['vtx'])

        print("true_dir: ", dataset['dir_true'])
        print("true_ekin: ", dataset['ekin_true'])
        print("true_vtx: ", dataset['vtx_true'])
        #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
        rep_loss = repel_loss(c['dir'], c['weights'], active, cfg)

        drf_loss_E = drift_loss(dataset['ekin_pred'], c['E'], active, cfg)

        loss = d_loss + rep_loss + drf_loss_E
        loss.backward()
        for name, p in params.named_parameters():
            if p.grad is None:
                print(name, "NOT CONNECTED (grad=None)")
            elif torch.all(p.grad == 0):
                print(name, "connected but ZERO gradient")
            else:
                print(name, "connected (nonzero grad)")
        optimizer.step()

        if (step+1) % cfg.prune_every == 0 and c['weights'] is not None:
            active = prune_tracks(params, cfg)
            active = merge_close_tracks(params, active,cfg)
            if len(active) == 0:
                active = [int(torch.argmax(c.constrained()['weights'].item()))]
    
    if c['weights'] is not None:
        active = prune_tracks(params, cfg)
        active = merge_close_tracks(params, active,cfg)
    if len(active) == 0:
        active = [int(torch.argmax(c.constrained()['weights'].item()))]
    
    # do lbfgs optimization
    if cfg.use_lbfgs:
        lbfgs_optimizer = torch.optim.LBFGS(params.parameters(), max_iter=cfg.lbfgs_steps, line_search_fn='strong_wolfe')
        
        def closure():
            lbfgs_optimizer.zero_grad(set_to_none=True)
            c = params.constrained()
            x_recon = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template)
            idx = torch.tensor([active[i] for i in active], device=x_recon.device)
            if c['weights'] is not None:
                weights = c['weights'][idx]
            else:
                weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
            x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
            if z_muon is not None:
                x_sum += exit_muon[0]
            if z_proton is not None:
                x_sum += exit_proton[0]
            d_loss = data_loss(x, x_sum, cfg.data_term)
            #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
            #rep = repel_loss(c['dir'], c['weights'], active, cfg)
            loss = d_loss
            loss.backward()
            return loss

        lbfgs_optimizer.step(closure)

        if c['weights'] is not None:
            active = prune_tracks(params, cfg)
            active = merge_close_tracks(params, active,cfg)
        if len(active) == 0:
            active = [int(torch.argmax(c.constrained()['weights'].item()))]

    c = params.constrained()
    x_recon_final = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template)
    idx = torch.tensor([active[i] for i in active], device=x_recon_final.device)
    if c['weights'] is not None:
        weights = c['weights'][idx]
    else:
        weights = torch.ones(x_recon_final.shape[0], device=x_recon_final.device, dtype=x_recon_final.dtype)
    x_recon_final = (x_recon_final * weights.view(-1, 1)).sum(dim=0)
    if z_muon is not None:
        x_recon_final += exit_muon[0]
    if z_proton is not None:
        x_recon_final += exit_proton[0]

    ekin_pred_final = (c['E'] * model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['recon_ekin']['std'] + model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['recon_ekin']['mean'])
    vtx_pred_final = (c['vtx'] * model_loader.test_dataset.cube_size * 1.5)

    output = {
        'x_recon_final': x_recon_final,
        'active': active,
        'dir_pred': c['dir'],
        'ekin_pred': ekin_pred_final,
        'vtx_pred': vtx_pred_final,
        'weights_pred': weights,
        'background_pred': c['b'],
        'x_true': x,
        'dir_true': dataset['dir_true'],
        'ekin_true': dataset['ekin_true'],
        'vtx_true': dataset['vtx_true'],
    }
    
    print("output: ", output)
        
    return None