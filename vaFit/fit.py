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
    dataset: Dict[str, List[np.ndarray]],
    more_particles: bool = False,
    n_more_particles: int = 0,
) -> Dict[str, torch.Tensor]:
    
    n_events = len(dataset['hits'])
    device, dtype = cfg.device, cfg.dtype
    x_raw = x.to(device, dtype)
    va_size = cfg.va_size
    img_size = cfg.img_size

    n_sample_for_template = cfg.n_sample_for_template
    n_fit_per_event = cfg.n_fit_per_event


    # Raw input
    N_pred_raw = torch.tensor([int(N_pred) for N_pred in dataset['N_pred']], device=device, dtype=torch.int32)
    
    
    pred_index_raw = [0] + torch.cumsum(N_pred_raw, dim=0).tolist()

    ekin_pred_raw = torch.cat([torch.tensor(chunk, device=device, dtype=dtype) for chunk in dataset['ekin_pred']], dim=0)
    dir_pred_raw = torch.cat([torch.tensor(chunk, device=device, dtype=dtype) for chunk in dataset['dir_pred']], dim=0)    
    vtx_pred_raw = torch.cat([torch.tensor(chunk, device=device, dtype=dtype).unsqueeze(0) for chunk in dataset['vtx_pred']], dim=0)

    # For now we fix the kinematics of the exiting particle
    n_exit_protons = np.array([len(chunk)-1 for chunk in dataset['exit_particle']])
    exit_particle = torch.cat([torch.tensor(chunk, device=device, dtype=dtype) for chunk in dataset['exit_particle']], dim=0)
    exit_muon_info = exit_particle[exit_particle[:, 7] == 1]
    exit_proton_info = exit_particle[exit_particle[:, 7] == 0]


    # repeat for multiple fits
    fit_repeats = torch.full_like(N_pred_raw, n_fit_per_event)
    x = torch.repeat_interleave(x_raw, fit_repeats, dim=0)
    N_pred = torch.repeat_interleave(N_pred_raw, fit_repeats)
    pred_index = [0] + torch.cumsum(N_pred, dim=0).tolist()
    # repeats is the number of times to repeat vtx for each predicted particle
    repeats = torch.as_tensor(N_pred, device=device, dtype=torch.int32)

    ekin_pred = torch.cat([ekin_pred_raw[pred_index_raw[i]:pred_index_raw[i+1]].repeat(fit_repeats[i], 1) for i in range(len(N_pred_raw))], dim=0)
    dir_pred = torch.cat([dir_pred_raw[pred_index_raw[i]:pred_index_raw[i+1]].repeat(fit_repeats[i], 1) for i in range(len(N_pred_raw))], dim=0)
    vtx_pred = torch.cat([vtx_pred_raw[i].repeat(fit_repeats[i], 1) for i in range(len(N_pred_raw))], dim=0)

    n_exit_protons = torch.cat([torch.tensor(n_exit_protons[i], device=device, dtype=torch.int32).repeat(fit_repeats[i]) for i in range(len(N_pred_raw))], dim=0)
    exit_muon_info = torch.cat([exit_muon_info[i].repeat(fit_repeats[i], 1) for i in range(len(N_pred_raw))], dim=0)
    exit_proton_info = torch.cat([exit_proton_info[i].repeat(fit_repeats[i], 1) for i in range(len(N_pred_raw)) if n_exit_protons[i] > 0], dim=0)


    #N_max = min(max(N_pred)+n_more_particles, cfg.N_max) if more_particles else min(max(N_pred), cfg.N_max)
    N_max = N_pred.sum()

    loss_history = []

    
    
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
    # if len(exit_muon_info) > 0:
    #     z_muon = model_loader.generator_model["muon"].model.nflow._distribution.sample(len(exit_muon_info)*n_sample_for_template).to(device, dtype).detach()    
    # else:
    #     z_muon = None
    # if len(exit_proton_info) > 0:
    #     z_proton = model_loader.generator_model["proton_exiting"].model.nflow._distribution.sample(len(exit_proton_info)*n_sample_for_template).to(device, dtype).detach()
    # else:
    #     z_proton = None
    
    # z_proton_contained = model_loader.generator_model["proton_contained"].model.nflow._distribution.sample(len(active)*n_sample_for_template).to(device, dtype).detach()
    if len(exit_muon_info) > 0:
        z_muon = model_loader.generator_model["muon"].model.nflow._distribution.sample(len(exit_muon_info)*n_sample_for_template).to(device, dtype).detach()    
    else:
        z_muon = None
    if len(exit_proton_info) > 0:
        z_proton = model_loader.generator_model["proton_exiting"].model.nflow._distribution.sample(len(exit_proton_info)*n_sample_for_template).to(device, dtype).detach()
    else:
        z_proton = None
    
    z_proton_contained = model_loader.generator_model["proton_contained"].model.nflow._distribution.sample(len(active)*n_sample_for_template).to(device, dtype).detach()
        
    for step in range(cfg.adam_steps):
        optimizer.zero_grad(set_to_none=True)

        c = params.constrained()

        if z_muon is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

            exit_muon = render_sum(model_loader, c, [0], img_size, exit_muon_info, z=z_muon, n_sample_for_template=n_sample_for_template)
            print("exit_muon shape: ", exit_muon.shape)
            torch.cuda.synchronize()
            print(torch.cuda.max_memory_allocated() / 1024**3, "GB peak for this call")
        if z_proton is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            exit_proton = render_sum(model_loader, c, [0], img_size, exit_proton_info, z=z_proton, n_sample_for_template=n_sample_for_template, n_exit_protons=n_exit_protons)
            torch.cuda.synchronize()
            print(torch.cuda.max_memory_allocated() / 1024**3, "GB peak for this call")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        x_recon = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template, repeats = repeats)
        torch.cuda.synchronize()
        print(torch.cuda.max_memory_allocated() / 1024**3, "GB peak for this call")

        idx = torch.tensor([active[i] for i in active], device=x_recon.device)
        # x_recon is a tensor of shape (len(active), img_size*img_size*img_size)
        # no need to reshape x_recon, it is already in the correct shape
        # idx is to find the corresponding weights for each track in x_recon
        if c['weights'] is not None:
            weights = c['weights'][idx]
        else:
            weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
        print("weights shape: ", weights.shape)
        #x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
        x_sum = torch.cat([x_recon[pred_index[i]:pred_index[i+1]].sum(dim=0).unsqueeze(0) for i in range(len(pred_index)-1)], dim=0)
        if z_muon is not None:
            x_sum = x_sum + exit_muon
        if z_proton is not None:
            exit_proton_ind = 0
            for i in range(len(N_pred)):
                if n_exit_protons[i] > 0:
                    x_sum[i] = x_sum[i] + exit_proton[exit_proton_ind]
                    exit_proton_ind += 1

        print("x_sum shape: ", x_sum.shape)
        print("x shape: ", x.shape)

        d_loss = data_loss(x, x_sum, cfg.data_term)
        print("d_loss: ", d_loss)
        print("dir: ", c['dir'])
        print("ekin: ", c['E'])
        print("vtx: ", c['vtx'])

        print("true_dir: ", dataset['dir_true'])
        print("true_ekin: ", dataset['ekin_true'])
        print("true_vtx: ", dataset['vtx_true'])
        #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
        #rep_loss = repel_loss(c['dir'], c['weights'], active, cfg)

        #drf_loss_E = drift_loss(dataset['ekin_pred'], c['E'], active, cfg)

        loss_history.append(d_loss.item())
        loss = d_loss
        loss.backward()
        for name, p in params.named_parameters():
            if p.grad is None:
                print(name, "NOT CONNECTED (grad=None)")
            elif torch.all(p.grad == 0):
                print(name, "connected but ZERO gradient")
            else:
                print(name, "connected (nonzero grad)")
        optimizer.step()

        # if (step+1) % cfg.prune_every == 0 and c['weights'] is not None:
        #     active = prune_tracks(params, cfg)
        #     active = merge_close_tracks(params, active,cfg)
        #     if len(active) == 0:
        #         active = [int(torch.argmax(c.constrained()['weights'].item()))]
    optimizer.zero_grad(set_to_none=True)
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
            if z_muon is not None:
                exit_muon = render_sum(model_loader, c, [0], img_size, exit_muon_info, z=z_muon, n_sample_for_template=n_sample_for_template)
            if z_proton is not None:
                exit_proton = render_sum(model_loader, c, [0], img_size, exit_proton_info, z=z_proton, n_sample_for_template=n_sample_for_template, n_exit_protons=n_exit_protons)
                
            x_recon = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template, repeats = repeats)
            idx = torch.tensor([active[i] for i in active], device=x_recon.device)
            if c['weights'] is not None:
                weights = c['weights'][idx]
            else:
                weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
            #x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
            x_sum = torch.cat([x_recon[pred_index[i]:pred_index[i+1]].sum(dim=0).unsqueeze(0) for i in range(len(pred_index)-1)], dim=0)
            if z_muon is not None:
                x_sum = x_sum + exit_muon
            if z_proton is not None:
                exit_proton_ind = 0
                for i in range(len(N_pred)):
                    if n_exit_protons[i] > 0:
                        x_sum[i] = x_sum[i] + exit_proton[exit_proton_ind]
                        exit_proton_ind += 1
            d_loss = data_loss(x, x_sum, cfg.data_term)
            #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
            #rep = repel_loss(c['dir'], c['weights'], active, cfg)
            loss = d_loss
            loss.backward()
            return loss

        lbfgs_optimizer.step(closure)

        # if c['weights'] is not None:
        #     active = prune_tracks(params, cfg)
        #     active = merge_close_tracks(params, active,cfg)
        if len(active) == 0:
            active = [int(torch.argmax(c.constrained()['weights'].item()))]

    c = params.constrained()
    x_recon_final = render_sum(model_loader, c, active, img_size, z=z_proton_contained, n_sample_for_template=n_sample_for_template, repeats = repeats)
    idx = torch.tensor([active[i] for i in active], device=x_recon_final.device)
    if c['weights'] is not None:
        weights = c['weights'][idx]
    else:
        weights = torch.ones(x_recon_final.shape[0], device=x_recon_final.device, dtype=x_recon_final.dtype)
    #x_recon_final = (x_recon_final * weights.view(-1, 1)).sum(dim=0)
    x_recon_final = torch.cat([x_recon_final[pred_index[i]:pred_index[i+1]].sum(dim=0).unsqueeze(0) for i in range(len(pred_index)-1)], dim=0)
    if z_muon is not None:
        x_recon_final = x_recon_final + exit_muon
    if z_proton is not None:
        exit_proton_ind = 0
        for i in range(len(N_pred)):
            if n_exit_protons[i] > 0:
                x_recon_final[i] = x_recon_final[i] + exit_proton[exit_proton_ind]
                exit_proton_ind += 1

    ekin_pred_final = (c['E'] * model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['std'] + model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['mean'])
    vtx_pred_final = (c['vtx'] * model_loader.test_dataset.cube_size * 1.5)

    ekin_true = [(dataset['ekin_true'][i] * model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['std'] + model_loader.generator_model["proton_contained"].metadata['statistics']['per_tree']['proton_contained']['true_iniekin']['mean']) for i in range(len(dataset['ekin_true']))]
    vtx_true = [(dataset['vtx_true'][i] * model_loader.test_dataset.cube_size * 1.5) for i in range(len(dataset['vtx_true']))]
    dir_true = dataset['dir_true']

    output = {
        'x_recon_final': x_recon_final,
        'active': active,
        'dir_pred': c['dir'],
        'ekin_pred': ekin_pred_final,
        'vtx_pred': vtx_pred_final,
        'weights_pred': weights,
        'background_pred': c['b'],
        'x_true': x_raw,
        'dir_true': dir_true,
        'ekin_true': ekin_true,
        'vtx_true': vtx_true,
        'loss_history': loss_history,
    }
    
    print("output: ", output)
        
    return output

def fit_single_track(
    x: torch.Tensor,
    model_loader: GlobalModelLoader,
    cfg: FitConfig,
    dataset: Dict[str, List[np.ndarray]],
    particle: str,
) -> Dict[str, torch.Tensor]:
    
    device, dtype = cfg.device, cfg.dtype
    x = x.cpu().numpy()

    #print("x: ", x)
    # renormalize the x_raw to original range
    max_charge = np.log(model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['recon_charge']['max'] + 1)
    min_charge = np.log(1)
    #print("x before renormalization: ", x)
    x = (x + 1) / 2
    x = x * (max_charge - min_charge) + min_charge
    x = np.exp(x) - 1
    x = torch.from_numpy(x).to(device, dtype)
    
    n_tracks = x.shape[0]
    va_size = cfg.va_size
    img_size = cfg.img_size

    n_sample_for_template = cfg.n_sample_for_template
    n_fit_per_event = cfg.n_fit_per_event

    true_parameters = dataset['parameters']
    true_parameters = torch.tensor(true_parameters, device=device, dtype=dtype)

    loss_history = []

    # random initialize the parameters, uniform distribution between -1 and 1
    # dir_pred = torch.rand(n_tracks, 3, device=device, dtype=dtype) * 2 - 1
    # dir_pred /= (dir_pred.norm(dim=-1, keepdim=True) + 1e-6)
    # ekin_pred = torch.rand(n_tracks, 1, device=device, dtype=dtype) * 2 - 1
    # vtx_pred = torch.rand(n_tracks, 3, device=device, dtype=dtype) * 2 - 1

    dir_smear = 0
    ekin_smear = 0.2
    vtx_smear = 0

    
    # dir_smear_factor_comp = 1 + (torch.rand(n_tracks*n_fit_per_event, 3, device=device, dtype=dtype) * 2 - 1) * dir_smear
    # ekin_smear_factor_comp = 1 + (torch.rand(n_tracks*n_fit_per_event, 1, device=device, dtype=dtype) * 2 - 1) * ekin_smear
    # vtx_smear_factor_comp = 1 + (torch.rand(n_tracks*n_fit_per_event, 3, device=device, dtype=dtype) * 2 - 1) * vtx_smear

    vtx_pred = true_parameters[:, :3]
    ekin_pred = true_parameters[:, 3].unsqueeze(1)
    dir_pred = true_parameters[:, 4:7]

    vtx_pred = vtx_pred.repeat(n_fit_per_event, 1)
    ekin_pred = ekin_pred.repeat(n_fit_per_event, 1)
    dir_pred = dir_pred.repeat(n_fit_per_event, 1)

    vtx_pred = vtx_pred * sample_bimodal_smear(vtx_pred.shape, vtx_smear, device, dtype)
    ekin_pred = ekin_pred * sample_bimodal_smear(ekin_pred.shape, ekin_smear, device, dtype)
    
    dir_pred = dir_pred * sample_bimodal_smear(dir_pred.shape, dir_smear, device, dtype)
    dir_pred = dir_pred / (dir_pred.norm(dim=-1, keepdim=True) + 1e-6)

    x = x.repeat(n_fit_per_event, 1)

    # Set fitting parameters
    params = FitParams(
        n_max=n_tracks*n_fit_per_event,
        init_dir=dir_pred,
        init_ekin=ekin_pred,
        init_vtx=vtx_pred,
        init_weights=None,
        background_mode=cfg.background_mode,
        va_size=va_size,
        device=device,
        dtype=dtype,
        more_particles=False,
        E_mean=model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['mean'],
        E_std=model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['std'],
    ).to(device)

    # Set optimizer
    optimizer = torch.optim.Adam(params.parameters(), lr=cfg.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.adam_steps, eta_min=0.0)
    active = list(range(n_tracks))



    z = model_loader.generator_model[particle].model.nflow._distribution.sample(len(active)*n_sample_for_template*n_fit_per_event).to(device, dtype).detach()
    
    for step in range(cfg.adam_steps):
        optimizer.zero_grad(set_to_none=True)

        c = params.constrained()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        x_recon = render_sum(model_loader, c, active, img_size, z=z, n_sample_for_template=n_sample_for_template)
        torch.cuda.synchronize()
        print(torch.cuda.max_memory_allocated() / 1024**3, "GB peak for this call")

        idx = torch.tensor([active[i] for i in active], device=x_recon.device)
        # x_recon is a tensor of shape (len(active), img_size*img_size*img_size)
        # no need to reshape x_recon, it is already in the correct shape
        # idx is to find the corresponding weights for each track in x_recon
        if c['weights'] is not None:
            weights = c['weights'][idx]
        else:
            weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
        print("weights shape: ", weights.shape)
        #x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
        print("x_recon shape: ", x_recon.shape)
        x_sum = x_recon

        print("calculating data loss:")
        d_loss = data_loss(x, x_sum, cfg.data_term)
        print("d_loss: ", d_loss)
        print("dir: ", c['dir'])
        print("ekin: ", c['E'])
        print("vtx: ", c['vtx'])

        print("true_vtx: ", dataset['parameters'][:, :3])
        print("true_ekin: ", dataset['parameters'][:, 3])
        print("true_dir: ", dataset['parameters'][:, 4:7])

        # print("x: ", x)
        # print("x_sum: ", x_sum)
        #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
        #rep_loss = repel_loss(c['dir'], c['weights'], active, cfg)

        #drf_loss_E = drift_loss(dataset['ekin_pred'], c['E'], active, cfg)

        loss_history.append(d_loss.item())
        loss = d_loss
        loss.backward()
        # for name, p in params.named_parameters():
        #     if p.grad is None:
        #         print(name, "NOT CONNECTED (grad=None)")
        #     elif torch.all(p.grad == 0):
        #         print(name, "connected but ZERO gradient")
        #     else:
        #         print(name, "connected (nonzero grad)")
        optimizer.step()
        scheduler.step()
        # if (step+1) % cfg.prune_every == 0 and c['weights'] is not None:
        #     active = prune_tracks(params, cfg)
        #     active = merge_close_tracks(params, active,cfg)
        #     if len(active) == 0:
        #         active = [int(torch.argmax(c.constrained()['weights'].item()))]
    optimizer.zero_grad(set_to_none=True)
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
            x_recon = render_sum(model_loader, c, active, img_size, z=z, n_sample_for_template=n_sample_for_template)
            idx = torch.tensor([active[i] for i in active], device=x_recon.device)
            if c['weights'] is not None:
                weights = c['weights'][idx]
            else:
                weights = torch.ones(x_recon.shape[0], device=x_recon.device, dtype=x_recon.dtype)
            #x_sum = (x_recon * weights.view(-1, 1)).sum(dim=0)
            x_sum = x_recon.sum(dim=0)
            d_loss = data_loss(x, x_sum, cfg.data_term)
            #sparsity_loss = cfg.lam_a * c['weights'][active].sum() if len(active) > 0 else 0.0
            #rep = repel_loss(c['dir'], c['weights'], active, cfg)
            loss = d_loss
            loss.backward()
            return loss

        lbfgs_optimizer.step(closure)

        # if c['weights'] is not None:
        #     active = prune_tracks(params, cfg)
        #     active = merge_close_tracks(params, active,cfg)
        if len(active) == 0:
            active = [int(torch.argmax(c.constrained()['weights'].item()))]

    c = params.constrained()
    x_recon_final = render_sum(model_loader, c, active, img_size, z=z, n_sample_for_template=n_sample_for_template)
    idx = torch.tensor([active[i] for i in active], device=x_recon_final.device)
    if c['weights'] is not None:
        weights = c['weights'][idx]
    else:
        weights = torch.ones(x_recon_final.shape[0], device=x_recon_final.device, dtype=x_recon_final.dtype)
    #x_recon_final = (x_recon_final * weights.view(-1, 1)).sum(dim=0)
    x_recon_final = x_recon_final.sum(dim=0)

    ekin_pred_final = (c['E'] * model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['std'] + model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['mean'])
    vtx_pred_final = (c['vtx'] * model_loader.generator_args.va_size * 1.5)

    ekin_true = [(dataset['parameters'][i, 3] * model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['std'] + model_loader.generator_model[particle].metadata['statistics']['per_tree'][particle]['true_iniekin']['mean']) for i in range(len(dataset['parameters']))]
    vtx_true = [(dataset['parameters'][i, :3] * model_loader.generator_args.va_size * 1.5) for i in range(len(dataset['parameters']))]
    dir_true = dataset['parameters'][:, 4:7]

    output = {
        'x_recon_final': x_recon_final,
        'active': active,
        'dir_pred': c['dir'],
        'ekin_pred': ekin_pred_final,
        'vtx_pred': vtx_pred_final,
        'dir_initial': dir_pred,
        'ekin_initial': ekin_pred,
        'vtx_initial': vtx_pred,
        'weights_pred': weights,
        'background_pred': c['b'],
        'x_true': x.cpu().numpy(),
        'dir_true': dir_true,
        'ekin_true': ekin_true,
        'vtx_true': vtx_true,
        'loss_history': loss_history,
    }
    
    print("output: ", output)
        
    return output

def sample_bimodal_smear(shape, smear, device, dtype):
    # Randomly choose +smear or -smear mean (50/50)
    choose_plus = (torch.rand(shape, device=device) < 0.5).to(dtype)
    mean = (1 - smear) + choose_plus * (2 * smear)   # either 1-smear or 1+smear
    std = 0.1 * smear
    return mean + std * torch.randn(shape, device=device, dtype=dtype)
