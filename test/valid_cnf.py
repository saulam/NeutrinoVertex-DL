import os
import sys

import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido


import numpy as np
import pandas as pd
import pickle as pkl
import sys
# sys.path.append('/opt/software/root/6.36.00/lib') 
#!pip install uproot
from ROOT import TFile, TTree, std
import ROOT
import matplotlib.pyplot as plt
import glob

module_path = os.path.abspath('..')
if module_path not in sys.path:
    sys.path.insert(0, module_path)

from torch.utils.data import DataLoader
from utils import args_transformer, args_gan, args_cnf
from datasets import CNFDataset
from models import LightningModelCNF

def calc_exit_point(
    start_point: np.ndarray,
    direction: np.ndarray,
    cube_size: float,
    va_size: int) -> np.ndarray:
    """
    Compute the exit point of a ray from a self.va_size^3 cube grid centered at the origin.
    
    Parameters
    ----------
    start_point : np.ndarray, shape (3,)
        A point inside the volume.
    direction : np.ndarray, shape (3,)
        Track direction.
    
    Returns
    -------
    exit_pt : np.ndarray, shape (3,)
        The 3D coordinates where the ray exits the box.
    """
    half_span = (cube_size * va_size) / 2.0
    mins = np.full(3, -half_span)
    maxs = np.full(3,  half_span)
    
    if np.any(start_point < mins) or np.any(start_point > maxs):
        raise ValueError(f"start_point {start_point} is outside the volume [{mins}, {maxs}].")

    # calculate candidate "times"
    t_exits = []
    for i in range(3):
        d = direction[i]
        if d > 0:
            t = (maxs[i] - start_point[i]) / d
            t_exits.append(t)
        elif d < 0:
            t = (mins[i] - start_point[i]) / d
            t_exits.append(t)
        else:
            t_exits.append(np.inf)
    
    t_exits = np.array(t_exits)
    t_pos = t_exits[t_exits > 0]
    if t_pos.size == 0:
        raise RuntimeError("Ray does not exit the box (direction may be degenerate).")
    t_exit = t_pos.min()
    
    exit_pt = start_point + t_exit * direction
    return exit_pt

def grid_to_sparse_points(grid, threshold=1e-6):
    """
    grid: numpy array (D, H, W)
    Returns arrays x, y, z, val for voxels above |value| > threshold.
    """

    mask = np.abs(grid) > threshold
    z_idx, y_idx, x_idx = mask.nonzero()  # (z, y, x)
    vals = grid[mask]
    return x_idx, y_idx, z_idx, vals

def make_cube_traces(x, y, z, v, cmin, cmax, showscale_first=True):
    """
    Build a list of Mesh3d traces, one per (x, y, z) cube.
    Opacity is proportional to v/cmax.
    Each cube has hover text with its coordinates and value.
    """

    traces = []
    first = True

    for cx, cy, cz, val in zip(x, y, z, v):
        # 8 corners of cube [cx, cx+1] x [cy, cy+1] x [cz, cz+1]
        verts = np.array([
            (cx,   cy,   cz),
            (cx+1, cy,   cz),
            (cx+1, cy+1, cz),
            (cx,   cy+1, cz),
            (cx,   cy,   cz+1),
            (cx+1, cy,   cz+1),
            (cx+1, cy+1, cz+1),
            (cx,   cy+1, cz+1),
        ])
        xs, ys, zs = verts[:, 0], verts[:, 1], verts[:, 2]

        # 12 triangles (2 per face)
        faces = [
            (0, 1, 2), (0, 2, 3),  # bottom
            (4, 5, 6), (4, 6, 7),  # top
            (0, 1, 5), (0, 5, 4),  # front
            (2, 3, 7), (2, 7, 6),  # back
            (1, 2, 6), (1, 6, 5),  # right
            (0, 3, 7), (0, 7, 4),  # left
        ]
        i = [f[0] for f in faces]
        j = [f[1] for f in faces]
        k = [f[2] for f in faces]

        intensity = [val] * 8
        opacity = 0.0 if cmax <= 0 else float(val) / cmax

        hover_text = [f"x={cx}, y={cy}, z={cz}, value={val:.4g}"] * 8

        traces.append(
            go.Mesh3d(
                x=xs,
                y=ys,
                z=zs,
                i=i,
                j=j,
                k=k,
                intensity=intensity,
                intensitymode="vertex",
                colorscale="Viridis",
                cmin=cmin,
                cmax=cmax,
                opacity=opacity,
                flatshading=True,
                text=hover_text,
                hoverinfo="text",
                showscale=showscale_first and first,
                colorbar=dict(title="Energy") if (showscale_first and first) else None,
            )
        )
        if first:
            first = False

    return traces

def draw_data_vs_generated(hit_charge, samples, output_plot_folder,n_plot = 1):

    for i in range(n_plot): 
        data = hit_charge[i]
        generated = samples[i]
        generated[generated<0.5] = 0
        global_max = float(max(data.max(), generated.max()))
        cmin, cmax = 0.01, global_max if global_max > 0 else 1.0
        x_t, y_t, z_t, v_t = grid_to_sparse_points(data)
        x_g, y_g, z_g, v_g = grid_to_sparse_points(generated)

        true_traces = make_cube_traces(x_t, y_t, z_t, v_t, cmin, cmax, showscale_first=True)
        gen_traces  = make_cube_traces(x_g, y_g, z_g, v_g, cmin, cmax, showscale_first=True)

        # Plot
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "scene"}, {"type": "scene"}]],
            subplot_titles=("True", "Generated"),
        )
        for tr in true_traces:
            fig.add_trace(tr, row=1, col=1)
        for tr in gen_traces:
            fig.add_trace(tr, row=1, col=2)
        axis_range = [-0.5, 5.5]
        for col in [1, 2]:
            fig.update_scenes(
                xaxis=dict(title="x", range=axis_range),
                yaxis=dict(title="y", range=axis_range),
                zaxis=dict(title="z", range=axis_range),
                aspectmode="cube",
                row=1,
                col=col,
            )

        fig.update_layout(
            showlegend=False,
        )

        # save the figure
        fig.write_image(f"{output_plot_folder}/data_vs_generated_{i}.png")
        #plt.close()
    return

def generate_samples(model, labels, n_set, n_sample, device, min_charge, max_charge):

    labels = labels.to(device)
    generated_charge = []
    
    for n in range(n_set):
    
        if n%10 == 0:
            print(n)
        with torch.no_grad():
            generated_voxels = model.sample(labels, num_samples=n_sample)
        #print(generated_voxels.shape)
    
        generated_p_1 = generated_voxels[0]

        generated_p_1 = (generated_p_1 + 1) / 2
        generated_p_1 *= (max_charge - min_charge)
        generated_p_1 += min_charge

        for s in range(n_sample):
            generated_p_s = generated_p_1[s]
            generated_charge.append(generated_p_s.detach().cpu().numpy())
            
    print(len(generated_charge))
    samples = np.array(generated_charge)
    samples = samples.reshape(samples.shape[0], 5, 5, 5)

    return samples

def get_data_for_sample(particle_type = "proton_contained", particle = "proton", sample_ind = 0, n_data_for_plot = 100):
    data_file = f"/pscratch/sd/b/botaoli/SFGD_VA/Data/genValid/{particle_type}/analysis_files/{particle}_{sample_ind}/output_PGun_{particle_type}.root"
    data_tree = TFile(data_file, "read")
    ana_tree = data_tree["ana"]
    n_entry = ana_tree.GetEntries()

    # dictionary for 5 x 5 x 5 grid
    hit_charge = {}
    for i in range(5):
        for j in range(5):
            for k in range(5):
                hit_charge[(i, j, k)] = []  

    print(hit_charge.keys())
    print ("number of entries: ", n_entry)
    hit_charge_array = np.zeros((n_data_for_plot, 5, 5, 5))
    for i in range(n_entry):
        ana_tree.GetEntry(i)
        x = np.array(ana_tree.recon_sfg_hitposx_rel)
        y = np.array(ana_tree.recon_sfg_hitposy_rel)
        z = np.array(ana_tree.recon_sfg_hitposz_rel)

        x = x + 2
        y = y + 2
        z = z + 2

        mask = (x < 5) & (x >= 0) & (y < 5) & (y >= 0) & (z < 5) & (z >= 0)

        x = x[mask]
        y = y[mask]
        z = z[mask]

        recon_exit_tag = ana_tree.recon_exit_tag
        true_sfg_con = ana_tree.true_sfg_con

        if (recon_exit_tag == 1 or true_sfg_con == 0):
            if particle_type == "proton_contained":
                # keep only contained protons
                continue
        else:
            if particle_type == "muon" or particle_type == "proton_exiting":
                # keep only escaping muons
                continue
        
        for ind in range(len(x)):
            pos = np.array([x[ind], y[ind], z[ind]])
            if i < n_data_for_plot:
                hit_charge_array[i, pos[0], pos[1], pos[2]] = ana_tree.recon_sfg_charge[ind]
            hit_charge[tuple(pos)].append(ana_tree.recon_sfg_charge[ind])

    return hit_charge, hit_charge_array



def get_sample_parameters(args, metadata, valid_params_folder_base, particle_type = "proton_contained", particle = "proton", sample_ind = 0):
    ############################################################################################
    # Load the parameters for the sample
    ############################################################################################
    params_file = valid_params_folder_base + f"/{particle}_{sample_ind}/{particle}_param.csv"

    param = pd.read_csv(params_file).iloc[:].values.reshape(-1)

    source_center_vec = [0.01,0.02,-192.87];

    # convert torch tensor to numpy array
    ke = float(param[6]) - metadata['statistics']['per_tree'][args.particle]['true_iniekin']['mean']
    ke /= metadata['statistics']['per_tree'][args.particle]['true_iniekin']['std']
    ini_pos = (param[0:3] - source_center_vec - args.cube_size/20) * 10/(args.cube_size * 1.5)
    ini_dir = param[3:6]

    if args.particle == "proton_exiting" or args.particle == "muon":
        exit_pos = calc_exit_point((param[0:3] - source_center_vec)*10, param[3:6], args.cube_size, args.va_size)/(args.cube_size * 3.5)
    else:
        exit_pos = None

    if exit_pos is not None:
        params = np.array([ini_pos[0], ini_pos[1], ini_pos[2], ke, ini_dir[0], ini_dir[1], ini_dir[2], exit_pos[0], exit_pos[1], exit_pos[2]])
    else:
        params = np.array([ini_pos[0], ini_pos[1], ini_pos[2], ke, ini_dir[0], ini_dir[1], ini_dir[2]])

    return params

def Validate_Diffusion(particle_type = "proton_contained", particle = "proton", sample_ind = 0, n_set = 100, n_sample = 100, n_data_for_plot = 10):
    
    # Arguments
    parser = args_cnf()
    args, unknown = parser.parse_known_args()

    args.particle = particle_type
    args.metadata_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl"
    args.dataset_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip"
    args.cnf_ind_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/gan_ind.pkl"
    args.save_dir = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/"
    args.checkpoint_path = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline/checkpoints"
    args.checkpoint_name = args.particle

    if args.particle == "muon" or args.particle == "proton_exiting":
        args.label_size = 10
    elif args.particle == "proton_contained":
        args.label_size = 7
    args.epochs = 50
    args.log_every_n_steps = 2000
    args.batch_size = 512
    args.hidden = 256
    args.num_workers = 64

    valid_params_folder_base = f"/pscratch/sd/b/botaoli/SFGD_VA/Data/genValid/{particle_type}/analysis_files"
    output_plot_folder = f"/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline/plots/{particle_type}/{particle}_{sample_ind}"
    os.makedirs(output_plot_folder, exist_ok=True)
    
    checkpoint_path = "/".join((args.checkpoint_path, args.particle, "train_loss","last.ckpt"))
    
    
    # load the model
    model = LightningModelCNF.load_from_checkpoint(checkpoint_path, img_shape = (args.img_size, args.img_size, args.img_size), 
                                    label_size = args.label_size, 
                                    hidden_features = args.hidden, 
                                    num_blocks_in_MADE = args.num_blocks_in_MADE, 
                                    num_transformers = args.num_transformers, 
                                    lr = args.lr, 
                                    wd = args.weight_decay)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ############################################################################################
    # Load the parameters for the sample
    ############################################################################################
    metadata = pkl.load(open(args.metadata_path, "rb"))
    params = get_sample_parameters(args, metadata, valid_params_folder_base, 
                                   particle_type = particle_type, 
                                   particle = particle, 
                                   sample_ind = sample_ind)

    labels = torch.tensor([params], dtype=torch.float32)

    print(labels)

    ############################################################################################
    # Load the data for the sample
    ############################################################################################
    
    hit_charge, hit_charge_array = get_data_for_sample(particle_type = particle_type, 
                                     particle = particle, 
                                     sample_ind = sample_ind,
                                     n_data_for_plot = n_data_for_plot)

    ############################################################################################
    # Generate the samples
    ############################################################################################
    
    min_charge = 0
    max_charge = metadata['statistics']['per_tree'][particle_type]['recon_charge']['max']

    samples = generate_samples(model, labels, n_set, n_sample, device, min_charge, max_charge)

    ############################################################################################
    # Plot the charge distribution
    ############################################################################################
    draw_data_vs_generated(hit_charge_array, samples, output_plot_folder, n_plot = n_data_for_plot)

    hit_charge = {key: np.array(value) for key, value in hit_charge.items()}
    
    for key, value in hit_charge.items():
        if hit_charge[key].size > 0:
            max_charge = min(500, hit_charge[key].max())
            plt.hist(hit_charge[key], bins=500, range=(0, max_charge), alpha=0.5, color="blue", density=True, label=f"data, n_hits: {len(hit_charge[key])}")
            #add the number of events in the plot as label
    
            generated_charge = samples[:,key[0],key[1],key[2]]
            generated_charge = generated_charge[generated_charge>5]
            #print(generated_charge)
            plt.hist(generated_charge, bins=500, range=(0, max_charge), alpha=0.5, color="red", density=True, label=f"generated,n_hits: {len(generated_charge)}")
            plt.legend()
            plt.savefig(f"{output_plot_folder}/charge_distribution_{key}.png")
            plt.close() 

def main():
    particle_dict = {
        #"proton_contained": "proton",
        #"muon": "muon",
        "proton_exiting": "proton"
    }
    sample_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    n_set = 20
    n_sample = 512

    for items in particle_dict.items():
        particle_type = items[0]
        particle = items[1]
        for ind in sample_ind:
            print(f"Validating {particle_type} {particle} {ind}")
            Validate_Diffusion(particle_type = particle_type, particle = particle, sample_ind = ind, n_set = n_set, n_sample = n_sample)

if __name__ == "__main__":
    main()