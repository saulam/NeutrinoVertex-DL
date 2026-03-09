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
    print(hit_charge.shape)
    print(samples.shape)
    for i in range(n_plot): 
        data = hit_charge[i,:].reshape(5,5,5)
        generated = samples[i,:].reshape(5,5,5)
        generated[generated<0.5] = 0
        global_max = float(max(data.max(), generated.max()))
        cmin, cmax = 0.01, global_max if global_max > 0 else 1.0
        x_t, y_t, z_t, v_t = grid_to_sparse_points(data.detach().cpu().numpy())
        x_g, y_g, z_g, v_g = grid_to_sparse_points(generated.detach().cpu().numpy())

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


def Validate_Diffusion(particle_type = "proton_contained", particle = "proton", batch_size = 512, n_data_for_plot = 10):
    
    # Arguments
    parser = args_cnf()
    args, unknown = parser.parse_known_args()

    args.particle = particle_type
    args.metadata_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl"
    args.dataset_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip"
    args.cnf_ind_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/gan_ind.pkl"
    args.save_dir = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/"
    args.checkpoint_path = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/checkpoints_v2"
    args.checkpoint_name = args.particle

    if args.particle == "muon" or args.particle == "proton_exiting":
        args.label_size = 7
    elif args.particle == "proton_contained":
        args.label_size = 7
    args.epochs = 50
    args.log_every_n_steps = 2000
    args.batch_size = batch_size
    args.hidden = 256
    args.num_workers = 64

    output_plot_folder = f"/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/plots/{particle_type}/on_train_sample/"
    os.makedirs(output_plot_folder, exist_ok=True)
    
    checkpoint_path = "/".join((args.checkpoint_path, args.particle, "val_loss","last.ckpt"))
    
    
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
    train_set = CNFDataset(args, split="train")
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)
    
    ############################################################################################
    # Generate the samples
    ############################################################################################
    min_charge = 0
    max_charge = metadata['statistics']['per_tree'][particle_type]['recon_charge']['max']
    max_charge = np.log(max_charge + 1)
    generated_charge = []
    real_charge = []
    n_batch = len(train_loader)
    i_batch = 0
    for batch in train_loader:
        if i_batch % 10 == 0:	
            print(f"Generated {i_batch} batches")
        if (i_batch+1)%200 == 0:
            break
        images, labels, _ = batch
        labels = labels.to(device)
        with torch.no_grad():
            generated_voxels = model.sample(labels)
        # print(generated_voxels.shape)
        # print(images.shape)
        generated_voxels = (generated_voxels + 1) / 2
        generated_voxels *= (max_charge - min_charge)
        generated_voxels += min_charge
        generated_voxels = generated_voxels.detach().cpu().numpy()
        generated_voxels = np.exp(generated_voxels) - 1
        generated_voxels[generated_voxels<1.5] = 0

        images = (images + 1) / 2
        images *= (max_charge - min_charge)
        images += min_charge
        images = images.detach().cpu().numpy()
        images = np.exp(images) - 1
        images[images<0.5] = 0

        i_batch += 1
        real_charge.append(images)
        generated_charge.append(generated_voxels.reshape(-1, 125))

        print("images.shape =", images.shape)
        print("generated_voxels.shape =", generated_voxels.shape)
        if i_batch == 0:
            draw_data_vs_generated(images, generated_voxels.reshape(-1, 125), output_plot_folder, n_plot = n_data_for_plot)
    
    real_charge = np.concatenate(real_charge, axis=0)
    generated_charge = np.concatenate(generated_charge, axis=0)
    print("real_charge.shape =", real_charge.shape)
    print("generated_charge.shape =", generated_charge.shape)

    ############################################################################################
    # Plot the charge distribution
    ############################################################################################
    #draw_data_vs_generated(hit_charge_array, samples, output_plot_folder, n_plot = n_data_for_plot)

    # hit_charge = {key: np.array(value) for key, value in hit_charge.items()}
    # shape of real charge: (n_batch, batch_size, 125)
    # shape of generated charge: (n_batch, batch_size, 1,125)
    for i in range(real_charge.shape[1]):
        x = i //25
        y = (i%25) //5
        z = i%5
        max_charge_i = min(500, real_charge[:,i].max())

        count_g, bins_g, _ = plt.hist(generated_charge[:,i], bins=200, range=(2, max_charge_i), alpha=0.5, color="red", density=True, label=f"generated,n_hits: {len(generated_charge[:,i])}")
        count_r, bins_r, _ = plt.hist(real_charge[:,i], bins=200, range=(2, max_charge_i), alpha=0.5, color="blue", density=True, label=f"data, n_hits: {len(real_charge[:,i])}")
        
        plt.legend()
        plt.ylim(0, max(count_r.max(), count_g.max())*1.1)
        print(count_r.max(), count_g.max())
        plt.xlabel("Charge")
        plt.ylabel("Number of events")
        plt.savefig(f"{output_plot_folder}/charge_distribution_voxel_{x}_{y}_{z}.png")
        plt.close()


def main():
    particle_dict = {
        "proton_contained": "proton",
        #"muon": "muon",
        "proton_exiting": "proton"
    }
    batch_size = 512

    for items in particle_dict.items():
        particle_type = items[0]
        particle = items[1]
        print(f"Validating {particle_type} {particle}")
        Validate_Diffusion(particle_type = particle_type, particle = particle, batch_size = batch_size)

if __name__ == "__main__":
    main()