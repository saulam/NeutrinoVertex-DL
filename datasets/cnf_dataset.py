"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: PyTorch dataset that for the generative-adversarial network (GAN).
"""

import numpy as np
import pickle as pk
import torch
from torch.utils.data import Dataset
import random
import bisect
from glob import glob
from zipfile import ZipFile
import io


class CNFDataset(Dataset):

    def __init__(self, args, split: str = "train"):
        """
        Dataset initialiser.

        Args:
            split (str): String indicating the purpose of the dataset ("train", "val", "test").

        Returns:
            None
        """
        with open(args.metadata_path, "rb") as fd:
            self.metadata = pk.load(fd)
        with open(args.cnf_ind_path, "rb") as fd:
            self.cnf_ind = pk.load(fd)

        self.particle = args.particle
        # lookup_tables: {subvoxel: number of events}
        self.lookup_tables = self.cnf_ind[self.particle]
        # lookup_keys: list of subvoxels
        self.lookup_keys = list(self.lookup_tables.keys())

        self.split = split
        self.dataset_path = args.dataset_path
        self.va_size = args.va_size
        self.img_size = args.img_size
        self.cube_size = args.cube_size  # cube (voxel) size in mm (one side)
        self.pad_value = args.pad_value



        # for key in self.lookup_tables.keys():
        #     random.shuffle(self.lookup_tables[key])

        self.lens = [self.lookup_tables[k] for k in self.lookup_keys]
        
        # Build cumulative end indices: [len(A), len(A)+len(B), ...]
        self.cums = []

        self.val_events = []
        total = 0
        for L in self.lens:
            total += L
            self.cums.append(total)
            # save the last event of each sub voxel as the test event
            self.val_events.append(total-1)
        
        self.total_events = self.__len__() # total number of different particles



    def __len__(self):
        if self.split == "train":
            return self.cums[-1] if self.cums else 0
        elif self.split == "val":
            return len(self.val_events)



    def collate_fn(self, batch):
        """
        Collates a batch of data into tensors.

        Args:
            batch (list): A list of particles.

        Returns:
            tuple: A tuple containing two tensors - image batch and parameters batch.
        """
        img_batch = np.array([event['image'] for event in batch if event['image'] is not None])
        if self.particle == "muon" or self.particle == "proton_exiting":
            params_batch = np.array([np.concatenate([event['pos_ini'], event['pos_exit'],
                                                     event['ke'], event['dir_ini']])
                                     for event in batch if event['pos_ini'] is not None])
        else:
            params_batch = np.array([np.concatenate([event['pos_ini'], event['ke'], event['dir_ini']])
                                     for event in batch if event['pos_ini'] is not None])
        img_batch = torch.tensor(img_batch).float()
        params_batch = torch.tensor(params_batch).float()

        return img_batch, params_batch

    def __getitem__(self, idx):
        """
        Retrieves a data sample at the given index.

        Args:
            idx (int): Index of the particle to retrieve.

        Returns:
            dict: A dictionary containing information about the particle image and kinematics.
        """
        # Support negative indices like a normal sequence
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        if self.split == "train":
            while idx in self.val_events:
                # pick a random event from the test set
                idx = random.randint(0, len(self.val_events) - 1)
        elif self.split == "val":
            idx = self.val_events[idx]
        # Find which sub-list this global index falls into
        j = bisect.bisect_right(self.cums, idx)  # 0..len(keys)-1
        prev_end = self.cums[j-1] if j > 0 else 0
        local_idx = idx - prev_end
        data_file = self.dataset_path.format(self.particle, 
                    self.lookup_keys[j][0],
                    self.lookup_keys[j][1],
                    self.lookup_keys[j][2])
        #print("path: ", self.dataset_path.format(self.particle, self.lookup_keys[j], local_idx))
        with ZipFile(data_file, "r") as zip_file:
            with zip_file.open(zip_file.namelist()[local_idx]) as f:
                    buf = io.BytesIO(f.read())
                    loaded_cand = np.load(buf)

        max_extend = self.va_size//2

        hit_x = loaded_cand['recon_sfg_hitposx_rel']
        hit_y = loaded_cand['recon_sfg_hitposy_rel']
        hit_z = loaded_cand['recon_sfg_hitposz_rel']
        hit_q = loaded_cand['recon_sfg_charge']
        pos_ini_mod = loaded_cand['true_inipos_mod'] - (self.cube_size / 2.0)
        pos_ini = loaded_cand['true_inipos']
        pos_end = loaded_cand['true_endpos']
        length = np.linalg.norm(pos_end - pos_ini)
        iniekin = loaded_cand['true_iniekin']
        inidir = loaded_cand['true_inidir']
        recon_exit_tag = loaded_cand['recon_exit_tag']

        # print("hit_x: ", hit_x	)
        # print("hit_y: ", hit_y)
        # print("hit_z: ", hit_z)
        # print("hit_q: ", hit_q)
        # print("pos_ini_mod: ", pos_ini_mod)
        # print("pos_ini: ", pos_ini)
        # print("pos_end: ", pos_end)

        mask = (
                  (hit_x >= -max_extend) & (hit_x <= max_extend)
                & (hit_y >= -max_extend) & (hit_y <= max_extend)
                & (hit_z >= -max_extend) & (hit_z <= max_extend)
                )

        hit_x_ind = hit_x[mask] + max_extend
        hit_y_ind = hit_y[mask] + max_extend
        hit_z_ind = hit_z[mask] + max_extend
        hit_q_val = hit_q[mask]

        if hit_x.shape[0] == 0:
            del loaded_cand
            return {'image': None,
                    'pos_ini': None,
                    'ke': None,
                    'dir': None}

        # Prepare particle
        output = {
            'image': np.zeros(shape=(self.img_size, self.img_size, self.img_size)),
            'pos_ini': np.zeros(shape=(3,)),
            'ke': np.zeros(shape=(1,)),
            'dir_ini': np.zeros(shape=(3,))
        }
        if self.particle == "muon" or self.particle == "proton_exiting":
            output['pos_exit'] = np.zeros(shape=(3,))

        # Reconstruct the image to a (self.va_size-2)x(self.va_size-2)x(self.va_size-2) flat volume
        #dense_image = np.zeros(shape=(self.va_size, self.va_size, self.va_size))
        dense_image = np.random.rand(self.va_size, self.va_size, self.va_size)
        dense_image[hit_x_ind[:], hit_y_ind[:], hit_z_ind[:]] = hit_q_val[:]
        boundary_size = self.va_size//2 - self.img_size//2
        output['image'] = dense_image[boundary_size:-boundary_size, boundary_size:-boundary_size, boundary_size:-boundary_size].reshape(-1)
        output['pos_ini'] = pos_ini_mod
        output['ke'] = np.array([iniekin])
        output['dir_ini'] = inidir
        if self.particle == "muon" or self.particle == "proton_exiting":
            output['pos_exit'] = self.calc_exit_point(pos_ini_mod, inidir)

        self.preprocess(self.particle, output)

        del loaded_cand

        return output

    def calc_exit_point(
        self,
        start_point: np.ndarray,
        direction: np.ndarray) -> np.ndarray:
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
        half_span = (self.cube_size * self.va_size) / 2.0
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


    def preprocess(self, particle, output):
        #output['image'] -= self.metadata['statistics']['per_tree'][particle]['recon_charge']['mean']
        #output['image'] /= self.metadata['statistics']['per_tree'][particle]['recon_charge']['std']
        # Normalize the image to -1,1 using min-max scaling
        #min_charge = self.metadata['statistics']['per_tree'][particle]['recon_charge']['min']
        min_charge = 0
        max_charge = self.metadata['statistics']['per_tree'][particle]['recon_charge']['max']

        output['image'][output['image'] > max_charge] = max_charge
        output['image'] = (output['image'] - min_charge) / (max_charge - min_charge)
        output['image'] = 2 * output['image'] - 1

        output['ke'] -= self.metadata['statistics']['per_tree'][particle]['true_iniekin']['mean']
        output['ke'] /= self.metadata['statistics']['per_tree'][particle]['true_iniekin']['std']
        output['pos_ini'] /= (self.cube_size * 1.5)
        if particle == "muon" or particle == "proton_exiting":
            output['pos_exit'] /= (self.cube_size * 3.5)
        
        output['image'] = torch.from_numpy(output['image'])
        output['pos_ini'] = torch.from_numpy(output['pos_ini'])
        output['ke'] = torch.from_numpy(output['ke'])
        output['dir_ini'] = torch.from_numpy(output['dir_ini'])
        if particle == "muon" or particle == "proton_exiting":
            output['pos_exit'] = torch.from_numpy(output['pos_exit'])
