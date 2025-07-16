import random
import numpy as np
import torch
import pickle as pkl
from glob import glob
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils import set_random_seed


class VADataset(Dataset):
    def __init__(self, args, split='test'):
        with open(args.metadata_path, 'rb') as f:
            self.metadata = pkl.load(f)

        set_random_seed(args.seed, random=random, numpy=np)
        self.dataset_path = args.dataset_path
        self.lookup_tables = self.metadata['lookup_tables']
        self.lookup_keys = list(self.lookup_tables['proton'].keys())
        self.statistics = self.metadata['statistics']
        self.max_p = args.max_p
        self.split = split
        self.total_events = self.__len__()
        self.cube_shift = 1
        self.cube_size = args.cube_size
        self.va_size = args.va_size
        self.pad_value = args.pad_value

        # max number of protons for val and test sets
        self.p_val = np.random.randint(1, self.max_p + 1, self.total_events)
        self.p_test = np.random.randint(1, self.max_p + 1, self.total_events)

        # Shuffle all the lists (particles starting from the same position) in the dictionary
        for particle in self.lookup_tables.keys():
            for key in self.lookup_tables[particle].keys():
                random.shuffle(self.lookup_tables[particle][key])

    def __len__(self):
        return min(len(x.keys()) for x in self.lookup_tables.values())

    def __getitem__(self, idx):
        while True:
            lookup_key = self.lookup_keys[idx]
            cand_p  = self.lookup_tables['proton'][lookup_key]
            cand_mu = self.lookup_tables['muon'][lookup_key]
    
            # check your minimum requirements, otherwise pick a new random idx
            if len(cand_p) >= 20 and len(cand_mu) >= 10:
                break
    
            idx = random.randint(0, self.total_events - 1)

        cands = {'muon': [], 'proton': []}
        if self.split == "train":
            # First K candidate protons are for training: randomly select 1 to max_p random candidates
            cand_p = cand_p[:-(self.p_val[idx] + self.p_test[idx])]
            cand_p = random.sample(cand_p, random.randint(1, min(self.max_p, len(cand_p))))
            # All muons expect the last two are for training: randomly select 1 candidate
            cand_mu = cand_mu[:-2]
            cand_mu = random.sample(cand_mu, 1)
        elif self.split == "val":
            set_random_seed(idx, random=random, numpy=np)  # for reproducibility
            # Next fixed 1 to max_p candidates are for validation
            cand_p = cand_p[-(self.p_val[idx] + self.p_test[idx]):-self.p_test[idx]]
            # Next muon is for validation
            cand_mu = cand_mu[-2:-1]
        else:
            set_random_seed(idx, random=random, numpy=np)  # for reproducibility
            # Last fixed 1 to max_p candidates are for test
            cand_p = cand_p[-self.p_test[idx]:]
            # Last muon is for test
            cand_mu = cand_mu[-1:]
        
        cands['muon'].extend(cand_mu)
        cands['proton'].extend(cand_p)

        # Retrieve the particle candidates
        particles = {'exiting': [], 'contained': []}
        for particle, cand in cands.items():
            for cand_id in cand:
                paths = glob(self.dataset_path.format(particle, cand_id))
                assert len(paths) == 1
                loaded_cand = np.load(paths[0])  # load particle
                if particle == 'muon':
                    particles['exiting'].append(loaded_cand)
                else:
                    particles['contained'].append(loaded_cand)

        # Random shift (same for all the particles)
        shift = np.random.randint(-self.cube_shift, self.cube_shift + 1, 3)

        # Prepare event
        output = {
            'exiting': {
                'nb_particles': len(particles['exiting']),
                'images': np.zeros(shape=(len(particles['exiting']), self.va_size, self.va_size, self.va_size)),
                'iniekins': np.zeros(shape=(len(particles['exiting']),)),
                'inidirs': np.zeros(shape=(len(particles['exiting']), 3)),
                'inipos': np.zeros(shape=(len(particles['exiting']), 3)),
                'exitpos': np.zeros(shape=(len(particles['exiting']), 3)),
                'lens': np.zeros(shape=(len(particles['exiting']),)),
            },
             'contained': {
                'nb_particles': len(particles['contained']),
                'images': np.zeros(shape=(len(particles['contained']), self.va_size, self.va_size, self.va_size)),
                'iniekins': np.zeros(shape=(len(particles['contained']),)),
                'inidirs': np.zeros(shape=(len(particles['contained']), 3)),
                'inipos': np.zeros(shape=(len(particles['contained']), 3)),
                'lens': np.zeros(shape=(len(particles['contained']),)),
            },
            'nb_particles': len(particles['exiting']) + len(particles['contained']),
            'va_image': np.zeros(shape=(self.va_size, self.va_size, self.va_size)),
            'vertex_pos': np.zeros(shape=(3,)),
        }

        max_extend = self.va_size//2
        for key in particles.keys():
            for i, data in enumerate(particles[key]):
                hit_x = data['recon_sfg_hitposx_rel']
                hit_y = data['recon_sfg_hitposy_rel']
                hit_z = data['recon_sfg_hitposz_rel']
                hit_q = data['recon_sfg_charge']
                pos_ini_mod = data['true_inipos_mod'] - (self.cube_size / 2.0)
                pos_ini = data['true_inipos']
                pos_end = data['true_endpos']
                length = np.linalg.norm(pos_end - pos_ini)
                iniekin = data['true_iniekin']
                inidir = data['true_inidir']
                recon_exit_tag = data['recon_exit_tag']
    
                hit_x_shifted = hit_x + shift[0]
                hit_y_shifted = hit_y + shift[1]
                hit_z_shifted = hit_z + shift[2]
                pos_ini_mod_shifted = pos_ini_mod + (self.cube_size * shift)
                
                if key == 'exiting':
                    assert recon_exit_tag == 1
                    # calculate exiting point
                    output[key]['exitpos'][i] = self.calc_exit_point(pos_ini_mod_shifted, inidir)
    
                mask = (
                      (hit_x_shifted >= -max_extend) & (hit_x_shifted <= max_extend)
                    & (hit_y_shifted >= -max_extend) & (hit_y_shifted <= max_extend)
                    & (hit_z_shifted >= -max_extend) & (hit_z_shifted <= max_extend)
                )
                hit_x_idx = hit_x_shifted[mask] + max_extend
                hit_y_idx = hit_y_shifted[mask] + max_extend
                hit_z_idx = hit_z_shifted[mask] + max_extend
                hit_q_val = hit_q[mask]

                output['va_image'][hit_x_idx, hit_y_idx, hit_z_idx] += hit_q_val
                output[key]['images'][i, hit_x_idx, hit_y_idx, hit_z_idx] = hit_q_val
                output[key]['iniekins'][i] = iniekin
                output[key]['inidirs'][i] = inidir
                output[key]['inipos'][i] = pos_ini_mod_shifted
                output[key]['lens'][i] = length
                output['vertex_pos'] += (pos_ini_mod_shifted/output['nb_particles'])

        # Sort particles by kinetic energy in descendent order
        order = output['exiting']['iniekins'].argsort()[::-1]
        for key in output['exiting'].keys():
            if key != 'nb_particles':
                output['exiting'][key] = output['exiting'][key][order]
        order = output['contained']['iniekins'].argsort()[::-1]
        for key in output['contained'].keys():
            if key != 'nb_particles':
                output['contained'][key] = output['contained'][key][order]

        output['va_image']= np.column_stack((np.argwhere(output['va_image']), output['va_image'][output['va_image'] != 0]))
        output['iniekins'] = output['contained']['iniekins']
        output['inidirs'] = output['contained']['inidirs']
        output['exit_info'] = np.concatenate(
            (output['exiting']['iniekins'].reshape(-1, 1), output['exiting']['exitpos'], output['exiting']['inidirs']),
            axis=1
        )
        if self.split != 'test':
            del output['exiting'], output['contained']

        self.preprocess(output)

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


    def preprocess(self, output):
        output['va_image'][:, 3] /= self.metadata['statistics']['per_tree']['proton']['recon_charge']['std']
        output['exit_info'][:, 0] -= self.metadata['statistics']['per_tree']['muon']['true_iniekin']['mean']
        output['exit_info'][:, 0] /= self.metadata['statistics']['per_tree']['muon']['true_iniekin']['std']
        output['exit_info'][:, 1:4] /= (self.cube_size * 3.5)
        output['vertex_pos'] /= (self.cube_size * 1.5)
        output['iniekins'] -= self.metadata['statistics']['per_tree']['proton']['true_iniekin']['mean']
        output['iniekins'] /= self.metadata['statistics']['per_tree']['proton']['true_iniekin']['std']

        output['va_image'] = torch.from_numpy(output['va_image'])
        output['exit_info'] = torch.from_numpy(output['exit_info'])
        output['vertex_pos'] = torch.from_numpy(output['vertex_pos'])
        output['iniekins'] = torch.from_numpy(output['iniekins']).reshape(-1, 1)
        output['inidirs'] = torch.from_numpy(output['inidirs'])
        

    def collate_fn(self, batch):
        img_batch, exit_batch, vertex_batch, ekins_batch, dirs_batch, isnext_batch = [], [], [], [], [], []

        if self.split == "test":
            test_images = []

        for event in batch:
            va_image = event['va_image']
            exit_info = event['exit_info']
            vertex_pos = event['vertex_pos']
            iniekins = event['iniekins']
            inidirs = event['inidirs']
            nb_particles = event['nb_particles']

            # Set the transformer-decoder ending condition
            is_next = torch.ones(size=(iniekins.shape[0],))
            is_next[-1] = 0

            # Append data to respective lists
            img_batch.append(va_image)
            exit_batch.append(exit_info)
            vertex_batch.append(vertex_pos)
            ekins_batch.append(iniekins)
            dirs_batch.append(inidirs)
            isnext_batch.append(is_next)

        assert len(img_batch) > 0

        # Convert lists to torch tensors and pad sequences
        img_batch = pad_sequence(img_batch, padding_value=self.pad_value).float()
        exit_batch = pad_sequence(exit_batch, padding_value=self.pad_value).float()
        vertex_batch = torch.stack(vertex_batch).float()
        ekins_batch = pad_sequence(ekins_batch, padding_value=self.pad_value).float()
        dirs_batch = pad_sequence(dirs_batch, padding_value=self.pad_value).float()
        isnext_batch = pad_sequence(isnext_batch, padding_value=self.pad_value).float()

        return img_batch, exit_batch, vertex_batch, ekins_batch, dirs_batch, isnext_batch
