import torch
from torch.utils.data import DataLoader
import tqdm
import pickle as pkl
import numpy as np
from datasets import TransformerConf3Dataset, CNFDataset
from models import TransformerConf3, LightningModelCNF
from utils import args_transformer, args_cnf, create_mask_src, create_mask_tgt

from model_wrappers import VertexTransformer, TrackGenerator

class GlobalModelLoader:
    def __init__(self, device: str):
        self.device = device
        self.transformer_model = None
        self.generator_model = {}
        self.test_loader = None
        self.test_dataset = None
        self.particle_types = ["proton_contained", "proton_exiting", "muon"]
        self.transformer_args = None
        self.generator_args = None

    def load_sample_data(self, sample_idx: int = 0) -> None:
        pad_value = self.test_loader.dataset.pad_value
        stats = self.test_loader.dataset.metadata

        pad_value = self.test_dataset.pad_value

        n_batches = len(self.test_loader)
        print(f"Number of batches: {n_batches}")

        t = tqdm.tqdm(enumerate(self.test_loader),total=n_batches, disable=False)
        candidates = {"ekin_pred": [], 
                      "dir_pred": [], 
                      "vtx_pred": [], 
                      "ekin_true": [], 
                      "dir_true": [], 
                      "vtx_true": [], 
                      "N_pred": [], 
                      "N_true": [], 
                      "hits": [],
                      "exit_particle": []}

        for i, batch in t:
            if i != sample_idx:
                continue

            hits, exit_particle, vtx_true, ekin_true, dir_true, keep_iter_true,_ ,_ = batch

            # Create masks for source and target sequences
            src_mask, src_padding_mask = create_mask_src(hits, exit_particle, pad_value, self.device)

            hits = hits.to(self.device)
            exit_particle = exit_particle.to(self.device)
            src_mask = src_mask.to(self.device)
            src_padding_mask = src_padding_mask.to(self.device)
        
            memory, vtx_pred = self.transformer_model.model.encode(hits, exit_particle, src_mask, src_padding_mask)
            #memory = memory.to(device)
    
            L, B, _ = ekin_true.shape
            preds = torch.zeros(L+5, B, 4).fill_(pad_value).type(torch.float).to(self.device)
            
            first_token = self.transformer_model.model.first_token.repeat(1, B, 1)
            preds[0, :, :] = first_token
        
            # keep track of predictions that finished (none before starting)
            prev_info = torch.ones(size=(B,)).bool().to(self.device)
        
            L_pred = 1
            for i in range(L+4):
                # create masks and run model
                tgt_mask, tgt_padding_mask = create_mask_tgt(preds[:i+1], pad_value, self.device)
                out_ekin, out_dir, is_next = self.transformer_model.model.decode(
                    preds[:i+1], memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        
                ekin_last = out_ekin[-1].reshape(out_ekin.shape[1], out_ekin.shape[2])
                dir_last = out_dir[-1].reshape(out_dir.shape[1], out_dir.shape[2])
                is_next_last = torch.sigmoid(is_next[-1]).squeeze() > 0.5
                
                preds[i+1, prev_info, :1] = ekin_last.detach()[prev_info, :]
                preds[i+1, prev_info, 1:] = dir_last.detach()[prev_info, :]
        
                prev_info = torch.logical_and(prev_info, is_next_last)
                L_pred += 1
                if prev_info.sum() == 0:
                    break
        
            ekin_pred = preds[1:L_pred, :, :1]
            dir_pred = preds[1:L_pred, :, 1:]
        
            output = {
                'ekin_pred': ekin_pred.detach().cpu().numpy(),
                'dir_pred': dir_pred.detach().cpu().numpy(),
                'vtx_pred': vtx_pred.detach().cpu().numpy(),
                'ekin_true': ekin_true.detach().cpu().numpy(),
                'dir_true': dir_true.detach().cpu().numpy(),
                'vtx_true': vtx_true.detach().cpu().numpy(),
                'exit_particle': exit_particle.detach().cpu().numpy(),
            }
    
            #output['ekin_true'] = (output['ekin_true'] * stats['statistics']['per_tree']['proton_contained']['true_iniekin']['std']) + stats['statistics']['per_tree']['proton_contained']['true_iniekin']['mean']  # back to original values
            #output['ekin_pred'] = (output['ekin_pred'] * stats['statistics']['per_tree']['proton_contained']['recon_ekin']['std'] + stats['statistics']['per_tree']['proton_contained']['recon_ekin']['mean'])
            #output['vtx_true'] *= (self.test_dataset.cube_size * 1.5)
            #output['vtx_pred'] *= (self.test_dataset.cube_size * 1.5)

            hits[:, :, 3] *= stats['statistics']['per_tree']['proton_contained']['recon_charge']['std']
            
    
            for event_idx in range(output['ekin_pred'].shape[1]):
                ekin_pred = output['ekin_pred'][:, event_idx]
                dir_pred = output['dir_pred'][:, event_idx]
                vtx_pred = output['vtx_pred'][event_idx]
                ekin_true = output['ekin_true'][:, event_idx]
                dir_true = output['dir_true'][:, event_idx]
                vtx_true = output['vtx_true'][event_idx]
                
                hits_true = hits[:, event_idx].cpu().numpy()
                image = np.zeros((self.transformer_args.va_size, self.transformer_args.va_size, self.transformer_args.va_size))
                idx = hits_true[:, :3].astype(int)
                val = hits_true[:, 3]
                image[idx[:, 0], idx[:, 1], idx[:, 2]] = val
                boundary_size = self.transformer_args.va_size//2 - self.generator_args.img_size//2
                image = image[boundary_size:-boundary_size, boundary_size:-boundary_size, boundary_size:-boundary_size].reshape(-1)

                exit_particle = output['exit_particle'][:, event_idx]
                # for exit_p in range(output['exit_particle'].shape[0]):
                #     # for exiting muon:
                #     if exit_particle[exit_p, 7] == 1:
                #         exit_particle[exit_p, 0] *= stats['statistics']['per_tree']['muon']['true_iniekin']['std']
                #         exit_particle[exit_p, 0] += stats['statistics']['per_tree']['muon']['true_iniekin']['mean']
                #     # for exiting proton:
                #     else:
                #         exit_particle[exit_p, 0] *= stats['statistics']['per_tree']['proton_exiting']['true_iniekin']['std']
                #         exit_particle[exit_p, 0] += stats['statistics']['per_tree']['proton_exiting']['true_iniekin']['mean']
                #     exit_particle[exit_p, 1:4] *= (self.test_dataset.cube_size * 3.5)

                
                # remove padding
                mask_pred = (dir_pred != pad_value)[:, 0]
                ekin_pred = ekin_pred[mask_pred].reshape(-1, ekin_pred.shape[1])
                dir_pred = dir_pred[mask_pred].reshape(-1, dir_pred.shape[1])
                mask_true = (dir_true != pad_value)[:, 0]
                ekin_true = ekin_true[mask_true].reshape(-1, ekin_true.shape[1])
                dir_true = dir_true[mask_true].reshape(-1, dir_true.shape[1])
                
                
                N_pred = ekin_pred.shape[0]
                N_true = ekin_true.shape[0]

                candidates['ekin_pred'].append(ekin_pred)
                candidates['dir_pred'].append(dir_pred)
                candidates['vtx_pred'].append(vtx_pred)
                candidates['ekin_true'].append(ekin_true)
                candidates['dir_true'].append(dir_true)
                candidates['vtx_true'].append(vtx_true)
                candidates['N_pred'].append(N_pred)
                candidates['N_true'].append(N_true)
                candidates['hits'].append(image)
                candidates['exit_particle'].append(exit_particle)
            break

        return candidates

    def load_models(self) -> None:
        self.transformer_model = self.load_transformer_model()
        for particle_type in self.particle_types:
            self.generator_model[particle_type] = self.load_generator_model(particle_type)

    def load_transformer_model(self) -> VertexTransformer:
        # Set up the transformer
        parser = args_transformer()
        self.transformer_args, unknown = parser.parse_known_args([
            "--metadata_path", "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl",
            "--dataset_path", "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip",
            "--checkpoint_path", "/pscratch/sd/b/botaoli/SFGD_VA/Results/checkpoints",
            "--checkpoint_name", "v3",
            "--batch_size", "16",
            "--num_workers", "32",
            "--encoder_layers", "10",
            "--decoder_layers", "10",
        ])
        print("\n- Arguments:")
        for arg, value in vars(self.transformer_args).items():
            print(f"  {arg}: {value}")

        self.test_dataset = TransformerConf3Dataset(self.transformer_args, split="test")
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.transformer_args.batch_size, num_workers=self.transformer_args.num_workers,
                collate_fn=self.test_dataset.collate_fn, pin_memory=True, persistent_workers=True, shuffle=False)

        model = TransformerConf3(
            num_encoder_layers=self.transformer_args.encoder_layers,
            num_decoder_layers=self.transformer_args.decoder_layers,
            emb_size=self.transformer_args.hidden,
            num_head=self.transformer_args.attn_heads,
            img_size=self.transformer_args.va_size,
            dropout=self.transformer_args.dropout,
            max_len=10,
        )
        checkpoint_path = "/".join((self.transformer_args.checkpoint_path, self.transformer_args.checkpoint_name, "val_loss", "last-v1.ckpt"))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {
            key.replace("model.", ""): value for key, value in checkpoint['state_dict'].items()
            if key.replace("model.", "") != "pos_encoding_tgt.pos_embedding"
        }
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device)
        print("Model weights loaded!")
        return VertexTransformer(model)

    def load_generator_model(self, particle_type: str) -> TrackGenerator:
        # Arguments
        parser = args_cnf()
        self.generator_args, unknown = parser.parse_known_args()
    
        self.generator_args.particle = particle_type
        self.generator_args.metadata_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl"
        self.generator_args.dataset_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip"
        self.generator_args.cnf_ind_path = "/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/gan_ind.pkl"
        self.generator_args.save_dir = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/"
        self.generator_args.checkpoint_path = "/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/checkpoints_v2"
        self.generator_args.checkpoint_name = self.generator_args.particle
    
        if self.generator_args.particle == "muon" or self.generator_args.particle == "proton_exiting":
            self.generator_args.label_size = 7
        elif self.generator_args.particle == "proton_contained":
            self.generator_args.label_size = 7
        self.generator_args.epochs = 50
        self.generator_args.log_every_n_steps = 2000
        self.generator_args.batch_size = 512
        self.generator_args.hidden = 256
        self.generator_args.num_workers = 64
    
        checkpoint_path = "/".join((self.generator_args.checkpoint_path, self.generator_args.particle, "val_loss","last.ckpt"))
        
        
        # load the model
        model = LightningModelCNF.load_from_checkpoint(checkpoint_path, img_shape = (self.generator_args.img_size, self.generator_args.img_size, self.generator_args.img_size), 
                                        label_size = self.generator_args.label_size, 
                                        hidden_features = self.generator_args.hidden, 
                                        num_blocks_in_MADE = self.generator_args.num_blocks_in_MADE, 
                                        num_transformers = self.generator_args.num_transformers, 
                                        lr = self.generator_args.lr, 
                                        wd = self.generator_args.weight_decay)
    
        model = model.to(self.device)
        
        #model.eval()
        metadata = pkl.load(open(self.generator_args.metadata_path, "rb"))
        return TrackGenerator(model, self.generator_args, metadata)