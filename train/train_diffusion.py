"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Training script for the proton diffusion model considering the first transformer configuration.
"""

import sys, os
sys.path.append(os.path.abspath("..")) 
import json
import torch
import pytorch_lightning as pl
# check lightning version
print(pl.__version__)

from torch.utils.data import DataLoader
from datasets import GenDataset
from models import ConditionalDiffusionModel, LightningModelDiffusion
from utils import args_gan
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping, Callback 

torch.set_float32_matmul_precision("medium")
pl_major = int(pl.__version__.split(".")[0])

class CustomProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.ascii = True  # Ensure ASCII characters are used
        
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()

        bar.ascii = True  # Ensure ASCII characters are used for validation
    
        return bar

def main():
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Arguments
    parser = args_gan()
    args, unknown = parser.parse_known_args()

    print(args)

    if args.particle == "proton_exiting" or args.particle == "muon":
        args.label_size = 10
    else:
        args.label_size = 7

    nb_gpus = len(args.gpus)
    gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])

    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


    # Training set and loader
    train_set = GenDataset(args, split="train")
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True, persistent_workers=True, 
                              pin_memory=True, prefetch_factor=2)

    # Validation set and loader
    val_set = GenDataset(args, split="test")
    val_loader = DataLoader(
        val_set,
        collate_fn=val_set.collate_fn,
        batch_size=args.batch_size,
        num_workers=min(4, args.num_workers),  # Fewer workers for validation
        shuffle=False,
        persistent_workers=True if args.num_workers > 0 else False,
        pin_memory=True
    )

    print(f"Training samples: {len(train_set)}")
    print(f"Validation samples: {len(val_set)}")


    # Diffusion model
    model = ConditionalDiffusionModel(
        time_dim=128, 
        cond_dim=128, 
        base_channels=32, 
        num_timesteps=1000,
        img_size=args.img_size
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print("Total trainable params: {}.".format(total_params))


    # Create lightning model
    lightning_model = LightningModelDiffusion(
        diffusion_model=model,
        lr=args.lr, 
        wd=args.weight_decay,
        energy_loss_weight=0.0,
    )

    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs/"+args.particle, name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs/"+args.particle, name=args.name)

    callbacks = []
    monitored_losses = ['train_loss', 'val_loss']

    for loss_name in monitored_losses:
        checkpoint = ModelCheckpoint(
            dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{loss_name}",
            every_n_epochs=1,
            save_last=True,
            save_top_k=3,
            monitor=loss_name,
            mode="min",
            save_on_train_epoch_end=True if loss_name == 'train_loss' else False,
        )
        callbacks.append(checkpoint)

    progress_bar = CustomProgressBar()
    callbacks.append(progress_bar)

    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    # Create trainer module
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        precision="bf16-mixed",
        devices=nb_gpus,
        num_nodes=args.num_nodes,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=False,
    )

    # Run the training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.load_checkpoint if args.load_checkpoint else None,
    )


if __name__ == "__main__":
    main()