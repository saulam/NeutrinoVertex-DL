"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Training script for the proton GAN considering the first transformer configuration.
"""

import sys, os
sys.path.append(os.path.abspath("..")) 
import json
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import GANDataset
from models import Generator, Critic, WGAN_GP_Loss, LightningModelGAN
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
    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(torch.cuda.device_count()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')
    # Arguments
    parser = args_gan()
    args, unknown = parser.parse_known_args()

    args.particle = "proton_contained"
    args.metadata_path = "/scratch/libota/sfgd_va_nn_data/NN_Data/metadata.pkl"
    args.dataset_path = "/scratch/libota/sfgd_va_nn_data/NN_Data/{}/{}/{}.npz"
    args.gan_ind_path = "/scratch/libota/sfgd_va_nn_data/NN_Data/gan_ind.pkl"
    args.save_dir = "/scratch2/libota/SFGD_Vertex_Activity/Results/gan/"
    args.checkpoint_path = "/scratch2/libota/SFGD_Vertex_Activity/Results/gan/checkpoints"
    args.checkpoint_name = "proton_contained"

    args.epochs = 50
    args.log_every_n_steps = 2000
    args.batch_size = 3072
    args.hidden = 64
    args.warmup_steps = 10
    args.num_workers = 64

    
    # Training set and loader
    train_set = GANDataset(args, split="train")
    train_loader = DataLoader(train_set, collate_fn=train_set.collate_fn, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True)


    # Geneator and critic models
    generator = Generator(input_size=args.input_size, label_size=args.label_size, noise_size=args.noise_size,
                          hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    critic = Critic(input_size=args.input_size, label_size=args.label_size, noise_size=args.noise_size,
                        hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads, dropout=args.dropout)
    
    generator._init_weights()
    critic._init_weights()
    gen_total_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    cri_total_params = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(generator)
    print(critic)
    print("Total trainable params: {} (generator), {} (discriminator).".format(gen_total_params, cri_total_params))

    # Loss function (for critic)
    adv_loss = WGAN_GP_Loss(args.lambda_gp)

    # Create lightning model
    lightning_model = LightningModelGAN(generator=generator,
                                        critic=critic,
                                        noise_size=args.noise_size,
                                        crit_repeats=args.crit_repeats,
                                        adversarial_loss=adv_loss, lr=args.lr, wd=args.weight_decay)

    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
    #checkpoint_callback = ModelCheckpoint(dirpath=config["save_path"], every_n_train_steps=5000)

    callbacks = []
    monitored_losses = ['val_loss',]

    for loss_name in monitored_losses:
        checkpoint = ModelCheckpoint(
            dirpath=f"{args.checkpoint_path}/{args.checkpoint_name}/{loss_name}",
            save_top_k=args.save_top_k,
            monitor=loss_name,
            mode="min",
            save_last=True
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
        precision="bf16",
        devices=torch.cuda.device_count(),
        logger=logger,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        log_every_n_steps=100,
        deterministic=True,
    )

    # Run the training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
    )


if __name__ == "__main__":
    main()
