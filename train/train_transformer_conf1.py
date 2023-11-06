"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: Training script for the first decomposing transformer configuration.
"""

import os
import json
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import TransformerConf1Dataset
from models import TransformerConf1, LightningModelTransformerConf1
from utils import args_transformer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Arguments
    parser = args_transformer(1)
    args = parser.parse_args()

    # Configuration file
    with open(args.config_path) as config_file:
        config = json.load(config_file)

    # Training and validation sets
    train_set = TransformerConf1Dataset(config=config, split="train")
    val_set = TransformerConf1Dataset(config=config, split="val")

    # Training and validation loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              collate_fn=train_set.collate_fn, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=val_set.collate_fn, shuffle=False)

    # Initialise model
    model = TransformerConf1(num_encoder_layers=args.encoder_layers,
                             num_decoder_layers=args.decoder_layers,
                             emb_size=args.hidden,
                             num_head=args.attn_heads,
                             img_size=config["img_size"],
                             tgt_size=config["target_size"],
                             dropout=args.dropout,
                             max_len=config["max_p"],
                             device=device,
                             )
    model.apply(model._init_weights)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(total_params))

    # Loss functions
    loss_fn1 = torch.nn.MSELoss()  # vertex position
    loss_fn2 = torch.nn.MSELoss()  # kinematic parameters
    loss_fn3 = torch.nn.CrossEntropyLoss()  # keep iterating

    # Calculate arguments for scheduler
    warmup_steps = len(train_loader) * args.warmup_steps // args.accum_grad_batches
    cosine_annealing_steps = len(train_loader) * args.scheduler_steps // args.accum_grad_batches

    # Create lightning model
    lightning_model = LightningModelTransformerConf1(model=model,
                                                     lr=args.lr,
                                                     beta1=args.beta1,
                                                     beta2=args.beta2,
                                                     weight_decay=args.weight_decay,
                                                     eps=args.eps,
                                                     lr_decay=args.lr_decay,
                                                     loss_fn1=loss_fn1,
                                                     loss_fn2=loss_fn2,
                                                     loss_fn3=loss_fn3,
                                                     pad_value=train_set.pad_value,
                                                     warmup_steps=warmup_steps,
                                                     cosine_annealing_steps=cosine_annealing_steps,
                                                     )

    # Define logger and checkpoint
    logger = CSVLogger(save_dir="logs/", name=config["log_path"])
    checkpoint_callback = ModelCheckpoint(dirpath=config["save_path"],
                                          save_top_k=3, monitor="val_loss")

    # Create trainer module
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accelerator="gpu",
        precision="bf16",
        devices=[0],
        logger=logger,
        log_every_n_steps=10,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
    )

    # Run the training
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
