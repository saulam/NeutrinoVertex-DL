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
from datasets import TransformerConf3Dataset
from models import TransformerConf3, LightningModelTransformerConf3
from utils import args_transformer, SphericalAngularLoss
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
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


class PrintEpochMetrics(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        ep = int(trainer.current_epoch)
        tl = float(m.get("train_loss", float("nan")))
        vl = float(m.get("val_loss", float("nan")))
        lr = float(m.get("lr", pl_module.optimizers().param_groups[0]['lr']))
        print(f"Epoch {ep:03d} | train_loss={tl:.4f} | val_loss={vl:.4f} | lr={lr:.3g}")

def main():
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = args_transformer(1)
    args, unknown = parser.parse_known_args()
    print("\n- Arguments:")
    #for arg, value in vars(args).items():
    #    print(f"  {arg}: {value}")
    nb_gpus = len(args.gpus)
    gpus = ', '.join(args.gpus) if nb_gpus > 1 else str(args.gpus[0])


    # Manually specify the GPUs to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus

    # Training and validation sets
    train_set = TransformerConf3Dataset(args, split="train")
    print("train_set length: ", len(train_set))
    val_set = TransformerConf3Dataset(args, split="val")
    print("val_set length: ", len(val_set))
    # Training and validation loaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                              collate_fn=train_set.collate_fn, pin_memory=True, persistent_workers=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=val_set.collate_fn, pin_memory=True, persistent_workers=True, shuffle=False)

    # Initialise model
    model = TransformerConf3(num_encoder_layers=args.encoder_layers,
                             num_decoder_layers=args.decoder_layers,
                             emb_size=args.hidden,
                             num_head=args.attn_heads,
                             img_size=args.va_size,
                             dropout=args.dropout,
                             max_len=5,
                             )
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable params: {}".format(total_params))

    # Loss functions
    loss_fn1 = torch.nn.MSELoss()  # vertex position
    loss_fn2 = torch.nn.MSELoss()  # ekin
    loss_fn3 = SphericalAngularLoss()  # dirs
    loss_fn4 = torch.nn.BCEWithLogitsLoss()  # keep iterating


    # Calculate arguments for scheduler
    nb_batches = len(train_loader)
    denom = args.accum_grad_batches * nb_gpus
    print(nb_batches)
    print(denom)


    #args.lr = args.lr * (args.batch_size * denom) / 256.
    args.scheduler_steps = nb_batches * args.cosine_annealing_steps // denom
    args.warmup_steps = nb_batches * args.warmup_steps // denom
    args.start_cosine_step = (nb_batches * args.epochs // denom) - args.scheduler_steps
    print(f"lr                = {args.lr}")
    print(f"scheduler_steps   = {args.scheduler_steps}")
    print(f"warmup_steps      = {args.warmup_steps}")
    print(f"start_cosine_step = {args.start_cosine_step}")
    print(f"eff. batch size   = {args.batch_size * denom}")


    # Define logger and checkpoint
    logger = CSVLogger(save_dir=args.save_dir + "/logs", name=args.name)
    tb_logger = TensorBoardLogger(save_dir=args.save_dir + "/tb_logs", name=args.name)
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
    if args.early_stop_patience > 0:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            patience=args.early_stop_patience,
            verbose=True,
            mode='min' 
        )
        callbacks.append(early_stop_callback)



    # Create lightning model
    lightning_model = LightningModelTransformerConf3(model=model,
                                                     loss_fn1=loss_fn1,
                                                     loss_fn2=loss_fn2,
                                                     loss_fn3=loss_fn3,
                                                     loss_fn4=loss_fn4,
                                                     args=args,
                                                     )
    callbacks.append(PrintEpochMetrics())


    # Log the hyperparameters
    logger.log_hyperparams(vars(args))
    tb_logger.log_hyperparams(vars(args))

    # Create trainer module
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        accelerator="gpu",
        precision="bf16-mixed" if pl_major >= 2 else 32,
        devices=nb_gpus,
        num_nodes=args.num_nodes,
        strategy="ddp" if nb_gpus > 1 else "auto",
        logger=[logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        deterministic=True,
        accumulate_grad_batches=args.accum_grad_batches,
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
