"""
Project: "Deep-learning-based decomposition of overlapping-sparse images:
          application at the vertex of neutrino interactions"
Paper: https://arxiv.org/abs/2310.19695.
Author: Dr. Saul Alonso-Monsalve
Contact: salonso@ethz.ch/saul.alonso.monsalve@cern.ch
Description: argparse arguments for each neural network.
"""

import argparse


def args_transformer(version=1):
    """
    Create an argument parser for Transformer model configuration.

    Args:
        version (int): The version of the Transformer model configuration (1 or 2).

    Returns:
        argparse.ArgumentParser: An argument parser with options for configuring the Transformer model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="path of dataset file")
    parser.add_argument("--metadata_path", type=str, default=None, help="path of metadata file")
    parser.add_argument("--event_in_folder", type=int, default=100000, help="number of events in each folder")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--max_p", type=int, default=5, help="maximum number of protons")

    parser.add_argument("--max_p_contained", type=int, default=5, help="maximum number of protons contained")
    parser.add_argument("--max_p_exiting", type=int, default=1, help="maximum number of protons exiting")

    parser.add_argument("--mom_smearing_p", type=float, default=0.07, help="momentum smearing for protons exiting")
    parser.add_argument("--mom_smearing_mu", type=float, default=0.07, help="direction smearing for protons exiting")
    

    parser.add_argument("--cube_size", type=float, default=10.27, help="cube size in mm")
    parser.add_argument("--va_size", type=int, default=7, help="VA region size in cubes")
    parser.add_argument("--pad_value", type=int, default=-10000, help="VA region size in cubes")
    parser.add_argument("-hs", "--hidden", type=int, default=192, help="hidden size of transformer model")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout of the model")
    parser.add_argument("-el", "--encoder_layers", type=int, default=10, help="number of encoder layers")
    parser.add_argument("-dl", "--decoder_layers", type=int, default=10, help="number of decoder layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=16, help="number of attention heads")
    parser.add_argument("-b", "--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=12420, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=32, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=2e-3, help="learning rate of the optimiser")
    parser.add_argument("-ag", "--accum_grad_batches", type=int, default=4, help="batches for gradient accumulation")
    parser.add_argument("--cosine_annealing_steps", type=int, default=400, help="scheduler steps")
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-2, help="weight_decay of the optimiser")
    parser.add_argument("-b1", "--beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("-b2", "--beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--eps", type=float, default=1e-9, help="value to prevent division by zero")
    parser.add_argument('-ws', '--warmup_steps', type=int, default=20, help='Maximum number of warmup steps')
    parser.add_argument("--save_dir", type=str, default="logs", help="log save directory")
    parser.add_argument("--name", type=str, default="v1", help="model name")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")
    parser.add_argument("--early_stop_patience", type=int, default=0, help="early stopping patience (0 means no early stopping)")
    parser.add_argument("--save_top_k", type=int, default=1, help="save top k checkpoints")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="checkpoint name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="name of the checkpoint to load")
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='list of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")

    return parser


def args_gan():
    """
    Create an argument parser for GAN (generative adversarial network) model configuration.

    Returns:
        argparse.ArgumentParser: An argument parser with options for configuring the GAN model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="path of dataset file")
    parser.add_argument("--metadata_path", type=str, default=None, help="path of metadata file")
    parser.add_argument("--gan_ind_path", type=str, default=None, help="path of gan index file")
    parser.add_argument("--particle", type=str, default="proton_contained", help="particle type")

    parser.add_argument("--save_dir", type=str, default="logs", help="log save directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="checkpoint name")
    parser.add_argument("--save_top_k", type=int, default=1, help="save top k checkpoints")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")

    parser.add_argument("--name", type=str, default="v1", help="model name")

    parser.add_argument("-va", "--va_size", type=int, default=7, help="VA region size in cubes")
    parser.add_argument("-cp", "--cube_size", type=float, default=10.27, help="cube size in mm")
    parser.add_argument("-pd", "--pad_value", type=int, default=-10000, help="pad value")
    parser.add_argument("-ims", "--img_size", type=int, default=5, help="image size in pixels")

    parser.add_argument("-is", "--input_size", type=int, default=1, help="input dimension (per cube)")
    parser.add_argument("-ls", "--label_size", type=int, default=7, help="number of labels (kinematic parameters)")
    parser.add_argument("-ns", "--noise_size", type=int, default=512, help="size of the noise")
    parser.add_argument("-hs", "--hidden", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=2, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-dr", "--dropout", type=float, default=0.1, help="dropout of the model")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of the optimiser")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="weight_decay of the optimiser")
    parser.add_argument('-cr', '--crit_repeats', type=int, default=5, help='Critic iterations per generator')
    parser.add_argument('-lgp', '--lambda_gp', type=int, default=10, help='Lambda value for gradient penalty')
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='list of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")

    parser.add_argument("--load_checkpoint", type=str, default=None, help="name of the checkpoint to load")
    
    return parser

def args_cnf():
    """
    Create an argument parser for GAN (generative adversarial network) model configuration.

    Returns:
        argparse.ArgumentParser: An argument parser with options for configuring the GAN model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None, help="path of dataset file")
    parser.add_argument("--metadata_path", type=str, default=None, help="path of metadata file")
    parser.add_argument("--cnf_ind_path", type=str, default=None, help="path of cnf index file")
    parser.add_argument("--particle", type=str, default="proton_contained", help="particle type")

    parser.add_argument("--save_dir", type=str, default="logs", help="log save directory")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Checkpoint path")
    parser.add_argument("--checkpoint_name", type=str, default="v1", help="checkpoint name")
    parser.add_argument("--save_top_k", type=int, default=1, help="save top k checkpoints")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="steps between logs")

    parser.add_argument("--name", type=str, default="v1", help="model name")

    parser.add_argument("-va", "--va_size", type=int, default=7, help="VA region size in cubes")
    parser.add_argument("-cp", "--cube_size", type=float, default=10.27, help="cube size in mm")
    parser.add_argument("-pd", "--pad_value", type=int, default=-10000, help="pad value")
    parser.add_argument("-ims", "--img_size", type=int, default=5, help="image size in pixels")

    parser.add_argument("-is", "--input_size", type=int, default=1, help="input dimension (per cube)")
    parser.add_argument("-ls", "--label_size", type=int, default=7, help="number of labels (kinematic parameters)")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-nt", "--num_transformers", type=int, default=8, help="number of transformers")
    parser.add_argument("-nb", "--num_blocks_in_MADE", type=int, default=2, help="number of blocks in MADE")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=8, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate of the optimiser")
    parser.add_argument("-wd", "--weight_decay", type=float, default=0, help="weight_decay of the optimiser")
    parser.add_argument('--gpus', nargs='*',  # 'nargs' can be '*' or '+' depending on your needs
                        default=[0],  # Default list
                        help='list of GPUs to use (more than 1 GPU will run the training in parallel)'
                        )
    parser.add_argument("--num_nodes", type=int, default=1, help="number of nodes to use")

    parser.add_argument("--load_checkpoint", type=str, default=None, help="name of the checkpoint to load")
    
    return parser

