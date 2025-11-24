#!/bin/bash

particle=$1
metadata_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl"
dataset_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip"
gan_ind_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/gan_ind.pkl"
save_dir="/pscratch/sd/b/botaoli/SFGD_VA/Results/gan/test/"
checkpoint_path="/pscratch/sd/b/botaoli/SFGD_VA/Results/gan/test/checkpoints"
checkpoint_name=$1
epochs=500
log_every_n_steps=1
batch_size=32
lr=5e-4
lambda_gp=10
hidden=64
critic_repeats=5
num_workers=64
save_top_k=5
gpus=(0)
num_nodes=$2
load_checkpoint=last

srun python -m train.train_gan \
    --metadata_path $metadata_path \
    --dataset_path $dataset_path \
    --gan_ind_path $gan_ind_path \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --lambda_gp $lambda_gp \
    --hidden $hidden \
    --crit_repeats $critic_repeats \
    --num_workers $num_workers \
    --save_dir $save_dir \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --save_top_k $save_top_k \
    --particle $particle \
    --gpus "${gpus[@]}" \
    --num_nodes $num_nodes \
    --log_every_n_steps $log_every_n_steps \
    --load_checkpoint $load_checkpoint

