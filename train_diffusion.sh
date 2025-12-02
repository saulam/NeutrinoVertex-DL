#!/bin/bash

particle="proton_contained"
img_size=5
metadata_path="/scratch/salonso/sparse-nns/vertex_activity/NN_Data_compressed/metadata.pkl"
dataset_path="/scratch/salonso/sparse-nns/vertex_activity/NN_Data_compressed/{}/{}/{}/{}.zip"
gan_ind_path="/scratch/salonso/sparse-nns/vertex_activity/NN_Data_compressed/gan_ind.pkl"
save_dir="/scratch2/salonso/SFGD_Vertex_Activity/Results/diffusion/test/"
checkpoint_path="/scratch2/salonso/SFGD_Vertex_Activity/Results/diffusion/test/checkpoints"
checkpoint_name="diffusion_v1"
epochs=100
log_every_n_steps=50
batch_size=128
lr=1e-4
num_workers=16
save_top_k=5
gpus=(1)
num_nodes=1
load_checkpoint="last"

python -m train.train_diffusion \
    --metadata_path $metadata_path \
    --dataset_path $dataset_path \
    --gan_ind_path $gan_ind_path \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --num_workers $num_workers \
    --save_dir $save_dir \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --save_top_k $save_top_k \
    --particle $particle \
    --img_size $img_size \
    --gpus "${gpus[@]}" \
    --num_nodes $num_nodes \
    --log_every_n_steps $log_every_n_steps
