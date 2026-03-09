#!/bin/bash

particle=$1
metadata_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/metadata.pkl"
dataset_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/{}/{}/{}/{}.zip"
cnf_ind_path="/pscratch/sd/b/botaoli/SFGD_VA/Data/NN_Data_compressed/gan_ind.pkl"
save_dir="/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/"
checkpoint_path="/pscratch/sd/b/botaoli/SFGD_VA/Results/cnf/test_spline_noexiting_rotation/checkpoints_v2"
name="v2"
checkpoint_name=$1
epochs=500
log_every_n_steps=200
batch_size=512
lr=5e-4
wd=1e-5
hidden=256
num_workers=64
save_top_k=5
gpus=(0 1 2 3)
num_nodes=$2
load_checkpoint=last
early_stop_patience=50

echo "Training $particle with $gpus GPUs on $num_nodes nodes"

python -m train.train_cnf \
    --metadata_path $metadata_path \
    --dataset_path $dataset_path \
    --cnf_ind_path $cnf_ind_path \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --wd $wd \
    --hidden $hidden \
    --num_workers $num_workers \
    --save_dir $save_dir \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --name $name \
    --save_top_k $save_top_k \
    --particle $particle \
    --gpus "${gpus[@]}" \
    --num_nodes $num_nodes \
    --log_every_n_steps $log_every_n_steps \
    --load_checkpoint $load_checkpoint \
    --early_stop_patience $early_stop_patience


