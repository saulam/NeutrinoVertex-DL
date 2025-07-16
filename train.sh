#!/bin/bash

# Default arguments
metadata_path="/scratch/salonso/sparse-nns/vertex_activity/images_sfgd/metadata.pkl"
dataset_path="/scratch/salonso/sparse-nns/vertex_activity/images_sfgd/{}/*/{}.npz"
eps=1e-12
batch_size=2048
epochs=10000
num_workers=32
lr=1e-4
accum_grad_batches=1
warmup_steps=100
cosine_annealing_steps=9900
weight_decay=0.05
beta1=0.9
beta2=0.95
save_dir="logs"
name="v2"
log_every_n_steps=10
save_top_k=1
checkpoint_path="checkpoints"
checkpoint_name="v2"
early_stop_patience=1000
gpus=(1)

python -m train.train_transformer_conf1 \
    --metadata_path $metadata_path \
    --dataset_path $dataset_path \
    --eps $eps \
    --batch_size $batch_size \
    --epochs $epochs \
    --num_workers $num_workers \
    --lr $lr \
    --accum_grad_batches $accum_grad_batches \
    --warmup_steps $warmup_steps \
    --cosine_annealing_steps $cosine_annealing_steps \
    --weight_decay $weight_decay \
    --beta1 $beta1 \
    --beta2 $beta2 \
    --save_dir $save_dir \
    --name $name \
    --log_every_n_steps $log_every_n_steps \
    --save_top_k $save_top_k \
    --checkpoint_path $checkpoint_path \
    --checkpoint_name $checkpoint_name \
    --early_stop_patience $early_stop_patience \
    --gpus "${gpus[@]}"

