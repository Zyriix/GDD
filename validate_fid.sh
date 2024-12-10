#!/bin/bash
export MKL_NUM_THREADS=1




ckpt_path=(
   all_ckpt/cifar_uncond_gdd_i.pkl
   all_ckpt/cifar_cond_gdd_i.pkl
   all_ckpt/ffhq_gdd_i.pkl
   all_ckpt/afhqv2_gdd.pkl
   all_ckpt/imagenet_gdd_i.pkl

)
ref_path=(
   refs/cifar10-32x32.npz
   refs/cifar10-32x32.npz
   refs/ffhq-64x64.npz
   refs/afhqv2-64x64.npz
   refs/imagenet-64x64.npz
)
log_name=(
   cifar_uncond
   cifar_cond
   ffhq
   afhqv2
   imagenet
)

ema=True
num_nodes=8

for ((j=0;j<${#ckpt_path[*]};j++))
do 
   cur_ckpt=${ckpt_path[j]}
   cur_log=${log_name[j]}
   cur_ref=${ref_path[j]}
   
   images_path=results/$cur_log

   torchrun --standalone --nproc_per_node=$num_nodes generate.py --steps=1 --sampler=one_step --outdir=$images_path --seeds=0-49999 --subdirs \
   --network=$cur_ckpt --ema=$ema

   torchrun --standalone --nproc_per_node=$num_nodes fid.py calc --images=$images_path \
   --ref=$cur_ref
   
done