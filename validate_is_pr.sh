#!/bin/bash
npz_path=(
   results/cifar_uncond/image_pack.npz
   results/imagenet/image_pack.npz
)

ref_path=(
   "none"
   iresults/imagenet/VIRTUAL_imagenet64_labeled.npz
)

for ((j=0;j<${#npz_path[*]};j++))
do 
   cur_npz=${npz_path[j]}
    ref_npz=${ref_path[j]}
   python evaluations/evaluator.py $cur_npz $ref_npz
   
done
