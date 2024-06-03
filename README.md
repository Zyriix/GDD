# GDD - Official Pytorch implementation
![image](./main.png)
**Diffusion Models Are Innate One-Step Generators**

Bowen Zheng, Tianming Yang

[ariv](https://arxiv.org/abs/2405.20750)

>Diffusion Models (DMs) have achieved great success in image generation and
other fields. By fine sampling through the trajectory defined by the SDE/ODE
solver based on a well-trained score model, DMs can generate remarkable high-
quality results. However, this precise sampling often requires multiple steps and is
computationally demanding. To address this problem, instance-based distillation
methods have been proposed to distill a one-step generator from a DM by having
a simpler student model mimic a more complex teacher model. Yet, our research
reveals an inherent limitations in these methods: the teacher model, with more steps
and more parameters, occupies different local minima compared to the student
model, leading to suboptimal performance when the student model attempts to
replicate the teacher. To avoid this problem, we introduce a novel distributional
distillation method, which uses an exclusive distributional loss. This method
exceeds state-of-the-art (SOTA) results while requiring significantly fewer training
images. Additionally, we show that DMs’ layers are activated differently at different
time steps, leading to an inherent capability to generate images in a single step.
Freezing most of the convolutional layers in a DM during distributional distillation
leads to further performance improvements. Our method achieves the SOTA results
on CIFAR-10 (FID 1.54), AFHQv2 64x64 (FID 1.23), FFHQ 64x64 (FID 0.85)
and ImageNet 64x64 (FID 1.16) with great efficiency. Most of those results are
obtained with only 5 million training images within 6 hours on 8 A100 GPUs. This
breakthrough not only enhances the understanding of efficient image generation
models but also offers a scalable framework for advancing the state of the art in
various applications.


- Validation codes and checkpoint will release in a few days.
- Training codes will be released once the 
