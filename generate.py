# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modified work Copyright 2024 Bowen Zheng
# Center for Excellence in Brain Science and Intelligence Technology 
# Chinese Academy of Sciences
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc,persistence
import torch.nn as nn
import torch.nn.functional as F
import torch.functional
import numpy as np
from torch.nn.functional import silu, relu, gelu
from torchvision import transforms
import json

#----------------------------------------------------------------------------
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, **kwargs
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def one_step_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=2e-3, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,**kwargs
):

    tmax = torch.full([latents.shape[0],1,1,1],sigma_max).to(latents.device).to(torch.float64)
    xt = latents.to(torch.float64) * tmax
    denoised = net(xt, tmax, class_labels=class_labels, force_fp32=True).to(torch.float64)

    return denoised

def two_step_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=2e-3, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, middle=1,
):
    t = torch.full([latents.shape[0],1,1,1],sigma_max).to(latents.device).to(torch.float64)
    u = torch.full_like(t,middle)
    s = torch.full_like(u,2e-3)
    xt = latents.to(torch.float64) * t

    denoised = net(xt, t, u, class_labels=class_labels, force_fp32=True).to(torch.float64)
    v1 = (denoised-xt)/t*(t-u) + xt
    denoised2 = net(v1, u, s, class_labels=class_labels, force_fp32=True).to(torch.float64)

    return denoised2


def multi_step_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=2e-3, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,**kwargs
):
    num_steps+=1
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    
    xt = latents.to(torch.float64)*t_steps[0]
    for i,(cur_t, next_t) in enumerate(zip(t_steps[:-1],t_steps[1:])):
        denoised = net(xt, cur_t, next_t, class_labels=class_labels, force_fp32=True).to(torch.float64)
        if i!=num_steps-2:
            xt = (denoised-xt)/cur_t*(cur_t-next_t) + xt
        else:
            xt = denoised

    return xt

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.
class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--ema', help='ema weight?', metavar='BOOL',                                type=bool, default=True, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)
@click.option('--middle',      help='sigma of intermediate step in two-step generator', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--sampler',                  help='Ablate ODE solver', metavar='slope_score|csm|slope|default',                       type=click.Choice(['multi_step','two_step','edm','one_step']), default='one_step', show_default=True)
@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):

    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net =  pickle.load(f)['ema'] if sampler_kwargs['ema'] else pickle.load(f)['net']
        net = net.eval().requires_grad_(False).to(device)
    del sampler_kwargs['ema']
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
    have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
    sampler_fn = edm_sampler

    sampler_fn = one_step_sampler if sampler_kwargs['sampler'] == 'one_step' else sampler_fn
    sampler_fn = two_step_sampler if sampler_kwargs['sampler'] == 'two_step' else sampler_fn
    sampler_fn = multi_step_sampler if sampler_kwargs['sampler'] == 'multi_step' else sampler_fn
    del sampler_kwargs['sampler']
    labels = []
    all_images = []

    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

        class_labels = None
        if net.label_dim:
            random_cls_idx = rnd.randint(net.label_dim, size=[batch_size], device=device)
            class_labels = torch.eye(net.label_dim, device=device)[random_cls_idx]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        images = sampler_fn(net, latents, class_labels,  randn_like=rnd.randn_like, **sampler_kwargs)
        # Save images.
        gathered_samples = [torch.zeros_like(images) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_samples, images) # support multi-gpu when evaluating inception score/precisoin recall
        gathered_samples = torch.cat(gathered_samples)
        gathered_samples_np = (gathered_samples * 127.5 + 128).clip(0,255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        all_images.append(gathered_samples_np)
        images_np = (images * 127.5 + 128).clip(0,255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        
        if class_labels is not None:
            gathered_labels = [torch.zeros_like(random_cls_idx) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered_labels, random_cls_idx)
            labels.append(torch.cat(gathered_labels).cpu().numpy())

    all_images = np.concatenate(all_images,axis=0)

    # save to npz for inception score, precision and recall
    if net.label_dim:
        labels = np.concatenate(labels,axis=0)
        np.savez(f"{outdir}/image_pack",all_images,labels)
    else:
        np.savez(f"{outdir}/image_pack",all_images)

    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
