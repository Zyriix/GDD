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

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

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

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--loss',       help='loss function', metavar='gdd|vp|ve|edm',       type=click.Choice(['gdd', 'vp', 've', 'edm']), default='gdd', show_default=True)

# Options when using instance-based disitllation
@click.option('--teacher-type',       help='teacher type when using instance based distillation(CD,CTM,etc)', metavar='BOOL',      type=str, default='none', show_default=True)
@click.option('--teacher-only',       help='only use instance-based distillation?', metavar='BOOL',       type=bool, default=False, show_default=True)
@click.option('--target',    help='teacher model when using instance-based distillation', metavar='none',  type=str, default='none', show_default=True)
@click.option('--lpips',         help='lpips loss' , metavar='MIMG',                                 type=bool, default=True, show_default=True)
@click.option('--max-steps',       help='the step of teacher model when using instance based distillation', metavar='BOOL',       type=click.IntRange(min=0), default=1024, show_default=True)

# Generators
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='|ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp','ncsnpp', 'adm']), default='ncsnpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm|gdd',       type=click.Choice(['gdd','vp', 've', 'edm']), default='gdd', show_default=True)
@click.option('--multi-step-g',         help='The step of generators',              type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--freeze',         help='Free Layer options',               type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--transfer',      help='Initialized from pretrained diffusion models', metavar='PKL|URL',   type=str)
@click.option('--middle-sigma',      help='sigma of intermediate step when use two step generators', type=click.FloatRange(min=0), default=0.8, show_default=True)

# Discriminators
@click.option('--d-type',       help='discriminator type',     type=str, default='style', show_default=True)
@click.option('--d-pretrained',    help='pretrained discriminator?',   type=bool, default=True, show_default=True)
@click.option('--diffaug',       help='use diffaug of not',        type=bool, default=False, show_default=True)
@click.option('--r1-type',       help='r1 type',      type=str, default='hingle', show_default=True)
@click.option('--use-gp',    help='gradient penalty?',   type=bool, default=True, show_default=True)
@click.option('--loss-type',       help='loss type of discriminator',      type=str, default='ns', show_default=True)
@click.option('--r1-gamma',      help='gamma of r1 p',                        type=click.FloatRange(min=0), default=0.01, show_default=True)
@click.option('--interp224',         help='proj to 224 before feed to discriminator?',              type=bool, default=True, show_default=True)
@click.option('--backbone',          help='The feature network for discriminator',                type=click.IntRange(min=0), default=0, show_default=True)

# Augmentation (disabled by default)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--augment-p',    help='path of pretrained score model', metavar='none',  type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ada-target',      help='target value in ada (default disabled)',                           type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)


# Hyperparameters.
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0), default=4e-4, show_default=True)
@click.option('--dlr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0), default=4e-4, show_default=True)

@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))

@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0), default=200, show_default=True)

@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)

@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--ema-warmup',           help='EMA half-life warmup', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.05, show_default=True)
@click.option('--ema-beta',           help='EMA half-life ratio',                              type=click.FloatRange(min=0), default=0, show_default=True)

@click.option('--optimizer-type',       help='optimizer type', metavar='BOOL',      type=str, default='adam', show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0, show_default=True)
@click.option('--weight-decay',            help='weight decay', metavar='FLOAT',                             type=click.FloatRange(min=0), default=0, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--persistent-workers',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=25, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0,0.99], eps=1e-8)
    c.d_network_kwargs = dnnlib.EasyDict()
    c.cond = opts.cond
    c.diffaug=opts.diffaug
    c.optimizer_type = opts.optimizer_type

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_name = dataset_obj.name
        c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
        c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
        if opts.cond and not dataset_obj.has_labels:
            raise click.ClickException('--cond=True requires labels specified in dataset.json')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
    else:
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
    elif opts.precond == 'gdd':
        c.network_kwargs.class_name = 'training.networks.GDDPrecond'
    else:
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'

    if opts.loss == 'vp':
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.loss == 've':
         c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.loss == 'gdd':
        c.loss_kwargs.class_name = 'training.loss.GDDLoss'
        c.loss_kwargs.lpips = opts.lpips
        c.loss_kwargs.max_steps = opts.max_steps
        c.loss_kwargs.target = opts.target
        c.loss_kwargs.d_type = opts.d_type
        c.loss_kwargs.use_gp = opts.use_gp
        c.loss_kwargs.r1_gamma = opts.r1_gamma
        c.loss_kwargs.middle_sigma = opts.middle_sigma
        c.loss_kwargs.r1_type = opts.r1_type
        c.loss_kwargs.loss_type = opts.loss_type
        c.loss_kwargs.teacher_type = opts.teacher_type
        c.loss_kwargs.cond = opts.cond
        c.loss_kwargs.teacher_only = opts.teacher_only
        c.loss_kwargs.multi_step_g = opts.multi_step_g
    else:
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
        c.loss_kwargs.lpips = opts.lpips

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    c.d_network_kwargs.fp16 = False
    if opts.augment:
        print("Using augment")
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe')
        c.augment_kwargs.update(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        c.augment_p = opts.augment_p

    print("FP16",opts.fp16)

    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.ema_beta = float(opts.ema_beta)
    c.ema_rampup_ratio = float(opts.ema_warmup)    
    print("EMA BETA: ", c.ema_beta)
    print("EMA RAMPUP RATIO: ", c.ema_rampup_ratio)
    print("EMA HALFLIFE: ", c.ema_halflife_kimg)
    c.interp224=opts.interp224
    c.ada_target = opts.ada_target if opts.ada_target >0 else None
    
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)
    
    c.lr = opts.lr
    c.dlr = opts.dlr
    if opts.backbone==0:
        backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'] 
    elif opts.backbone==1:
        backbones=['vgg16_bn', 'tf_efficientnet_lite0'] 
    elif opts.backbone==2:
        backbones=['hf_hub:timm/vit_base_patch16_224.augreg_in21k_ft_in1k', 'hf_hub:timm/tf_efficientnet_b0.ns_jft_in1k'] 
    else:
        backbones=['tf_efficientnet_lite0'] 


    if opts.freeze == 0:
        freeze_layer = []
    elif opts.freeze==1:
        freeze_layer = ['conv','proj']
    elif opts.freeze==2:
        freeze_layer = ['conv','proj','qkv']
    elif opts.freeze==3:
        freeze_layer = ['conv','proj','qkv','skip']

    c.freeze_layer = freeze_layer
    c.backbones=backbones

    # Load from pretrained diffusion models
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer

    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-{opts.loss:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    
    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning:  {opts.precond}')
    dist.print0(f'loss:  {opts.loss}')

    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
