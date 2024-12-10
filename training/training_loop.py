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

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import dill
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
import torch.nn as nn
import lpips
from training.network_d import Discriminator, PretrainedD
from piq import LPIPS
from itertools import chain
from torch.cuda.amp import GradScaler,autocast
import torchvision as tv
import math

#----------------------------------------------------------------------------
class Normalize(nn.Module):
    def forward(self,in_feat,eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        return in_feat/(norm_factor+eps)
def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    d_network_kwargs    = {},
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    ema_beta    = 0.,     # EMA ramp-up coefficient, None = no rampup.
    augment_p = 0.2,
    lr_rampup_kimg      = 0,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    ada_kimg =500,
    ada_interval = 4,
    ada_target = 0.,
    r1_interval = 16,
    lr=1e-4,
    dlr=1e-4,
    cond=0,
    interp224=True,
    backbones=['deit_base_distilled_patch16_224', 'tf_efficientnet_lite0'],
    diffaug=True,
    optimizer_type='adam',
    freeze_layer = []
):
    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    common_kwargs = dict(c_dim=dataset_obj.label_dim, img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels)

    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    fp16 = d_network_kwargs.fp16
    feature_net = None

    dist.print0("LOSS TYPE",loss_kwargs.d_type)
    if not loss_kwargs.teacher_only:
        D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminatorOnly',
        backbones=backbones,
        diffaug=diffaug,
        interp224=(dataset_obj.resolution < 224 and interp224),
        backbone_kwargs=dnnlib.EasyDict(),
        )
        dist.print0("INterpolate to 224",dataset_obj.resolution < 224 and interp224)
        D_kwargs.backbone_kwargs.cout = 64
        D_kwargs.backbone_kwargs.expand = True
        D_kwargs.backbone_kwargs.proj_type = 2  
        D_kwargs.backbone_kwargs.num_discs = 4
        D_kwargs.backbone_kwargs.cond = cond
        D_kwargs.backbone_kwargs.rand_embedding = False

        D_kwargs.backbone_kwargs.im_res = dataset_obj.resolution if dataset_obj.resolution>=224 or not interp224 else 256 
        dist.print0("Conditional",cond)
        D_feature_kwargs = copy.deepcopy(D_kwargs)
        D_feature_kwargs.class_name = 'pg_modules.discriminator.ProjectedFeatureOnly'
        feature_net = dnnlib.util.construct_class_by_name(**D_feature_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        D_kwargs.channels = [f.CHANNELS for _,f in feature_net.feature_networks.items()]
        D_kwargs.resolutions = [f.RESOLUTIONS for _,f in feature_net.feature_networks.items()]
        dist.print0("RESOLUTIONS",D_kwargs.resolutions)
        discriminator = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(True).to(device) # subclass of torch.nn.Module

    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.ones([batch_gpu, net.label_dim], device=device)
            auglabels = torch.ones([batch_gpu, 1], device=device)

            if loss_kwargs.teacher_type!='none' or loss_kwargs.multi_step_g>1:
                misc.print_module_summary(net, [images,sigma, sigma,labels])
            else:
                misc.print_module_summary(net, [images, sigma,labels])

            if not loss_kwargs.teacher_only:
                if loss_kwargs.d_type=='xl':
                    feat = feature_net(images)
                    misc.print_module_summary(discriminator, [feat,labels])
                else:
                    misc.print_module_summary(discriminator, [images])

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    g_params = net.parameters()
    if len(freeze_layer)!=0:
        g_params = []
        for k,v in net.named_parameters():
            if ("block" in k or "up" in k or "down" in k or "in" in k) and any([fl in k for fl in freeze_layer]):
                dist.print0("Skipping addition to the optimizer:",k)
                v.requires_grad_(False)
                continue
            g_params.append(v)

    ddp_g = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    ema = copy.deepcopy(net).eval().requires_grad_(False) 

    if optimizer_type=='adam':
        optimizer = torch.optim.Adam(params=g_params, lr=lr, betas=[0,0.99], eps=1e-8)
    else:
        optimizer = torch.optim.RAdam(params=g_params, lr=lr, betas=[0,0.99], eps=1e-8)

    if not loss_kwargs.teacher_only:
        ddp_d = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[device], broadcast_buffers=False)

        if optimizer_type=='adam':
            d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=dlr*r1_interval/(r1_interval+1), betas=[0,0.99**(r1_interval/(r1_interval+1))], eps=1e-8)
        else:
            d_optimizer = torch.optim.RAdam(params=discriminator.parameters(), lr=dlr*r1_interval/(r1_interval+1), betas=[0,0.99**(r1_interval/(r1_interval+1))], eps=1e-8)
    else:
        ddp_d = None
        feature_net = None

    dist.print0('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        dist.print0("AUGMENT P",augment_p)
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # load tearcher net 
    score_net=None
    if loss_kwargs.target != 'none':
        dist.print0(f'Loading SCORE NET from "{loss_kwargs.target}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(loss_kwargs.target, verbose=(dist.get_rank() == 0)) as f:
            score_net = pickle.load(f)['ema'].eval().requires_grad_(False).to(device)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow

    # load lpips when using instance-based distillation
    lpips_net = None
    if loss_kwargs.teacher_type!='none':
        lpips_net = lpips.LPIPS(net='vgg').to(device)
        lpips_net.eval().requires_grad_(False)
    loss_fn = dnnlib.util.construct_class_by_name(g=ddp_g,d=ddp_d,feature_net=feature_net,freeze_layer=freeze_layer,lpips_net=lpips_net,score_net=score_net,**loss_kwargs) 

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    batch_idx=0

    DO_GMAIN=0
    DO_DMAIN=1
    DO_DR1=2
    dist.print0("R1 TYPE",loss_kwargs.r1_type)
  
    while True:           
        training_ratio=cur_nimg / (total_kimg * 1000)
        images, labels = next(dataset_iterator)
        images = images.to(device).to(torch.float32) / 127.5 - 1
        labels = labels.to(device)

        # set the training order
        if not loss_kwargs.teacher_only:
            stages = [DO_GMAIN,DO_DMAIN] # Generator go first then Discriminator
            stages_opt = [optimizer, d_optimizer]
            stages_net = [net,discriminator]
            stages_gain = [1,1]
            if loss_kwargs.r1_type!='none' and batch_idx!=0 and batch_idx%r1_interval==0:
                stages.append(DO_DR1)
                stages_opt.append(d_optimizer)
                stages_net.append(discriminator)
                stages_gain.append(r1_interval)
        else:
            # if only use instance-based distillation, skip discriminator iters
            stages = [DO_GMAIN]
            stages_opt = [optimizer]
            stages_net = [net]
            stages_gain = [1]
        
        for stage, opt, cur_net, gain in zip(stages, stages_opt, stages_net, stages_gain):
            opt.zero_grad(set_to_none=True)
            loss_fn.accumulate_gradients(images=images, class_labels=labels,augment_pipe=augment_pipe, stage=stage, gain=gain,cur_nimg=cur_nimg)
            
            # Check NAN, this can be commented when training is stable
            # NAN_FLAG =False
            for param in cur_net.parameters():
                if param.grad is not None:
                    # if not NAN_FLAG and torch.any(torch.isnan(param.grad)):
                    #    NAN_FLAG=True
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            opt.step()  
            # assert not NAN_FLAG, f"NAN DETECTED AT STAGE {stage}" 

        batch_idx+=1

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None and ema_rampup_ratio!=0:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        if ema_beta == 0:
            ema_r = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8)) 
        else:
            ema_r = ema_beta
        
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_r))

        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
            
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            # data = dict(net=net, d=discriminator, ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))

            data = dict(net=net, ema=ema, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time

        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
