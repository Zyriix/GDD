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

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import copy
import torch
from torch_utils import persistence
import torch.nn.functional as F
import pickle
import dnnlib
import numpy as np
import random
import math
from torch_utils import training_stats
from torch_utils import misc
import torch.distributed as dist
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from torch_utils import distributed as dist_utils
DO_GMAIN=0
DO_DMAIN=1
DO_DR1=2
STUDENT_GEN  = 0
TEACHER_GEN = 1
STUDENT_DIST = 2

class Loss:
    def accumulate_gradients(self, images, labels, augment_pipe=None, step='g', do_r1=False ,**kwargs): # to be overridden by subclass
        raise NotImplementedError()

class GDDLoss(Loss):
    def __init__(self, g,d, lpips_net, feature_net, score_net,freeze_layer=[],multi_step_g=1, cond=False,teacher_only=False, teacher_type='none', loss_type='ns',middle_sigma=0.8,  r1_type='ns',  r1_gamma=0.01,  max_steps=4, use_gp=True, **kwargs):
        # Generator
        self.g = g
        self.freeze_layer=freeze_layer
        self.cond = cond
        self.multi_step_g = multi_step_g
        self.middle_sigma = middle_sigma
        dist_utils.print0("Generator Steps:", multi_step_g) 

        # Discriminator
        self.feature_net =feature_net
        self.d = d
        self.loss_type = loss_type

        # GP
        self.r1_type=r1_type
        self.r1_gamma = r1_gamma
        self.use_gp=use_gp

        # Instance-based distillation
        self.teacher_type = teacher_type

        self.score_net = score_net
        self.lpips_net = lpips_net

        self.teacher_only = teacher_only
        self.max_steps = max_steps


        # Other
        self.show_info=True

        assert self.teacher_type!='none' or not self.teacher_only # one type of distillation loss at least


    def run_D(self, images, augment_pipe, class_labels=None):
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y_teacher = None
        if self.feature_net is not None:
            feat = self.feature_net(y)
            logits = self.d(feat, class_labels)
        else:
            logits = self.d(y)

        return logits

    # teacher steps when using instance-distillation 
    def run_dist_G(self, x, z, class_labels=None, augment_labels=None):
        bz = x.shape[0]
        t_shape = (bz,1,1,1)

        # set the t,u,s 
        if self.teacher_type == 'pd':
            level = torch.randint(0, int(np.log2(self.max_steps)),t_shape).to(torch.float32).to(x.device)
            max_steps = torch.exp2(level)
            t_idx = self.randint_uniform(torch.ones_like(max_steps),max_steps) # t
            s_idx =  t_idx-1 # s
            u_idx = (t_idx + s_idx)/2 #(t+s)/2
            ratio = self.max_steps / max_steps
            t_idx = t_idx * ratio
            s_idx = s_idx * ratio
            u_idx = u_idx * ratio
            max_steps  = self.max_steps


        elif self.teacher_type == 'cd':
            max_steps = self.max_steps
            t_idx = torch.randint(2,max_steps+1,t_shape).to(torch.float32).to(x.device) # t
            u_idx = t_idx - 1 # t-1
            s_idx = torch.zeros_like(t_idx) # 0

            
        elif self.teacher_type=='ctm':
            max_steps = self.max_steps
            t_idx = torch.randint(2,max_steps+1,t_shape).to(torch.float32).to(x.device) # t
            u_idx = self.randint_uniform(torch.ones_like(t_idx), t_idx-1) #u
            s_idx = self.randint_uniform(torch.zeros_like(t_idx), u_idx-1) # s


        elif self.teacher_type=='rcd':
            max_steps = self.max_steps
            t_idx = torch.full(t_shape, max_steps).to(torch.float32).to(x.device)
            s_idx = self.randint_uniform(torch.zeros_like(t_idx),t_idx-2)
            u_idx = s_idx + 1
            
        t_ratio = t_idx/max_steps
        u_ratio = u_idx/max_steps
        s_ratio = s_idx/max_steps

        t = self.cut_sigma(t_ratio)
        u = self.cut_sigma(u_ratio)
        s = self.cut_sigma(s_ratio)

        v = x + t * z
        with torch.no_grad():

            euler_v1 = self.euler_step(v, t, u, class_labels=class_labels, augment_labels=augment_labels)
            heun_v1 = self.heun_step(v, t, u, class_labels=class_labels, augment_labels=augment_labels)
            v1 = torch.where(t_idx-u_idx==1,heun_v1,euler_v1)

            euler_v2 = self.euler_step(v1, u, s, class_labels=class_labels, augment_labels=augment_labels)
            heun_v2 = self.heun_step(v1, u, s, class_labels=class_labels, augment_labels=augment_labels)
            v2 = torch.where(u_idx-s_idx==1, heun_v2, euler_v2)

            if self.teacher_type=='ctm':
                target = self.euler_step(v2, s, torch.full_like(s,2e-3), class_labels=class_labels, augment_labels=augment_labels)
                target_heun = self.heun_step(v2, s, torch.full_like(s,2e-3), class_labels=class_labels, augment_labels=augment_labels)
                target = torch.where(s_idx==1,target_heun,target)

            elif self.teacher_type=='cd':
                target = v2

            else:
                target  = (v2-v)/(t-s)*t + v
                target = target.detach()
        
        pred = self.g(v, t, s, class_labels=class_labels, augment_labels=augment_labels)
        # pred = (x0_pred-v)/t*(t-s) + v

        return pred, target

    def heun_step(self, v, t, u, class_labels, augment_labels=None):
        # adopt heun when one step
        x0 = self.score_net(v, t, class_labels=class_labels, augment_labels=augment_labels)
        d_x0 = (x0-v)/t
        v1 = d_x0*(t-u) + v
        x0_heun = self.score_net(v1, u, class_labels=class_labels, augment_labels=augment_labels)
        d_x0_heun = (x0_heun-v1)/u
        v1_heun = (0.5*d_x0 + 0.5*d_x0_heun) * (t-u) + v
        return v1_heun
    
    def euler_step(self, v, t, u, class_labels, augment_labels=None):
        x0 = self.g(v, t, u, class_labels=class_labels, augment_labels=augment_labels)
        d_x0 = (x0-v)/t
        v1 = d_x0*(t-u) + v
        return v1

    # One-Step Generator (or multi-step generator)
    def run_G(self, v, class_labels, augment_labels=None):
        t = torch.full((v.shape[0],1,1,1),80.).to(v.device) 

        if self.multi_step_g==1: # default set
            return self.g(v, t, class_labels=class_labels, augment_labels=augment_labels)
        elif self.multi_step_g==2:
            u = torch.full_like(t, self.middle_sigma)
            s = torch.full_like(u, 2e-3)
            x0 = self.g(v, t, u, class_labels=class_labels, augment_labels=augment_labels)
            v1 = (x0-v)/t*(t-u) + v
            return self.g(v1, u, s, class_labels=class_labels, augment_labels=augment_labels)
        else:

            cur_v = v
            for cur_i in range(self.multi_step_g,0,-1):
                cur_idx = torch.full(([v.shape[0],1,1,1]),cur_i).to(torch.float32).to(v.device)
                next_idx = cur_idx-1

                cur_sigma = self.cut_sigma(cur_idx/self.multi_step_g)
                next_sigma = self.cut_sigma(next_idx/self.multi_step_g)
                denoised = self.g(cur_v, cur_sigma, next_sigma, class_labels=class_labels, augment_labels=augment_labels)
                if cur_i!=1:
                    cur_v = (denoised-cur_v)/cur_sigma*(cur_sigma-next_sigma) + cur_v
                else:
                    cur_v = denoised
                if self.show_info:
                    dist_utils.print0(f"cur_idx {cur_idx[0].item()}, net_idx{next_idx[0].item()}, cur_sigma{cur_sigma[0].item()}, next_sigma{next_sigma[0].item()}")

            return cur_v



    def accumulate_gradients(self, images, class_labels, augment_pipe=None, stage=DO_GMAIN, gain=1,cur_nimg=0 ,**kwargs):
        b,c,h,w = images.shape
        device = images.device

        t = torch.full((b,1,1,1), 80.).to(device) # fixed max time step
        s = torch.full((b,1,1,1), 2e-3).to(device)   # fixed min time step
        z = torch.randn((b, c, h, w)).to(device) 
        zt = z * t

        if stage == DO_GMAIN:
            self.g = self.g.requires_grad_(True)
            if len(self.freeze_layer)!=0:
                for k,v in self.g.named_parameters():
                    if ("block" in k or "in" in k) and any([fl in k for fl in self.freeze_layer]):
                        v.requires_grad_(False)
                        if self.show_info:
                            dist_utils.print0("Freezing Block",k)

            g_loss = 0 
            if not self.teacher_only:
                self.d = self.d.requires_grad_(False)

                with torch.autograd.profiler.record_function('G_forward'):
                    v = self.run_G(zt, class_labels=class_labels, augment_labels=None)
                    g_logits = self.run_D(v, augment_pipe,class_labels=class_labels) 

                    if self.loss_type=='ns':
                        g_loss = sum([F.softplus(-glogit).mean() for glogit in g_logits])
                    else:
                        g_loss = sum([-glogit.mean() for glogit in g_logits])

                    training_stats.report(f'Loss/logits/fake_eff', g_logits[1])
                    g_logits = torch.cat(g_logits)
                    training_stats.report(f'Loss/logits/fake', g_logits)
                    training_stats.report(f'Loss/probs/fake', g_logits.sigmoid())

            dist_loss = 0
            score_loss = 0
            if self.teacher_type != 'none': # Instance-based distillation
                dist_pred, dist_target = self.run_dist_G(images, z, class_labels=class_labels, augment_labels=None)
                dist_pred = F.interpolate(dist_pred,(224,224),mode='bilinear')
                dist_target = F.interpolate(dist_target,(224,224),mode='bilinear')
                dist_loss = self.lpips_net(dist_pred, dist_target).mean()


                training_stats.report(f'Loss/dist/fake', dist_loss)
            with torch.autograd.profiler.record_function('G_backward'):
                (dist_loss + g_loss).mul(gain).backward()
    

        elif stage==DO_DMAIN:
            self.g = self.g.requires_grad_(False)
            self.d = self.d.requires_grad_(True)
            d_gen_loss = 0
            d_gt_loss = 0
            with torch.autograd.profiler.record_function('Dgen_forward'):
                with torch.no_grad():
                    gen_v = self.run_G(zt, class_labels=class_labels, augment_labels=None)
                d_gen_logits = self.run_D(gen_v, augment_pipe, class_labels=class_labels)
                
                if self.loss_type=='ns':
                    d_gen_loss = sum([F.softplus(dgenlogit).mean() for dgenlogit in d_gen_logits])
                else:
                    d_gen_loss = sum([F.relu(1+dgenlogit).mean() for dgenlogit in d_gen_logits])

                d_gen_logits = torch.cat(d_gen_logits)
            with torch.autograd.profiler.record_function('Dgt_forward'):
                d_gt_logits = self.run_D(images, augment_pipe, class_labels=class_labels)
              
                if self.loss_type=='ns':
                    d_gt_loss = sum([F.softplus(-dgtlogit).mean() for dgtlogit in d_gt_logits])
                else:
                    d_gt_loss = sum([F.relu(1-dgtlogit).mean() for dgtlogit in d_gt_logits])


                d_gt_logits = torch.cat(d_gt_logits)
                training_stats.report(f'Loss/logits/real', d_gt_logits)
                training_stats.report(f'Loss/signs/real', d_gt_logits.sign())

                training_stats.report(f'Loss/probs/real', d_gt_logits.sigmoid())
            
            with torch.autograd.profiler.record_function('D_backward'):
                (d_gen_loss + d_gt_loss).mul(gain).backward()

        elif stage==DO_DR1 and self.use_gp: 
            self.g = self.g.requires_grad_(False)
            self.d = self.d.requires_grad_(True)

            with torch.autograd.profiler.record_function('Dr1_forward'), conv2d_gradfix.no_weight_gradients():
                real_images = images.clone().detach().requires_grad_(True)
                d_r1_logits = self.run_D(real_images, None, class_labels=class_labels)
                sum_r1_logits = sum([dr1logit.sum() for dr1logit in d_r1_logits])

                r1_grads = torch.autograd.grad(outputs=[sum_r1_logits], inputs=[real_images], create_graph=True, only_inputs=True)[0]

                r1_penalty = r1_grads.square().sum(axis=[1,2,3])
                r1_loss = r1_penalty * self.r1_gamma / 2 
                training_stats.report('Loss/r1_penalty', r1_penalty)
                training_stats.report('Loss/D/r1_loss', r1_loss)  
    
            with torch.autograd.profiler.record_function('Dr1_backward'):
                r1_loss.mean().mul(gain).backward()

        self.show_info = False

    def randint_uniform(self, low, high):
        high = high + 0.4999
        low = low - 0.4999
        return torch.round(torch.rand_like(low, device=low.device) * (high - low) + low)

    def cut_sigma(self, ratio, sigma_max=80, sigma_min=2e-3):
        return (sigma_max **(1 / 7) +  (1-ratio) * (sigma_min ** (1 / 7) - sigma_max**(1/7))) ** 7
