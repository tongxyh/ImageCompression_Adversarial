import os
import sys
import time
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
from PIL import Image
from thop import profile

import model
from utils import torch_msssim, ops
from anchors import balle
from torchvision import datasets, transforms
from datetime import datetime
import coder
import lpips


def load_data(train_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return train_loader

def add_noise(x):
    noise = np.random.uniform(-0.5, 0.5, x.size())
    noise = torch.Tensor(noise).cuda()
    return x + noise

def train(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):

    TRAINING = True
    dev_id = "cuda:0"
    C = 3
    ckpt_index = 0
    batch_size = 8
    # print('====> Encoding Image:', im_dir)

    ## model initalization
    MODEL = args.model
    quality = args.quality
    arch_lists = ["factorized", "hyper", "context", "cheng2020", "nlaic", "elic"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"
    if MODEL == "elic":
        image_comp = model.ImageCompression(256)
        image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
        image_comp.to(dev_id).eval()
        print("[ ARCH  ]:", MODEL) 

    if MODEL in ["factorized", "hyper", "context", "cheng2020"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric, pretrained=args.download).to(dev_id)
        print("[ ARCH  ]:", MODEL, quality, args.metric)
        if args.download == False:
            print("[ CKPTS ]:", args.ckpt)
            image_comp.load_state_dict(torch.load(args.ckpt))
            image_comp.to(dev_id).train()
        else:
            print("[ CKPTS ]: Download from CompressAI Model Zoo", )
    # image_comp = nn.DataParallel(image_comp, device_ids=[0])
    if args.metric == "ms-ssim":
        loss_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)

    lamb = args.lamb_attack
    # lr_decay_iters = [70-ckpt_index,80-ckpt_index,90-ckpt_index,95-ckpt_index]
    lr_decay_iters = [600,1200,1800]
    decay_gamma = 0.33
    noise_thres = args.noise

    print("Lambda:", lamb)
    N_ADV=0
    print(f"num of ori/adv examples: {N_ADV}/{8-N_ADV}")
    if args.metric == "mse":
        lamb = lamb * 255. * 255.
    #Augmentated Model
    print("Refine with Adversarial examples")
    model_dir = f"{args.model}-{args.quality}"
    ckpt_dir = f"./ckpts/attack/{model_dir}/iter/{args.metric}"

    # optimizer 
    parameters = set(p for n, p in image_comp.named_parameters() if not n.endswith(".quantiles"))
    aux_parameters = set(p for n, p in image_comp.named_parameters() if n.endswith(".quantiles"))
    optimizer = torch.optim.Adam(parameters, lr=args.lr_train)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=1e-3)

    # optimizer = torch.optim.Adam(image_comp.parameters(),lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_iters, gamma=decay_gamma, last_epoch=-1)
    
    for epoch in range(1):    
        bpp_epoch, loss_epoch = 0., 0.
        train_loader = load_data(f'/workspace/ct/datasets/datasets/div2k', batch_size)
        for step, (batch_x, targets) in enumerate(train_loader):
            # noise_thres = min(args.noise, 0.00001 + (args.noise-0.00001)*step/500)
            noise_thres = args.noise
            t = time.time()
            batch_x = batch_x.to('cuda')
            num_pixels = batch_x.size()[0]*batch_x.size()[2]*batch_x.size()[3]

            # generate batch_x_adv
            im_s = batch_x[N_ADV:,:,:,:]
            noise = torch.zeros(im_s.size())
            noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
            adv_optimizer = torch.optim.Adam([noise],lr=args.lr_attack)

            lr_adv_scheduler = torch.optim.lr_scheduler.MultiStepLR(adv_optimizer, [300,600,900], gamma=0.33, last_epoch=-1)
            noise_range = 0.5
            for i in range(args.steps):
                noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
                im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

                y_main = image_comp.net.g_a(im_in)
                x_ = image_comp.net.g_s(y_main)

                output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)

                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))                
                # loss_o = 1. - torch.mean((im_in - output_) * (im_in - output_)) # MSE(x_, y_)
                loss_o = 1. - torch.mean((im_s - output_) * (im_s  - output_)) # MSE(x_, y_)
                # if i==999:
                #     print(loss_i.item(), loss_o.item())

                if loss_i >= noise_thres:
                    loss = loss_i
                else:
                    loss = loss_o

                adv_optimizer.zero_grad()
                loss.backward()
                adv_optimizer.step()
                lr_adv_scheduler.step()

            im_uint8 = torch.round(im_in * 255.0)/255.0
            batch_x_new = batch_x.detach()
            batch_x_new[N_ADV:] = torch.clamp(im_uint8, min=0., max=1.0).detach()

            output, y_main, y_hyper, p_main, p_hyper = image_comp(batch_x_new, TRAINING, CONTEXT, POSTPROCESS)
            
            # lpips_loss = torch.mean(loss_func(batch_x, output))
            if args.metric == "ms-ssim":
                dloss = 1. - loss_func(batch_x_new, output)
            if args.metric == "mse":
                
                dloss = torch.mean((batch_x_new - output)**2)

            train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
            train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
            bpp = train_bpp_main + train_bpp_hyper
            # loss = dloss + lamb * bpp
            ## about lambda: https://interdigitalinc.github.io/CompressAI/zoo.html 
            # [q3 - mse: 0.0067 * 255^2]
            # [q3 - mssim: 8.73]
            loss = lamb * dloss + bpp

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            aux_loss = image_comp.net.entropy_bottleneck.loss()
            aux_loss.backward()
            aux_optimizer.step()

            bpp_epoch += bpp.item()
            loss_epoch += loss.item()

            print('step:', step, 'loss:', loss.item(), "distortion:", dloss.item(), 'rate:', bpp.item(), 'time:', time.time()-t, "noise thres:", noise_thres)
            if step % 100 == 0:
                # torch.save(image_comp.module.state_dict(), os.path.join(ckpt_dir,'ae_%d_%d_%0.8f_%0.8f.pkl' % (epoch, step, loss_epoch/(step+1), bpp_epoch/(step+1))))
                torch.save(image_comp.state_dict(), os.path.join(ckpt_dir,'ae_%d_%d_%0.8f_%0.8f.pkl' % (epoch, step, loss_epoch/(step+1), bpp_epoch/(step+1))))
        
            lr_scheduler.step()
        # torch.save(image_comp.module.state_dict(), os.path.join(ckpt_dir,'ae_%d_%0.8f_%0.8f.pkl' % (epoch, loss_epoch/(step+1), bpp_epoch/(step+1))))
        torch.save(image_comp.state_dict(), os.path.join(ckpt_dir,'ae_%d_%0.8f_%0.8f.pkl' % (epoch, loss_epoch/(step+1), bpp_epoch/(step+1))))

if __name__ == "__main__":
    args = coder.config()
    checkpoint = None
    if args.model == "nonlocal":
        checkpoint = glob('./ckpts/%d_%s/ae_%d_*' %(int(args.lamb), args.job, args.ckpt_num))[0]
        print("[CONTEXT]:", args.context)
        print("==== Loading Checkpoint:", checkpoint, '====')
    
    train(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))