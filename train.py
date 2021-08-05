import os
import sys
import argparse
from glob import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from thop import profile

import model
from utils import torch_msssim, ops
from anchors import balle
from torchvision import datasets, transforms
from datetime import datetime

import lpips

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda() # [n_out, n_in, k_x, k_y]

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    self.weight_x = nn.Parameter(data=kernel_x.repeat(1,3,1,1), requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y.repeat(1,3,1,1), requires_grad=False)

  def forward(self, x):
    grad_x = nn.functional.conv2d(x, self.weight_x, padding=1)
    grad_y = nn.functional.conv2d(x, self.weight_y, padding=1)
    gradient = torch.tanh(torch.abs(grad_x) + torch.abs(grad_y))
    return gradient

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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    return train_loader

def load_multi_data(train_data_dir, attack_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    attack_dataset = datasets.ImageFolder(
        attack_data_dir, 
        transform=train_transform
    )
    im_dataset = torch.utils.data.ConcatDataset([train_dataset, attack_dataset])
    train_loader = torch.utils.data.DataLoader(
        im_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
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
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric).to(dev_id)
        print("[ ARCH  ]:", MODEL, quality, args.metric)
    image_comp = nn.DataParallel(image_comp, device_ids=[0]) 
    # loss_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)
    loss_func = lpips.LPIPS(net='alex').to(dev_id) # best forward scores

    lamb = args.lamb_attack
    lr_decay_iters = [70-ckpt_index,80-ckpt_index,90-ckpt_index,95-ckpt_index]
    # lr_decay_iters = [5,10,15]
    decay_gamma = 0.33

    print("Lambda:", lamb)
    # model_dir = f"{args.model}-{args.quality}"
    # ckpt_dir = f"./ckpts/attack/{model_dir}"
    
    # Anchor
    ckpt_dir = f"./ckpts/attack/anchor/lpips"

    # optimizer 
    optimizer = torch.optim.Adam(image_comp.parameters(),lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_iters, gamma=decay_gamma, last_epoch=-1)
    
    for epoch in range(200):    
        bpp_epoch, loss_epoch = 0., 0.  
        train_loader = load_data('/workspace/ct/datasets/datasets', batch_size)
        # train_loader = load_multi_data('/workspace/ct/datasets/datasets', f'/workspace/ct/datasets/attack/{model_dir}', batch_size)
        # train_loader = load_data('/workspace/ct/datasets/attack/hyper-1', batch_size)
        for step, (batch_x, targets) in enumerate(train_loader):
            batch_x = batch_x.to('cuda')
            num_pixels = batch_x.size()[0]*batch_x.size()[2]*batch_x.size()[3]
            output, y_main, y_hyper, p_main, p_hyper = image_comp(batch_x, TRAINING, CONTEXT, POSTPROCESS)
            
            # dloss = 1. - loss_func(batch_x, output)
            lpips_loss = torch.mean(loss_func(batch_x, output))
            l2_loss = torch.mean((batch_x - output)**2)

            dloss = lpips_loss + l2_loss
            train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
            train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
            bpp = train_bpp_main + train_bpp_hyper
            loss = dloss + lamb * bpp
            print('step:', step, 'LPIPS:', lpips_loss.item(), "mse:", l2_loss.item(), 'loss:', loss.item(), 'bpp:', bpp.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            bpp_epoch += bpp.item()
            loss_epoch += loss.item()
            if step % 1000 == 0:
                torch.save(image_comp.module.state_dict(), os.path.join(ckpt_dir,'ae_%d_%d_%0.8f_%0.8f.pkl' % (epoch, step, loss_epoch/(step+1), bpp_epoch/(step+1))))
        
        lr_scheduler.step()
        torch.save(image_comp.module.state_dict(), os.path.join(ckpt_dir,'ae_%d_%0.8f_%0.8f.pkl' % (epoch, loss_epoch/(step+1), bpp_epoch/(step+1))))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NIC config
    parser.add_argument("-cn", "--ckpt_num", type=int,
                        help="load checkpoint by step number")
    parser.add_argument("-l", "--lamb", type=float,
                        default=6400., help="lambda")
    parser.add_argument("-j", "--job", type=str, default="", help="job name")
    parser.add_argument('--ctx', dest='context', action='store_true')
    parser.add_argument('--no-ctx', dest='context', action='store_false')
    parser.add_argument('--post', dest='post', action='store_true')

    parser.add_argument('-itx',    dest='iter_x', type=int, default=0,          help="iter step updating x")
    parser.add_argument('-ity',    dest='iter_y', type=int, default=0,          help="iter step updating y")
    parser.add_argument('-m',      dest='model',  type=str, default="hyper", help="compress model in 'factor','hyper','context','nonlocal'")
    parser.add_argument('-metric', dest='metric', type=str, default="ms-ssim",  help="mse or ms-ssim")
    parser.add_argument('-q',      dest='quality',type=int, default="2",        help="quality in [1-8]")
    
    # attack config
    parser.add_argument('-step',dest='steps',       type=int,   default=10001,  help="attack iteration steps")
    parser.add_argument("-la",  dest="lamb_attack", type=float, default=0.2,    help="attack lambda")
    parser.add_argument("-lr",  dest="lr_attack",   type=float, default=0.001,  help="attack learning rate")
    parser.add_argument("-s",   dest="source",      type=str,   default=None,   help="source input image")
    parser.add_argument("-t",   dest="target",      type=str,   default=None,   help="target image")

    parser.add_argument('--log', dest='log', action='store_true')
    args = parser.parse_args()

    checkpoint = None
    if args.model == "nonlocal":
        checkpoint = glob('./ckpts/%d_%s/ae_%d_*' %(int(args.lamb), args.job, args.ckpt_num))[0]
        print("[CONTEXT]:", args.context)
        print("==== Loading Checkpoint:", checkpoint, '====')
    
    train(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))