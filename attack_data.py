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
from datetime import datetime
import time

import coder

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

def add_noise(x):
    noise = np.random.uniform(-0.5, 0.5, x.size())
    noise = torch.Tensor(noise).cuda()
    return x + noise

def attack(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):

    TRAINING = True
    dev_id = "cuda:0"
    # read image
    precise = 16
    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0
    # print('====> Encoding Image:', im_dir)

    ## model initalization
    MODEL = args.model
    quality = args.quality
    arch_lists = ["factorized", "hyper", "context", "cheng2020", "nlaic", "elic"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"
    if MODEL == "elic":
        image_comp = model.ImageCompression(256)
        image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
        # image_comp.load_state_dict(torch.load(checkpoint_dir).state_dict())
        # torch.save(image_comp.state_dict(), "./checkpoints/elic-0.0.1/ae.pkl")
        image_comp.to(dev_id).eval()
        # print("[ ARCH  ]:", MODEL) 

    # if MODEL == "nlaic":
    #     # index - [0-15]
    #     models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
    #       "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]
    #     model_index = args.quality
    #     M, N2 = 192, 128
    #     if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
    #         M, N2 = 256, 192
    #     image_comp = model.Image_coding(3, M, N2, M, M//2)
    #     ######################### Load Model #########################
    #     image_comp.load_state_dict(torch.load(
    #         os.path.join(checkpoint_dir, models[model_index] + r'.pkl'), map_location='cpu'))

    if MODEL in ["factorized", "hyper", "context", "cheng2020"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric).to(dev_id)
        # print("[ ARCH  ]:", MODEL, quality, args.metric)
    # Gradient Mask
    # gnet = Gradient_Net().to(dev_id)
    #msssim_func = msssim_func.cuda()

    # img_s = Image.open(source_dir).resize((16,16))
    img_s = Image.open(args.source)
    # img_s = np.array(img_s)/255.0/5.0+0.5
    img_s = np.array(img_s)/255.0

    if len(img_s.shape) < 3:
        H, W = img_s.shape
        img_s = np.tile(img_s.reshape((H,W,1)), (1,1,3))
    else:
        H, W, _ = img_s.shape

    num_pixels = H * W
    H_PAD = int(tile * np.ceil(H / tile))
    W_PAD = int(tile * np.ceil(W / tile))
    PADDING = 0
    
    im_s = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im_s[PADDING:H+PADDING, PADDING:W+PADDING, :] = img_s[:, :, :3]
    im_s = torch.FloatTensor(im_s)
    im_s = im_s.permute(2, 0, 1).contiguous()
    im_s = im_s.view(1, C, H_PAD, W_PAD).to(dev_id)

    if args.target != None:
        # print("[ ===== ]: Target Attack")
        # img_t = Image.open(args.target).resize((64,64))
        img_t = Image.open(args.target)
        # img_t = np.array(img_t)/255.0/10.0+0.5
        img_t = np.array(img_t)/255.0

        if len(img_t.shape) < 3:
            H, W = img_t.shape
            img_t = np.tile(img_t.reshape((H,W,1)), (1,1,3))
        else:
            H, W, _ = img_t.shape
        
        im_t = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
        im_t[PADDING:H+PADDING, PADDING:W+PADDING, :] = img_t[:, :, :3]
        im_t = torch.FloatTensor(im_t)        
        im_t = im_t.permute(2, 0, 1).contiguous()
        im_t = im_t.view(1, C, H_PAD, W_PAD).to(dev_id)
        with torch.no_grad():
            output_t, _, _, _, _ = image_comp(im_t, False, CONTEXT, POSTPROCESS)
        
        ## TODO: background masking
        mask = torch.ones(1,C,H_PAD,W_PAD)
        # mask[:,:, PADDING:H+PADDING, PADDING:W+PADDING] = torch.zeros(1,C,H,W)
        # x0,y0 = 83,78
        # x1,y1 = 151,103
        x0,y0 = 0,0
        x1,y1 = W,H
        mask[:,:, y0:y1, x0:x1] = torch.zeros(1,C,y1-y0,x1-x0) #(y0, y1), (x0, x1)
        mask_bkg = mask.to(dev_id)
        mask_tar = 1. - mask_bkg

    mssim_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)
    with torch.no_grad():
        # mask = gnet(im_s)
        mask = torch.ones_like(im_s)

        output_s, y_main_, y_hyper, p_main, p_hyper = image_comp(im_s, False, CONTEXT, POSTPROCESS)
        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        # print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
        # print("Original PSNR:", )
        # print("Original MS-SSIM:", )
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        lamb_bkg = 0.01
        LOSS_FUNC = "L2"
        # print("using loss func:", LOSS_FUNC)
        # print("Lambda:", lamb)
        # im = im_s.clone().detach().requires_grad_(True) + torch.randn(im_s.size()).cuda()
        noise = torch.zeros(im_s.size())
        if args.target == None:
            noise = torch.rand(im_s.size()) - 0.5

        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=args.lr_attack)
        
        # im = (im_s+noise/10.0).clone().detach().requires_grad_(True)

        # im = im_s.clone().detach().requires_grad_(True)
        # optimizer = torch.optim.Adam([im],lr=1e-3)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
        noise_range = 0.5
        for i in range(args.steps):
            # clip noise range
            # noise_clipped = torch.clamp(mask*noise, min=-noise_range, max=noise_range)
            noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
            # im_in = torch.clamp(im_s+noise_clipped, min=0., max=1.0)
            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
            if args.target != None:
                im_in = torch.clamp(im_s+noise, min=0., max=1.0)
            # print("noised:",im_in)
            # print("source:",im_s)
            # 1. NO PADDING
            im_in[:,:,H:,:] = 0.
            im_in[:,:,:,W:] = 0.
            
            output, y_main, y_hyper, p_main, p_hyper = image_comp(im_in, TRAINING, CONTEXT, POSTPROCESS)
            # output_ = torch.clamp(output, min=0., max=1.0)
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(output, 0.), 1.)

            if LOSS_FUNC == "L2":
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))                
                if args.target == None:
                    # loss_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(y_s, y_)
                    loss_o = 1. - torch.mean((im_s - output_) * (im_s - output_)) # MSE(x_, y_)
                else:
                    # loss_o = torch.mean((output_t - output_) * (output_t - output_)) # MSE(y_t, y_s)
                    loss_o = torch.mean((im_t - output_) * (im_t - output_)) # MSE(y_t, y_s)

            # L1 loss
            if LOSS_FUNC == "L1":
                loss_i = torch.mean(torch.abs(im_s - im_in))
                if args.target == None:
                    loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
                else:    
                    loss_o = torch.mean(torch.abs(output_t - output_))
            
            if LOSS_FUNC == "masked":
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in)) * mask_tar + lamb_bkg * torch.mean((im_s - im_in) * (im_s - im_in)) * mask_bkg
                loss_o = torch.mean((output_t - output_) * (output_t - output_)) * mask_tar

            if loss_i >= args.noise:
                loss = loss_i
            else:
                loss = loss_o

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # original recons
    output_ = torch.clamp(im_in, min=0., max=1.0)
    out = output_.data[0].cpu().numpy()
    out = np.round(out * 255.0)
    out = out.astype('uint8')
    out = out.transpose(1, 2, 0)

    img = Image.fromarray(out[:H, :W, :])
    filename = args.source.split('/')[-1]
    img.save("/workspace/ct/datasets/attack/hyper-3/5k-2/"+filename)

    return 0, 0


if __name__ == "__main__":
    args = coder.config()

    print("============================================================")
    print("[ IMAGE ]:", args.source, "->", args.target)

    checkpoint = None
    if args.model == "nonlocal":
        checkpoint = glob('./ckpts/%d_%s/ae_%d_*' %(int(args.lamb), args.job, args.ckpt_num))[0]
        print("[CONTEXT]:", args.context)
        print("==== Loading Checkpoint:", checkpoint, '====')

    # sort images
    images = sorted(glob("/workspace/ct/datasets/datasets/div2k/*.png"))
    N = len(images)
    index = 0
    for image in images[20000:]:
        index += 1
        print(f'[{index}/{N}]')
        args.source = image
        t_begin = time.time()
        bpp, psnr = attack(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
        print(time.time() - t_begin)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))