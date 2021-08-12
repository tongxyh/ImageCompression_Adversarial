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
        print("[ ARCH  ]:", MODEL) 

    if MODEL in ["factorized", "hyper", "context", "cheng2020"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric).to(dev_id)
        print("[ ARCH  ]:", MODEL, quality, args.metric)
        if args.download == False:
            # load from local ckpts
            image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
            image_comp.to(dev_id).train()
    # Gradient Mask
    gnet = Gradient_Net().to(dev_id)
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
        print("[ ===== ]: Target Attack")
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
        if args.mask_loc != None:
            lamb_bkg = args.lamb_bkg
            lamb_tar = args.lamb_tar
            print(args.mask_loc, "lamb_bkg:", lamb_bkg)
        x0,x1,y0,y1 = args.mask_loc # W, H
        
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
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
        # print("Original PSNR:", )
        # print("Original MS-SSIM:", )
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        LOSS_FUNC = "L2"
        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        # im = im_s.clone().detach().requires_grad_(True) + torch.randn(im_s.size()).cuda()
        noise = torch.zeros(im_s.size())
        # if args.target == None:
            # noise = torch.rand(im_s.size()) - 0.5

        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device

        noise_range = 0.5
        print("[WARNINTG] Quantization Skipped!")
        
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
        # output_ = ops.Up_bound.apply(ops.Low_bound.apply(output, 0.), 1.)

        x_ = image_comp.net.g_s(y_main)
        output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)

        if LOSS_FUNC == "L2" and args.mask_loc == None:
            # print("[Loss] L2 with no mask")
            loss_i = torch.mean((im_s - im_in) * (im_s - im_in))                
            if args.target == None:
                # loss_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(y_s, y_)
                loss_o = torch.mean((im_s - output_) * (im_s - output_)) # MSE(x_, y_)
            else:
                loss_o = torch.mean((output_t - output_) * (output_t - output_)) # MSE(y_t, y_s)
                # loss_o = torch.mean((im_t - output_) * (im_t - output_)) # MSE(y_t, y_s)

        # L1 loss
        if LOSS_FUNC == "L1":
            loss_i = torch.mean(torch.abs(im_s - im_in))
            if args.target == None:
                loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
            else:    
                loss_o = torch.mean(torch.abs(output_t - output_))
        
        if LOSS_FUNC == "L2" and args.mask_loc != None:
            # print("[Loss] L2 with target mask")
            l1_loss = torch.mean(torch.abs(im_s - im_in) * mask_tar)
            l2_loss = torch.mean((im_s - im_in) * (im_s - im_in) * mask_tar)
            loss_tar = 0.01*l1_loss + l2_loss
            loss_i = lamb_tar * loss_tar + lamb_bkg * torch.mean((im_s - im_in) * (im_s - im_in) * mask_bkg)
            # loss_i = lamb_tar * torch.mean((im_s - im_in) * (im_s - im_in) * mask_tar) + lamb_bkg * torch.mean((im_s - im_in) * (im_s - im_in) * mask_bkg)
            # loss_i = lamb_tar * torch.mean(torch.abs(im_s - im_in) * mask_tar) + lamb_bkg * torch.mean((im_s - im_in) * (im_s - im_in) * mask_bkg)
            # loss_o = torch.mean((output_t - output_) * (output_t - output_) * mask_tar)
            loss_o = torch.mean(torch.abs(output_t - output_) * mask_tar)

        loss = loss_i + lamb * loss_o

            # noise_range = 0.9999*noise_range + 0.0001*50/255
        # bpp attack
        # train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        # train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        # bpp = train_bpp_main + train_bpp_hyper
        # lambda decay
        # loss = lamb * mse_o + (0.25 * mse_padding_o) + (mse_i + 0.25 * mse_padding_i)
        # loss = lamb * (-mse_o) + mse_i
        # loss = (mse_i-1e-3)**2.0 / (lamb *(mse_o) + 1e-6)
        
        # target loss
        # loss = mse_i - lamb * bpp

        image_comp.zero_grad()
        if noise.grad is not None:
            noise.grad.data.fill_(0)
        loss_o.backward()

        noise.grad.sign_()
        # grad_l = torch.sign(noise.grad+1e-8)
        # grad_r = torch.sign(noise.grad-1e-8)
        # noise.grad = 0.5 * (grad_l + grad_r)

        print("noise:",torch.mean(noise.grad**2))
        eps = args.lr_attack
        print("eps:", eps)
        im_adv = im_in + eps*noise.grad
        im_adv = torch.clamp(im_adv, 0., 1.)
    
        with torch.no_grad():
            im_uint8 = torch.round(im_adv * 255.0)/255.0
            print("noise:",torch.mean(noise.grad**2))
            loss_in = eps * eps * torch.mean(noise.grad**2)
            loss_in_ = torch.mean((im_adv - im_s)**2)
            loss_in_2 = torch.mean((im_s - im_in)**2)
            print("loss_in", loss_in_.item(), loss_in_2.item())

            # 1. NO PADDING
            # im_uint8[:,:,H:,:] = 0.
            # im_uint8[:,:,:,W:] = 0.
            
            # save adverserial input
            im_ =  torch.clamp(im_uint8, min=0., max=1.0)
            # print(torch.min(im_), torch.max(im_))
            # im_ =  torch.clamp(im, min=0., max=1.0)
            # torch.set_printoptions(threshold=1000000)
            # print(im_)
            fin = im_.data[0].cpu().numpy()
            fin = np.round(fin * 255.0)
            fin = fin.astype('uint8')
            fin = fin.transpose(1, 2, 0)
            img = Image.fromarray(fin[:H, :W, :])
            # img = Image.fromarray(fin[PADDING:H+PADDING, PADDING:W+PADDING, :])
            
            img.save("./attack/kodak/fake_in_%0.8f.png"%(loss_in.item())) 
            # img.save("./attack/kodak/final_in.png")

            output, y_main, y_hyper, p_main, p_hyper = image_comp(im_, False, CONTEXT, POSTPROCESS)   
            
            bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
            bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
            bpp = bpp_main + bpp_hyper
            output_ = torch.clamp(output, min=0., max=1.0)
            
            loss_out = torch.mean((output_ - output_s)**2)
            print("bpp:", bpp.item(), loss_in.item(), loss_out.item())
            out = output_.data[0].cpu().numpy()
            out = np.round(out * 255.0)
            out = out.astype('uint8')
            out = out.transpose(1, 2, 0)
            
            # img = Image.fromarray(out[PADDING:H+PADDING, PADDING:W+PADDING, :])
            img = Image.fromarray(out[:H, :W, :])
            img.save("./attack/kodak/fake_out_%0.4f_%0.8f.png"%(bpp.item(), loss_out.item()))
    
    # attacked recons
    # img.save("./attack/kodak/final_out.png")

    # original recons
    output_ = torch.clamp(output_s, min=0., max=1.0)
    out = output_.data[0].cpu().numpy()
    out = np.round(out * 255.0)
    out = out.astype('uint8')
    out = out.transpose(1, 2, 0)

    img = Image.fromarray(out[:H, :W, :])
    img.save("./attack/kodak/origin_out.png")

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

    checkpoint = args.ckpt
    bpp, psnr = attack(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))