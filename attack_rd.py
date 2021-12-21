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
import lpips
from anchors import balle
from datetime import datetime

import coder


def attack(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):
    dev_id = "cuda:0"
    # read image
    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0
    # print('====> Encoding Image:', im_dir)

    if args.log:
        # Writer will output to ./runs/ directory by default
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

        tensor_log_path = './logs'
        if not os.path.exists(tensor_log_path):
            os.mkdir(tensor_log_path)
        writer = SummaryWriter(tensor_log_path+'/'+TIMESTAMP)
        # writer = SummaryWriter("./logs/")

    ## model initalization
    MODEL = args.model
    quality = args.quality
    arch_lists = ["factorized", "hyper", "context", "cheng2020"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"

    
    image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric, pretrained=args.download).to(dev_id)
    print("[ ARCH  ]:", MODEL, quality, args.metric)
    if args.download == False:
        print("[ CKPTS ]:", args.ckpt)
        image_comp.load_state_dict(torch.load(args.ckpt))
        image_comp.to(dev_id).train()
    else:
        print("[ CKPTS ]: Download from CompressAI Model Zoo")

    img_s = Image.open(args.source)
    filename = args.source.split("/")[-1][:-4]
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
        img_t = Image.open(args.target)
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
            output_t[:,:,H:,:] = 0.
            output_t[:,:,:,W:] = 0.            
            out = torch.clamp(output_t, min=0., max=1.0)
            out = out.data[0].cpu().numpy()
            out = np.round(out * 255.0)
            out = out.astype('uint8')
            out = out.transpose(1, 2, 0)
            img = Image.fromarray(out[:H, :W, :])
            img.save("./attack/kodak/%s_target_out.png"%(filename))

        mask = torch.zeros(1,C,H_PAD,W_PAD)
        if args.mask_loc != None:
            mask = torch.ones(1,C,H_PAD,W_PAD)
            lamb_tar = args.lamb_tar
            print(args.mask_loc, "lamb_bkg:", args.lamb_bkg_in, args.lamb_bkg_out, "lamb_tar:", lamb_tar)
            x0,x1,y0,y1 = args.mask_loc # W, H
            mask[:,:, y0:y1, x0:x1] = torch.zeros(1,C,y1-y0,x1-x0) #(y0, y1), (x0, x1)
        mask_bkg = mask.to(dev_id)
        mask_tar = 1. - mask_bkg

    mssim_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)
    lpips_func = lpips.LPIPS(net='alex').to(dev_id)
    with torch.no_grad():
        # mask = gnet(im_s)
        mask = torch.ones_like(im_s)

        output_s, y_main_, y_hyper, p_main, p_hyper = image_comp(im_s, False, CONTEXT, POSTPROCESS)
        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        # LOSS_FUNC = "L2"
        LOSS_FUNC = args.att_metric

        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        noise = torch.zeros(im_s.size())

        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=args.lr_attack)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
        noise_range = 0.5
        print("[WARNINTG] Quantization Skipped!")
        for i in range(args.steps):

            noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)

            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

            im_in[:,:,H:,:] = 0.
            im_in[:,:,:,W:] = 0.

            y_main = image_comp.net.g_a(im_in)
            x_ = image_comp.net.g_s(y_main)
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
            output_[:,:,H:,:] = 0.
            output_[:,:,:,W:] = 0.
            
            if LOSS_FUNC == "L2" and args.mask_loc == None:
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))               
                if args.target == None:
                    loss_o = 1. - torch.mean((im_s - output_) * (im_s - output_)) # MSE(x_, y_)
                else:
                    loss_o = torch.mean((output_t - output_) * (output_t - output_)) # MSE(y_t, y_s)

            # L1 loss
            if LOSS_FUNC == "L1":
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))
                if args.target == None:
                    loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
                else:    
                    loss_o = torch.mean(torch.abs(output_t - output_))

            if LOSS_FUNC == "ms-ssim":

                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))
                loss_o = mssim_func(im_s, output_)
            
            if LOSS_FUNC == "lpips":

                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))
                loss_o = - lpips_func(im_s, output_)

            if LOSS_FUNC == "L2" and args.mask_loc != None:

                loss_tar = torch.mean((im_s - im_in) * (im_s - im_in) * mask_tar)

                loss_bkg = torch.mean((im_s - im_in) * (im_s - im_in) * mask_bkg)
                loss_i = args.lamb_bkg_in * loss_bkg + loss_tar 

                loss_o_tar = torch.mean((output_t - output_) * (output_t - output_) * mask_tar)
                loss_o_bkg = torch.mean((output_t - output_) * (output_t - output_) * mask_bkg)

                loss_o = args.lamb_bkg_out * loss_bkg + loss_o_tar

                if loss_tar >= args.noise:
                    loss = loss_i
                else:
                    loss = loss_o

            else:                
                # loss = loss_i + lamb * loss_o
                if loss_i >= args.noise:
                    loss = loss_i
                else:
                    loss = loss_o

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log:
                writer.add_scalar('Loss/mse_all', loss.item(), i)
                writer.add_scalar('Loss/mse_in', mse_i.item(), i)
                writer.add_scalar('Loss/mse_out',  mse_o.item(), i)
                writer.add_scalar('Loss/bpp',  bpp.item(), i)
            
            if i%100 == 0:
                if args.mask_loc != None: 
                    print(i, "loss_rec(ALL/TAR/BKG):", loss_o.item(), loss_o_tar.item(), loss_o_bkg.item(), "loss_in(ALL/TAR/BKG):", loss_i.item(), loss_tar.item(), loss_bkg.item())
                else:
                    print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item())
                    
            if i % (args.steps//3) == 0:
                # print(torch.mean(mask), torch.mean(att))
                print("step:", i, "loss_rec:", loss_o.item(), "loss_in:", loss_i.item())
                lr_scheduler.step()
                with torch.no_grad():
                    im_uint8 = torch.round(im_in * 255.0)/255.0
                    
                    # 1. NO PADDING
                    # im_uint8[:,:,H:,:] = 0.
                    # im_uint8[:,:,:,W:] = 0.
                    
                    # save adverserial input
                    im_ =  torch.clamp(im_uint8, min=0., max=1.0)
                    fin = im_.data[0].cpu().numpy()
                    fin = np.round(fin * 255.0)
                    fin = fin.astype('uint8')
                    fin = fin.transpose(1, 2, 0)
                    img = Image.fromarray(fin[:H, :W, :])
                    # img = Image.fromarray(fin[PADDING:H+PADDING, PADDING:W+PADDING, :])
                    img.save("./attack/kodak/%s_fake%d_in_%0.8f.png"%(filename, i, loss.item())) 

                    output, y_main, y_hyper, p_main, p_hyper = image_comp(im_, False, CONTEXT, POSTPROCESS)   
                    
                    bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                    bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                    bpp = bpp_main + bpp_hyper
                    print("bpp:", bpp.item())
                    output_ = torch.clamp(output, min=0., max=1.0)
                    # print("loss:", torch.mean((output_ - output_t)**2*mask_tar))
                    out = output_.data[0].cpu().numpy()
                    out = np.round(out * 255.0)
                    out = out.astype('uint8')
                    out = out.transpose(1, 2, 0)
                    
                    # img = Image.fromarray(out[PADDING:H+PADDING, PADDING:W+PADDING, :])
                    img = Image.fromarray(out[:H, :W, :])
                    img.save("./attack/kodak/%s_fake%d_out_%0.4f_%0.8f.png"%(filename, i, bpp.item(), loss.item()))
    if args.log:
        writer.close()

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