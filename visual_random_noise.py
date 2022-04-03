import os
import sys
import argparse
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from thop import profile

import model
import coder
from anchors import balle
from utils import torch_msssim, ops


def save_tensor_to_image(im, H, W, save_dir):
    im = im.data[0].cpu().numpy()
    im = np.round(im * 255.0)
    im = im.astype('uint8').transpose(1, 2, 0)
    im = Image.fromarray(im[:H, :W, :])
    im.save(save_dir)
    return im

def test(args, model, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):
    image_comp = model
    dev_id = args.gpu
    TRAINING = False
    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0
    # print('====> Encoding Image:', im_dir)

    ## model initalization
    mssim_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)

    img = Image.open(args.source)
    img = np.array(img)/255.0
    if len(img.shape) < 3:
        H, W = img.shape
        im = np.tile(img.reshape((H,W,1)), (1,1,3))
    else:
        H, W, _ = img.shape

    num_pixels = H * W
    H_PAD = int(tile * np.ceil(H / tile))
    W_PAD = int(tile * np.ceil(W / tile))
    PADDING = 0
    
    im = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im[PADDING:H+PADDING, PADDING:W+PADDING, :] = img[:, :, :3]
    im = torch.FloatTensor(im)
    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD).to(dev_id)

    # TODO: add_noise here
    m = torch.distributions.normal.Normal(0.0, 0.03162277660168379)
    noise = m.sample(im.size()).cuda()
    print(torch.mean(noise ** 2), im.size(),noise.size())
    im_in = ops.Up_bound.apply(ops.Low_bound.apply(im+noise, 0.), 1.)

    # TODO: save input image
    filename = args.source.split("/")[-1][:-4]
    save_tensor_to_image(im_in, H, W, "./attack/random_noise_1e-3/"+filename+".png")

    with torch.no_grad():
        # original_image

        output, y_main_, y_hyper, p_main, p_hyper = image_comp(im_in, False, CONTEXT, POSTPROCESS)
        # with open('test_modified.npy', 'rb') as f:
            # y_hat = np.load(f)
        # x_hat = image_comp.net.g_s(torch.Tensor(y_hat).cuda())
        # output = torch.clamp(x_hat, min=0., max=1.0)
        with open('y_hat.npy', 'wb') as f:
            np.save(f, torch.round(y_main_).cpu().numpy())
        
        # output = torch.clamp(output, min=0., max=1.)
        # out = output.data[0].cpu().numpy()
        # out = np.round(out * 255.0)
        # out = out.astype('uint8')
        # out = out.transpose(1, 2, 0)

        # img = Image.fromarray(out[:H, :W, :])
        # img.save("./attack/kodak/modified_out.png")

        bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        
        output = torch.clamp(output, min=0., max=1.0)

        # PSNR
        mse = torch.mean((im - output)**2)
        psnr = -10. * np.log10(mse.item())

        out = output.data[0].cpu().numpy()
        out = np.round(out * 255.0)
        out = out.astype('uint8')
        out = out.transpose(1, 2, 0)

        img = Image.fromarray(out[:H, :W, :])
        img.save(args.target)
        # img.save("./attack/kodak/out.png")

    return bpp_hyper + bpp_main, psnr, mssim_func(output, im)


if __name__ == "__main__":
    args = coder.config()
    print("============================================================")
    print("[ IMAGE ]:", args.source, "->", args.target)
    
    model = coder.load_model(args)
    
    images = glob(args.source)
    print(images)
    for image in images:
        args.source = image
        bpp, psnr, quality = test(args, model, None, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
        print("bpp:", bpp.item(), "PSNR:", psnr, "MS-SSIM:", quality.item())