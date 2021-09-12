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


def test(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):

    TRAINING = False
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
    download_model_zoo = args.download
    arch_lists = ["factorized", "hyper", "context", "cheng2020", "nlaic", "elic"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"
    if MODEL == "elic":
        image_comp = model.ImageCompression(256)
        image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
        # image_comp.load_state_dict(torch.load(checkpoint_dir).state_dict())
        # torch.save(image_comp.state_dict(), "./checkpoints/elic-0.0.1/ae.pkl")
        image_comp.to(dev_id).eval()
        # print("[ ARCH  ]:", MODEL) 

    if MODEL in ["factorized", "hyper", "context", "cheng2020"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric, pretrained=True).to(dev_id)
        # print("[ ARCH  ]:", MODEL, quality, args.metric)
        if download_model_zoo == False:
            # load from local ckpts
            image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
            image_comp.to(dev_id).eval()
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

    with torch.no_grad():
        # original_image
        output, y_main_, y_hyper, p_main, p_hyper = image_comp(im, False, CONTEXT, POSTPROCESS)

        bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        output = torch.clamp(output, min=0., max=1.0)

        # PSNR
        mse = torch.mean((im - output)**2)
        psnr = -10. * np.log10(mse.item())

    return bpp_hyper + bpp_main, psnr, mssim_func(output, im), y_main_, output

def config():
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
    parser.add_argument('-m',      dest='model',  type=str, default="nonlocal", help="compress model in 'factor','hyper','context','nonlocal'")
    parser.add_argument('-metric', dest='metric', type=str, default="ms-ssim",  help="mse or ms-ssim")
    parser.add_argument('-q',      dest='quality',type=int, default="2",        help="quality in [1-8]")
    
    # attack config
    parser.add_argument('-steps',dest='steps',      type=int,   default=10001,  help="attack iteration steps")
    parser.add_argument("-la",  dest="lamb_attack", type=float, default=0.2,    help="attack lambda")
    parser.add_argument("-lr",  dest="lr_attack",   type=float, default=0.001,  help="attack learning rate")
    parser.add_argument("-s",   dest="source",      type=str,   default=None,   help="source input image")
    parser.add_argument("-t",   dest="target",      type=str,   default=None,   help="target image")
    parser.add_argument("-ckpt",dest="ckpt",        type=str,   default=None,   help="local checkpoint dir")
    parser.add_argument('--download',  dest='download',    action='store_true')
    parser.add_argument('--mask_loc', nargs='+', type=int, default=None)
    parser.add_argument("-la_bkg",  dest="lamb_bkg",type=float, default=1.0,    help="attack lambda of background area")
    parser.add_argument("-la_tar",  dest="lamb_tar",type=float, default=1.0,    help="attack lambda of target area")    

    # train config
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')

    parser.add_argument('--log',dest='log', action='store_true')
    return parser.parse_args()