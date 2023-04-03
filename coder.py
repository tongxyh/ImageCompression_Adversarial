import os
import sys
import argparse
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from thop import profile

from utils import torch_msssim, ops
from anchors import model as models
from anchors import balle
os.environ["TORCH_HOME"] = "./ckpts/torch/"


def read_image(filename, padding=64):
    img = Image.open(filename)
    img = np.array(img)/255.0
    C = 3
    if len(img.shape) < 3:
        print("[WARNING] automatically convert gray image to rgb format!")
        H, W = img.shape
        img = np.tile(img.reshape((H,W,1)), (1,1,3))
    else:
        H, W, _ = img.shape

    H_PAD = int(padding * np.ceil(H / padding))
    W_PAD = int(padding * np.ceil(W / padding))
    
    im = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im[:H,:W,:3] = img
    im = torch.FloatTensor(im)
    im = im.permute(2, 0, 1).contiguous()
    im = im.view(1, C, H_PAD, W_PAD)
    return im, H, W

def write_image(x, filename, H=None, W=None):
    if H == None and W == None:
        H, W = x.shape[2:]
    x = np.round(x.data[0].cpu().numpy() * 255.0)
    x = x.astype('uint8').transpose(1, 2, 0)
    img = Image.fromarray(x[:H, :W, :])
    img.save(filename)

def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    # if args.adv:
    #     parameters = {n for n, p in net.named_parameters() if ('g_s' in n or 'g_a' in n) and p.requires_grad}
    #     print(parameters)
    # else:
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }

    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    if not args.adv:
        assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.lr_train,
    )
    aux_optimizer = torch.optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer

def load_model(args, training):
    MODEL = args.model
    quality = args.quality
    arch_lists = ["factorized", "hyper", "context", "cheng2020", "debug"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"
    print("==================== NETWORK SETTINGS ===================")
    print("[ARCH]", MODEL, quality, args.metric)
    download = False
    if not args.checkpoint and not args.new:
        print("[CKPT] Download from CompressAI Model Zoo!")
        download = True
    elif not args.checkpoint:
        print("[CKPT] No Checkpoint Loaded!!!")
    net = models.init_model(MODEL, quality=quality, metric=args.metric, pretrained=download).to(args.device)
    # for param_tensor in net.state_dict():
    #     print(param_tensor,'\t',net.state_dict()[param_tensor].size())
    if args.checkpoint: # load from local ckpts
        print("[CKPT] Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if checkpoint.get("state_dict"):
            net.load_state_dict(checkpoint["state_dict"])
        else:
            # old version ckpts
            net_old = balle.Image_coder(MODEL=args.model, quality=args.quality, metric=args.metric, pretrained=False)
            net_old.load_state_dict(checkpoint)
            # save ckpts
            torch.save({"state_dict": net_old.net.state_dict()}, args.checkpoint+"new")
            checkpoint = torch.load(args.checkpoint+"new", map_location=args.device)
            net.load_state_dict(checkpoint["state_dict"])
    # for param_tensor in checkpoint["state_dict"]:
        # print(param_tensor,'\t',checkpoint["state_dict"][param_tensor].size())

    print("=========================================================")
    if training:
        # optimizer
        last_epoch = 0
        optimizer, aux_optimizer = configure_optimizers(net, args)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_iters, gamma=decay_gamma, last_epoch=-1)       
        if args.checkpoint: # load from local ckpts
            if checkpoint.get("state_dict"):
                last_epoch = checkpoint["epoch"] + 1
                print("resume training from epoch", last_epoch)
                optimizer.load_state_dict(checkpoint["optimizer"])
                # for param_tensor in checkpoint["optimizer"]["state"]:
                #     print(param_tensor, checkpoint["optimizer"]["state"][param_tensor]["exp_avg"].size())
                aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        return net.train(), last_epoch, optimizer, aux_optimizer, lr_scheduler
    else:
        if args.checkpoint: # load from local ckpts
            if args.eval:
                last_epoch = checkpoint["epoch"]
                print("Trained epoch", last_epoch)
                if checkpoint["step"]:
                    print("Trained step", checkpoint["step"])
                optimizer, aux_optimizer = configure_optimizers(net, args)
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        return net.eval()


def eval_result():
    pass

@torch.no_grad()
def code(args, net, input_file, out_file=None):
    net.eval()
    im, _, _ = read_image(input_file)
    im = im.to(args.device)

    result = net(im)

    if out_file:
        write_image(torch.clamp(result["x_hat"], min=0.0, max=1.0), out_file)

    return result

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=str, default="cuda:0", help="dev id")
    # Train config
    parser.add_argument("-lr_train",  dest="lr_train",   type=float, default=0.0001,  help="train learning rate")
    parser.add_argument("-lamb",  dest="lamb",   type=float, default=None,  help="training lambda")
    parser.add_argument("--eval", dest="eval", action="store_true", help="evaluation mode")
    parser.add_argument('--adv', action='store_true', help='Adversarial training')
    parser.add_argument('-batch_size', type=int, default=8, help='Batch size')
    # NIC config
    parser.add_argument("-cn", "--ckpt_num", type=int, help="load checkpoint by step number")
    parser.add_argument("-l",  "--lamb",     type=float, default=6400., help="lambda")
    parser.add_argument("-j",  "--job",      type=str,   default="", help="job name")
    parser.add_argument('--ctx',    dest='context', action='store_true')
    parser.add_argument('--no-ctx', dest='context', action='store_false')
    parser.add_argument('--post', dest='post', action='store_true')

    parser.add_argument('-itx',    dest='iter_x', type=int, default=0,          help="iter step updating x")
    parser.add_argument('-ity',    dest='iter_y', type=int, default=0,          help="iter step updating y")
    parser.add_argument('-m',      dest='model',  type=str, default="hyper", help="compress model in 'factor','hyper','context','cheng2020','nonlocal'")
    parser.add_argument('-metric', dest='metric', type=str, default="ms-ssim",  help="mse or ms-ssim")
    parser.add_argument('-q',      dest='quality',type=int, default="3",        help="quality in [1-8]")
    parser.add_argument('--new',   dest='new', action='store_true', help='train new model')
    parser.add_argument('-padmode',dest='padding_mode', type=str, default="reflect", help="pad mode")
    # attack config
    parser.add_argument('-steps',dest='steps',      type=int,   default=1001,  help="attack iteration steps")
    parser.add_argument('-random',dest='random',    type=int,   default=1,  help="random start numbers")
    parser.add_argument("-la",  dest="lamb_attack", type=float, default=0.2,    help="attack lambda")
    parser.add_argument("-noise",dest="noise",      type=float, default=0.0001,  help="input noise threshold")
    
    parser.add_argument("-lr_attack",  dest="lr_attack",   type=float, default=0.01,  help="attack learning rate")
    parser.add_argument("-s",   dest="source",      type=str,   default="/workspace/ct/datasets/kodak/kodim*.png",   help="source input image")
    parser.add_argument("-t",   dest="target",      type=str,   default=None,   help="target image")
    parser.add_argument("-ckpt",dest="checkpoint",        type=str,   default=None,   help="local checkpoint dir")
    parser.add_argument('--mask_loc', nargs='+', type=int, default=None)
    parser.add_argument("-la_bkg_in",  dest="lamb_bkg_in", type=float, default=1.0,    help="attack lambda of background area of input")
    parser.add_argument("-la_bkg_out", dest="lamb_bkg_out",type=float, default=1.0,    help="attack lambda of background area of output")    
    parser.add_argument("-la_tar",  dest="lamb_tar",type=float, default=1.0,    help="attack lambda of target area")   
    parser.add_argument('-att_metric', dest='att_metric', type=str, default="L2",  help="L1, L2, ms-ssim or lpips") 
    parser.add_argument("-e", dest="epsilon", type=float, default=16.0, help="noise max value epsilon")
    parser.add_argument("-r", dest="rate", action="store_true", help="rate/distortion attack flag")
    parser.add_argument("-p", dest="pad", type=int, default=None, help="padding size")
    parser.add_argument('--log',  dest='log', type=str, default="./logs/log.txt", help="log file")
    parser.add_argument('--debug',dest='debug', action='store_true')
    parser.add_argument('--no-clamp',dest='clamp', action='store_false')

    parser.add_argument("-ssteps", dest="search_steps", type=int, default=20, help="binary search steps for CW")
    parser.add_argument("-re", dest="recompress", type=int, default=None, help="recompress times")
    
    parser.add_argument("--defend", action="store_true", help="defend mode")
    parser.add_argument("--defend_m", dest="method", type=str, default="ensemble", help="defend method in ['ensemble', 'resize']")
    parser.add_argument("-degrade", dest="degrade",  type=str, default=None, help="degrade method in ['deblur']")
    parser.add_argument("--fintune", action="store_true")

    return parser