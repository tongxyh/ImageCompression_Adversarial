import os
from pydoc import doc
import sys
import math
import argparse
from glob import glob
import time
from datetime import datetime

import lpips
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import coder
from anchors import balle
from utils import torch_msssim, ops
from anchors import model as models


def clamp_value_naive(args, y_main):
    if args.adv:
        profile = f"{args.model}-{args.metric}-{args.quality}-adv"
    else:
        profile = f"{args.model}-{args.metric}-{args.quality}"
    channel_max, channel_min = torch.load(f"./attack/data/{profile}_range.pt")
    # print(y_main.shape, channel_max.shape)
    channel_max = channel_max.view(1,-1,1,1)
    channel_min = channel_min.view(1,-1,1,1)
    y_main = torch.where(y_main > channel_max, channel_max, y_main)
    y_main = torch.where(y_main < channel_min, channel_min, y_main)
    return y_main

def plot_bar(ax, ax_x, data, labels, save_path, stack=False):
    loc_interval = 0.5/len(data)
    locs = [-0.5 + loc_interval * (2*i+1) for i in range(len(data))]
    width = loc_interval * 2
    colors = ["g", "r", "b"]
    if stack:
        locs = [0 for i in range(len(data))]
        width = 0.8    
    for i in range(len(data)):
        if i > 0 and stack:
            # ax.bar(ax_x+locs[i], data[i].detach().cpu().numpy(), width=width, bottom = data[i-1].detach().cpu().numpy(), color=colors[i%3], label=labels[i])
            ax.plot(ax_x, data[i], label=labels[i])
        else:
            ax.plot(ax_x, data[i], label=labels[i])
            # ax.bar(ax_x+locs[i], data[i].detach().cpu().numpy(), width=width, color=colors[i%3], label=labels[i])
    ax.legend(prop={'size': 14})
    # ax.yticks(list(range(0, int(data[0].max()), 1))) 
    ax.grid(linewidth=0.1, linestyle="--")
    plt.ylim(ymin=0, ymax=40)
    plt.tight_layout()
    plt.savefig(f"./logs/{save_path}")

def show_max_bar(data, labels, save_path, sort=True, stack=False, vi=None):
    fig, ax = plt.subplots()
    # maxs = [torch.amax(torch.abs(i), dim=(0,2,3)) for i in data]
    maxs = [torch.amax(i, dim=(0,2,3)) for i in data]
    mins = [torch.amin(i, dim=(0,2,3)) for i in data]

    ax_x = np.arange(0, maxs[0].shape[0], 1)

    # fill between
    if args.adv:
        profile = f"{args.model}-{args.metric}-{args.quality}-adv"
    else:
        profile = f"{args.model}-{args.metric}-{args.quality}"
    channel_max, channel_min = torch.load(f"./attack/data/{profile}_range.pt")

    if sort:
        # reorder max_1 with the order of sorted max_0
        _, indices = maxs[0].sort(descending=True)
        maxs = [ i[indices] for i in maxs]
        mins = [ i[indices] for i in mins]
        
        channel_max, channel_min = channel_max[indices], channel_min[indices]
        # ax.fill_between(ax_x, channel_min.cpu(), channel_max.cpu(), alpha=.5, linewidth=0)
        ax.fill_between(ax_x, channel_min.cpu(), channel_max.cpu(), alpha=.25, linewidth=0.5, label="safe zone")
    # for max_v, min_v, label_v in zip(maxs, mins, labels):
    #     ax.fill_between(ax_x, min_v.detach().cpu(), max_v.detach().cpu(), alpha=.5, linewidth=0, label=label_v)
    # ax.fill_between(ax_x, mins[0].detach().cpu(), maxs[0].detach().cpu(), alpha=0.4, linewidth=1, label=labels[0], color='g')
    ax.plot(ax_x, mins[0].detach().cpu(), linewidth=1, label=labels[0], color='g')
    ax.plot(ax_x, maxs[0].detach().cpu(), linewidth=1, color='g')

    # ax.fill_between(ax_x, mins[1].detach().cpu(), maxs[1].detach().cpu(), alpha=0.4, linewidth=1, label=labels[1], color='r')
    ax.plot(ax_x, mins[1].detach().cpu(), linewidth=1, label=labels[1], color='r')
    ax.plot(ax_x, maxs[1].detach().cpu(), linewidth=1, color='r')    
    
    ax.legend(prop={'size': 14})
    ax.grid(linewidth=0.1, linestyle="--")
    ax.text(0.95, 0.05, f"$\Delta$PSNR={vi:0.2f}", fontsize=20, ha='right', va='bottom', transform=ax.transAxes)

    ax.set_xlabel("channel index", fontsize=14)
    ax.set_ylabel("activation magnitude", fontsize=14)
    plt.ylim(ymin=-25, ymax=25)
    plt.tight_layout()
    plt.savefig(f"./logs/{save_path}")

def clip_dead_channel(y_main, profile):
    dead_channels = torch.load(profile)["dead"]
    ranks_max, ranks_min = torch.load(profile)["rank"]
    y_main_ = torch.zeros_like(y_main)
    
    rank = torch.argsort(torch.amax(torch.abs(y_main), dim=(2,3)), dim=1, descending=True) # sort by 
    rank_ = torch.zeros_like(rank)
    for j in range(rank.shape[1]):
        # print(j,rank[0,j])
        rank_[0, rank[0,j]] = j

    index_abs_max = torch.amax(torch.abs(y_main), dim=(2,3), keepdim=True)
    for index in range(y_main_.shape[1]):
        if index in dead_channels:
            y_main_[:,index,:,:] = torch.clamp(y_main[:,index,:,:], min=-1.5, max=1.5)
        elif rank_[0, index] < ranks_min[index]-100:
            if args.debug:
                print(index, rank_[0, index], ranks_min[index], ranks_max[index])   
            y_main_[:,index,:,:] = torch.clamp(y_main[:,index,:,:], min=-index_abs_max[0,ranks_min[index],0,0].item(), max=index_abs_max[0,ranks_min[index],0,0].item())
        else:
            y_main_[:,index,:,:] = y_main[:,index,:,:]
    return y_main_

def defend(y_main, args):
    return clamp_value_naive(args, y_main)

def crop(x, padding):
    return x[:,:,padding:-padding,padding:-padding]

@torch.no_grad()
def eval(im_adv, im_s, output_s, net, args):
    net.eval()
    # im_uint8 = torch.round(im_adv * 255.0)/255.0
    # im_ =  torch.clamp(im_uint8, min=0., max=1.0)
    im_ = torch.clamp(im_adv, min=0., max=1.0)

    # save adverserial input
    # coder.write_image(im_, "%s_advin_%d_%0.8f.png"%(filename, i, loss.item()), H, W)
    if args.pad:
        result = net(F.pad(im_, (args.pad, args.pad, args.pad, args.pad), mode=args.padding_mode))
    else:
        result = net(im_)
        
        # y_main = net.g_a(im_)  
        # x_ = net.g_s(y_main)
        # result["x_hat"] = x_

    x_hat = result["x_hat"]
    if args.pad:
        # x_hat = net.g_s(torch.nn.functional.pad(crop(result["y_hat"], padding_y), (padding_y, padding_y, padding_y, padding_y), mode=args.padding_mode))    
        x_hat = crop(x_hat, args.pad)
    
    mse_in = torch.mean((im_ - im_s)**2)
    if args.defend:
        y_main = net.g_a(im_)  
        y_main = defend(y_main, args)
        _, _, result["likelihoods"] = models.entropy_estimator(y_main, net, args.model)
        x_ = net.g_s(torch.round(y_main))
        output_ = torch.clamp(x_, min=0., max=1.0)
    else:
        if args.clamp:
            output_ = torch.clamp(x_hat, min=0., max=1.0)
        else:
            output_ = x_hat

    num_pixels = (im_adv.shape[2]) * (im_adv.shape[3])
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())
    # recalculate bpp
    # bpp = torch.log(result["likelihoods"]["y"]).sum() / (-math.log(2) * num_pixels)
    # bpp +=torch.log(result["likelihoods"]["z"]).sum() / (-math.log(2) * num_pixels)

    mse_out = torch.mean((output_ - output_s)**2)
    if mse_in > 1e-20 and mse_out > 1e-20:
        vi = 10. * math.log10(mse_out/mse_in)
    else:
        vi = None
        print(f"[!] Warning: mse_in ({mse_in}) or mse_out {mse_out} is zero")
    if args.debug:
        print("loss_rec:", mse_out.item(), "loss_in:", mse_in.item(), "VI (mse):", vi)
    # MS-SSIM
    return im_, output_, bpp, mse_in, mse_out, vi

def batch_test(args):    
    images = sorted(glob(args.source))
    for i, image in enumerate(images):
        # evaluate time of each attack
        start = time.time()
        eval()
        end = time.time()
        print(image, "Time:", end-start)

def main(args):
    if args.quality > 0:
        batch_test(args)
    else:
        q_max = 7 if args.model == "cheng2020" else 9
        for q in range(1, q_max):
            args.quality = q
            batch_test(args)

if __name__ == "__main__":
    args = coder.config()
    args = args.parse_args()
    main(args)
    