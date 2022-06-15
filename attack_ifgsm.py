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
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import coder
from anchors import balle
from utils import torch_msssim, ops
from anchors import model as models
from anchors.utils import layer_compare


def entropy(net, y_main, model="hyper"):
    z = net.h_a(y_main)
    z_hat, z_likelihoods = net.entropy_bottleneck(z)
    params = net.h_s(z_hat)
    if model == "hyper":
        scales_hat = params
        means_hat = 0.
    if model == "context":
        ctx_params = net.context_prediction(y_main)
        gaussian_params = net.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
    # print(scales_hat.shape, means_hat.shape, y_main.shape)
    return means_hat, scales_hat

def clamp_feature_with_p(x, y_main, means_hat, scales_hat, epsilon=50):
    scales_hat = torch.clamp(scales_hat, min=0.11)
    pred_error = (y_main-means_hat) / scales_hat
    print(torch.max(y_main), torch.max(pred_error), torch.max(scales_hat))
    pred_error = torch.where(pred_error < epsilon, pred_error, torch.tensor(epsilon, dtype=pred_error.dtype, device=pred_error.device))
    pred_error = torch.where(pred_error > - epsilon, pred_error, torch.tensor(epsilon, dtype=pred_error.dtype, device=pred_error.device))
    # print(pred_error[0,0,:,0])
    return pred_error * (scales_hat) + means_hat

def clamp_value_naive(args, y_main):
    # index_max = torch.amax(y_main, dim=(2,3), keepdim=True)
    # index_min = torch.amin(y_main, dim=(2,3), keepdim=True)
    thres = 1.5
    index_max, index_min = torch.load(f"{args.model}-{args.metric}-{args.quality}.pt")
    y_main_ = torch.where(y_main > index_max, index_max, y_main)
    y_main_ = torch.where(y_main_ < index_min, index_min, y_main_)
    y_main = torch.where(index_max < thres, y_main_, y_main)
    y_main = torch.where(index_min > -thres, y_main_, y_main)

    # y_main = torch.where(y_main > index_max, index_max, y_main)
    # y_main = torch.where(y_main < index_min, index_min, y_main)
    return y_main

def anti_noise(x, scale=0.5):
    # #up/down sample
    print("resize bicubic")
    x_down = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True)
    # x_down = F.interpolate(x, scale_factor=scale, mode="nearest")
    x_up = F.interpolate(x_down, scale_factor=1/scale, mode="bicubic", align_corners=False, antialias=True)
    x_res = x - x_up

    coder.write_image(x_res+0.5, "./logs/resize_nearst_high.png", H=x.shape[2], W=x.shape[3])
    coder.write_image(x_up, "./logs/resize_nearst_low.png", H=x.shape[2], W=x.shape[3])
    return x_up
    #  return F.interpolate(x_down, scale_factor=1/scale, mode="bicubic", align_corners=False, antialias=True)

def self_ensemble(net, x):
    x0 = torch.flip(x, [2])
    x1 = torch.flip(x, [3])
    x2 = torch.flip(x0, [3])

    x3 = torch.rot90(x, 1, [2,3])
    x4 = torch.flip(x3, [2])
    x5 = torch.flip(x3, [3])
    x6 = torch.flip(x4, [3])

    x_set0 = torch.cat((x,x0,x1,x2), dim=0)
    x_set1 = torch.cat((x3,x4,x5,x6), dim=0) 
    result = net(x_set0)
    best_mse = 1
    for x, x_hat in zip(x_set0, result[x_hat]):
        mse = torch.mean((x - x_hat)**2)
        if mse < best_mse:
            best_x, best_x_hat = x, x_hat
            best_mse = mse
    result = net(x_set1)
    for x, x_hat in zip(x_set0, result[x_hat]):
        mse = torch.mean((x - x_hat)**2)
        if mse < best_mse:
            best_x, best_x_hat = x, x_hat
            best_mse = mse    
    return best_mse, best_x, best_x_hat

def clamp_with_prior(x, net, y_main, save_path=None):
    x_ = anti_noise(x, scale=0.8)
    y_main_ = net.g_a(x_)
    index_max = torch.amax(y_main_, dim=(2,3), keepdim=True)
    index_min = torch.amin(y_main_, dim=(2,3), keepdim=True)
    
    index_max_ = torch.amax(y_main, dim=(2,3), keepdim=True)
    index_min_ = torch.amin(y_main, dim=(2,3), keepdim=True)

    y_main = torch.where(y_main > index_max, index_max, y_main)
    y_main = torch.where(y_main < index_min, index_min, y_main)
    if save_path:
        # y_main = torch.where(torch.abs(y_main_) > torch.abs(y_main), y_main, y_main_)
        ax_x = np.arange(0, y_main.shape[1], 1)
        fig, ax = plt.subplots()
        v_input = []
        v_resize = []
        ax.bar(ax_x-0.25, index_max[0,:,0,0].detach().cpu().numpy(), width=0.5, color="r", label="origin")
        ax.bar(ax_x+0.25, index_max_[0,:,0,0].detach().cpu().numpy(), width=0.5, color="b", label="adversarial")
        ax.legend()
        v_resize.append(index_max)
        v_input.append(index_max_)
        plt.savefig(f"./logs/{save_path}")
        return y_main, v_input, v_resize
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
            ax.bar(ax_x+locs[i], data[i].detach().cpu().numpy(), width=width, bottom = data[i-1].detach().cpu().numpy(), color=colors[i%3], label=labels[i])
        else:
            ax.bar(ax_x+locs[i], data[i].detach().cpu().numpy(), width=width, color=colors[i%3], label=labels[i])
    ax.legend(prop={'size': 14})
    # ax.yticks(list(range(0, int(data[0].max()), 1))) 
    ax.grid(linewidth=0.1, linestyle="--")
    plt.ylim(ymax=10)
    plt.tight_layout()
    plt.savefig(f"./logs/{save_path}")

def show_max_bar(data, labels, save_path, sort=True, stack=False):
    fig, ax = plt.subplots()
    maxs = [torch.amax(torch.abs(i), dim=(0,2,3)) for i in data]
    # maxs = [torch.mean(torch.clamp(i, min=0), dim=(0,2,3)) for i in data]
    # maxs[0] = torch.mean(torch.clamp(data[0], min=0), dim=(0,2,3))
    # maxs[1] = torch.mean(torch.clamp(data[1], min=0), dim=(0,2,3))
    if sort:
        # reorder max_1 with the order of sorted max_0
        _, indices = maxs[0].sort(descending=True)
        maxs = [ i[indices] for i in maxs]
    torch.save(maxs, f"./logs/max_0.pt")
    ax_x = np.arange(0, maxs[0].shape[0], 1)
    # ax.plot(ax_x, maxs[0].detach().cpu().numpy(), label=labels[0], linewidth=0.5)
    plot_bar(ax, ax_x, maxs, labels, save_path, stack)

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
    if args.adv:
        profile = f"{args.model}-{args.metric}-{args.quality}-adv.pt"
    else:
        profile = f"{args.model}-{args.metric}-{args.quality}.pt"
    # means_hat, scales_hat = entropy(self.net, y_main, model=args.model)
    # y_main = clamp_with_prior(im_in, self.net, y_main)
    # y_main, v_i, v_r = clamp_with_prior(im_, self.net, y_main, "final.pdf")

    # y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

    # for likelihoods in result["likelihoods"].values():
    #     if likelihoods.shape == y_main.shape:
    #         y_main = torch.where(likelihoods > 0.01, y_main, torch.tensor(0., dtype=y_main.dtype, device=y_main.device))
    # print(sorted(torch.sum(torch.round(y_main), (0,2,3), dtype=int).cpu().numpy()))

    # y_main = torch.where(y_main > index_max, index_max, y_main)
    # y_main = torch.where(y_main < index_min, index_min, y_main)
    
    # y_main = torch.where(index_c, y_main, torch.tensor(0., dtype=y_main.dtype, device=y_main.device))
    return clip_dead_channel(y_main, profile)

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

        # result["likelihoods"]["y"] = crop(result["likelihoods"]["y"], padding_y)
        # TODO: pad z_hat ?
        # padding_z = args.pad//64
        # result["likelihoods"]["z"] = crop(result["likelihoods"]["z"], padding_z)
    
    mse_in = torch.mean((im_ - im_s)**2)
    if args.defend:
        y_main = net.g_a(im_)  
        y_main = defend(y_main, args)
        x_ = net.g_s(torch.round(y_main))
        output_ = torch.clamp(x_, min=0., max=1.0)
    else:
        if args.clamp:
            output_ = torch.clamp(x_hat, min=0., max=1.0)
        else:
            output_ = x_hat

    if args.debug:
        layer_compare(net, im_, im_s)
            # v_std = torch.mean((layer-(mean_after**0.5))**2)
            # index += 1

            # # if index in [2, 4, 6]:
            # #     norm = layer/temp[0]
            # #     norm_s = layer/temp[1]
            # #     print("GDN norm:", torch.max(norm), torch.max(norm_s), torch.mean(norm), torch.mean(norm_s))

            # temp = layer, layer_s
            # if self.args.model == "debug":
            #     pass
            # else:
            #     if index in [2, 4, 6]:
            #         # gamma_mean = layerout_s["gdn_gamma_mean"][3:][index//2]
            #         # beta_mean = layerout_s["gdn_beta_mean"][3:][index//2 ]
            #         # print("GDN factor:", math.sqrt((mean_after+v_std)*128*gamma_mean+beta_mean))
            #         beta = f.beta_reparam(f.beta)
            #         gamma = f.gamma_reparam(f.gamma)
            #         _, C, _, _ = temp[0].size()
            #         gamma = gamma.reshape(C, C, 1, 1)
            #         norm_ori = torch.sqrt(torch.nn.functional.conv2d(temp[1] ** 2, gamma, beta))
            #         norm_adv = torch.sqrt(torch.nn.functional.conv2d(temp[0] ** 2, gamma, beta))
            #         print("GDN norm:", torch.max(norm_ori), torch.max(norm_adv), torch.mean(norm_ori), torch.mean(norm_adv))
    
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

def attack_onestep(im_s, net):
    # TODO: FreeAT
    pass
    # im_adv = im_s + torch.clamp(im_s - im_s_hat, min=0., max=1.0)
    # return im_adv
def attack_cw(im_s, output_s, im_in, net, args):
    c = args.lamb_attack
    loss_i = torch.mean((im_s - im_in) ** 2)

    y_main = net.g_a(im_in)  
    x_ = net.g_s(y_main)
    if args.clamp: # default: False
        output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
    else:
        output_ = x_
    
    loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
    if loss_o < 1 - args.noise:
    # if loss_i > args.noise:
        c1 = 1
        c = 0.
    else:
        c1 = 0.
        c = 1.

    loss = c1 * loss_i + c * loss_o

    return loss, loss_i, loss_o

def attack_our(im_s, output_s, im_in, net, args):
    loss_i = torch.mean((im_s - im_in) ** 2)
    if loss_i > args.noise:
        loss = loss_i
        loss_o = torch.Tensor([0.])
    else:
        # if args.pad:
        #     y_main = net.g_a(F.pad(im_in, padder, mode=args.padding_mode))
        # else:
        y_main = net.g_a(im_in)
        if args.defend:
            y_main = defend(y_main, args)
        # if args.pad:
        #     y_main = torch.nn.functional.pad(y_main[:,:,padding_y:-padding_y,padding_y:-padding_y], padder_y, mode=args.padding_mode)    
        x_ = net.g_s(y_main)
        # if args.pad:
        #     x_ = crop(x_, padding)
        # x_ = self.net(im_in)["x_hat"]
        if args.clamp: # default: False
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
        else:
            output_ = x_

        # output_[:,:,H:,:] = 0.
        # output_[:,:,:,W:] = 0.

        # if LOSS_FUNC == "L2":
        loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
            # loss_o = 1. - (output_s - output_) ** 2
        # if LOSS_FUNC == "L1":
        #     loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
        
        # if im_s.shape[0] == 1:
        # x_over = (x_ - output_)**2

        # loss = torch.mean(torch.where(x_over>0.0001, x_over, loss_o))
        # loss_o = torch.mean(loss_o)
        loss = loss_o

        # if im_s.shape[0] > 1:
        #     loss = torch.where(loss_i_batch > args.noise, loss_i_batch, loss_o)
        #     loss = torch.mean(loss)
    return loss, loss_i, loss_o

def attack_ifgsm(im_s, net, args):
    H, W = im_s.shape[2], im_s.shape[3]
    num_pixels = H * W

    # generate original output
    with torch.no_grad():
        net.eval()
        result = net(im_s)
        output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
        bpp_ori = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())

    im_adv = im_s.detach().requires_grad_(True)
    net.train()
    eps = args.epsilon/255.0
    for i in range(args.steps):
        y_main = net.g_a(im_adv)
        output_ = net.g_s(y_main)
        loss_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)

        net.zero_grad()
        if im_adv.grad is not None:
            im_adv.grad.data.fill_(0)
        loss_o.backward()

        im_adv.grad.sign_()
        # grad_l = torch.sign(noise.grad+1e-8)
        # grad_r = torch.sign(noise.grad-1e-8)
        # noise.grad = 0.5 * (grad_l + grad_r)

        # print("noise:",torch.mean(noise.grad**2))
        # print("eps:", eps)
        im_adv = im_adv + eps/args.steps*im_adv.grad
        im_adv = torch.where(im_adv > im_s+eps, im_s+eps, im_adv)
        im_adv = torch.where(im_adv < im_s-eps, im_s-eps, im_adv)
        im_adv = im_adv.detach().requires_grad_(True)
        # im_adv = im_in + (eps/args.steps)*noise.grad
        # im_adv = torch.clamp(im_adv, 0., 1.)
        loss_i = torch.mean((im_adv - im_s)**2)
        if i%1 == 0 and args.debug:
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", 10*math.log10((1.-loss_o.item())/(loss_i.item()+1e-10)))
                
        if i%(args.steps//3) == 0:
            if args.debug:
                # print(im_s.shape, output_s.shape)
                _, _, bpp, mse_in, mse_out, vi = eval(im_adv, im_s, output_s, net, args)    
                net.train()

    im_adv, output_adv, bpp, mse_in, mse_out, vi = eval(im_adv, im_s, output_s, net, args)         
    return im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi

def attack_(im_s, net, args):
    # C = 3
    # im_s, H, W = coder.read_image(image_file)
    # im_s = im_s.to(self.args.device)
    # image_dir = "./attack/kodak/"
    # filename = image_dir + self.model_config + image_file.split("/")[-1][:-4]
    
    # pad im_s
    padding = 0
    if args.pad:
        padding = args.pad
        # padding_y = padding//16
        padder = (padding, padding, padding, padding)
        # padder_y = (padding_y, padding_y, padding_y, padding_y)

    # print("[*] Input size:", im_s.size())
    H, W = im_s.shape[2], im_s.shape[3]
    num_pixels = H * W

    # generate original output
    with torch.no_grad():
        net.eval()
        if args.pad:
            result = net(F.pad(im_s, padder, mode=args.padding_mode))
        else:
            result = net(im_s)
        if not args.pad:
            output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
        else:
            # result = models.compressor(im_s, net, args.model)
            # y_hat = torch.nn.functional.pad(result["y_hat"][:,:,padding_y:-padding_y,padding_y:-padding_y], padder_y, mode=args.padding_mode)
            # x_hat = crop(net.g_s(y_hat), padding)
            # result["likelihoods"]["y"] = crop(result["likelihoods"]["y"], padding_y)
            output_s = torch.clamp(result["x_hat"][:,:,args.pad:-args.pad,args.pad:-args.pad], min=0., max=1.0)

        bpp_ori = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())
        
        if args.defend:
            psnr = -10*math.log10(torch.mean((output_s - im_s)**2)) 
            print("Original PSNR:", psnr)
            y_main_s = net.g_a(im_s)
            
            # im_s_4 = anti_noise(im_s, scale=0.25)
            # im_s_2 = anti_noise(im_s, scale=0.5)
            # freq_1 = im_s_4
            # freq_2 = im_s_2 - im_s_4
            # freq_3 = im_s - im_s_2 - (im_s_2 - im_s_2)
            # y_main_low = net.g_a(im_s_2)
            # y_main_high = net.g_a(im_s - im_s_2 + 0.5)
            # y_main_anchor = net.g_a(torch.zeros_like(im_s)+0.5)
            # show_max_bar([y_main_s, y_main_low, y_main_high - y_main_anchor] , ["origin", "low frequency", "high frequency"], save_path="origin.pdf", sort=True, stack=True)
            # show_max_bar([y_main_s, y_main_low] , ["origin", "low frequency", "high frequency"], save_path="low_freq.pdf", sort=True)
            # show_max_bar([y_main_s, y_main_high - y_main_anchor] , ["origin", "high frequency"], save_path="high_freq.pdf", sort=True)

            y_main_ = defend(y_main_s, args)
            
            # show_max_bar(y_main, y_main_, label_a="origin", label_b="downsampled", save_path="origin.pdf")

            x_ = net.g_s(torch.round(y_main_))
            psnr = -10*math.log10(torch.mean((x_ - im_s)**2)) 
            print("PSNR after clipping:", psnr)
        
        # index_c = torch.ones_like(y_main)            
        # for i in range(y_main.shape[1]):
        #     y_main_ = torch.zeros_like(y_main)
        #     y_main_ = 0. + y_main
        #     y_main_[:,i,:,:] = 0.
        #     x_ = self.net.g_s(y_main_)
        #     # print(psnr, psnr + 10*math.log10(torch.mean((x_ - im_s)**2)))
        #     # remove channel with minumn psnr loss
        #     if abs(psnr + 10*math.log10(torch.mean((x_ - im_s)**2))) < 0.005:
        #         index_c[:,i,:,:] = 0

        # index_c = torch.sum(index_c, (0,2,3), keepdim=True, dtype=bool)
        # y_main = torch.where(index_c, y_main, torch.tensor(0., dtype=y_main.dtype, device=y_main.device))
        # x_ = self.net.g_s(y_main)
        # psnr = -10*math.log10(torch.mean((x_ - im_s)**2)) 
        # print(psnr)
        # index_c = torch.sum(torch.round(torch.abs(y_main)), (0,2,3), keepdim=True)
        # index_max = torch.where(torch.abs(y_main)>3, True, False)
        # index_max = torch.sum(index_max, (0,2,3), keepdim=True)
        # print(index_c, index_max)
        # torch.bitwise_or(torch.max(torindex_c) > 1, torch.sum(index_c) > 100)

        # index_c = torch.where(index_c > 10 + index_max, True, False)
        # print(index_c)
        # print(sorted(torch.sum(torch.round(y_main), (0,2,3), dtype=bool).cpu().numpy()))

        # output_s = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
        # if self.args.debug:
        #     layerout_s = layer_print(self.net, im_s)
        #     if self.args.model == "debug":
        #         pass
        #     else:
        #         print("GDN parameters:")
        #         layerout_s["gdn_gamma_mean"] = []
        #         layerout_s["gdn_beta_mean"] = []
        #         for gamma, beta in zip(layerout_s["gdn_gamma"], layerout_s["gdn_beta"]):
        #             mean_gamma, max_gamma = torch.mean(torch.abs(gamma)).item(), torch.max(torch.abs(gamma)).item()
        #             mean_beta, max_beta = torch.mean(torch.abs(beta)).item(), torch.max(torch.abs(beta)).item()
        #             print(mean_gamma, max_gamma, mean_beta, max_beta)
        #             # save mean_gamma, max_gamma to layerout_s as list
        #             layerout_s["gdn_gamma_mean"].append(mean_gamma)
        #             layerout_s["gdn_beta_mean"].append(mean_beta)

    batch_attack = False
    # LOSS_FUNC = args.att_metric
    noise_range = args.epsilon/255.0
    noise = torch.zeros(im_s.size())
    # noise = torch.Tensor(im_s.size()).uniform_(-1e-2,1e-2)
    noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise], lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
    net.train()
    c = args.lamb_attack
    for i in range(args.steps):
        noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
        # if args.pad:
            # noise_clipped = torch.nn.functional.pad(noise_clipped, (args.pad,args.pad, args.pad, args.pad), mode='constant', value=0)
        im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

        # im_in[:,:,H:,:] = 0.
        # im_in[:,:,:,W:] = 0.
        if batch_attack:
            loss_i_batch = torch.mean((im_s - im_in) ** 2, (1,2,3))
            # TODO: add batch attack
        else:
            loss, loss_i, loss_o = attack_our(im_s, output_s, im_in, net, args)
            # loss, loss_i, loss_o = attack_cw(im_s, output_s, im_in, net, args)

        # if args.debug and loss_i <= args.noise:
        #     if i == 1 or i > 900:
        #         optimizer.zero_grad()
        #         #get max grad and its index
        #         v_max = torch.amax(torch.abs(y_main), (0,2,3))
        #         v_mean = torch.mean(torch.abs(y_main), (0,2,3))
        #         grad_y_main = torch.autograd.grad(loss, y_main, retain_graph=True)[0]
        #         grad_mean = torch.mean(torch.clamp(grad_y_main, min=0.), dim=(0,2,3))
        #         r = grad_mean/v_mean
        #         print(torch.topk(r, k=10).detach())
        #         # grad_max = torch.max(torch.abs(y_main.grad), dim=1)
        #         index_max = torch.argmax(torch.abs(grad_y_main), dim=1)
        #         # numbers of each value in index_max
        #         index_c = torch.bincount(index_max.reshape(-1))

        optimizer.zero_grad()
        loss.backward()                
        optimizer.step()
        
        if i%100 == 0 and args.debug:
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", (1.-loss_o.item())/(loss_i.item()+1e-10))
                
        if i%(args.steps//3) == 0:
            lr_scheduler.step()
            if args.debug:
                # print(im_s.shape, output_s.shape)
                _, _, bpp, mse_in, mse_out, vi = eval(im_in, im_s, output_s, net, args)    
                net = net.train()
                # coder.write_image(output_, "%s_out_%d_%0.8f.png"%(filename, i, loss.item()), H, W)
    # noise = noise_clipped

    im_adv, output_adv, bpp, mse_in, mse_out, vi = eval(im_in, im_s, output_s, net, args) 
    # TODO: recalculate bpp
        
    return im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi

class attacker:
    def __init__(self, args):
        self.args = args
        self.mssim_func = torch_msssim.MS_SSIM(max_val=1).to(self.args.device)
        self.lpips_func = lpips.LPIPS(net='alex').to(self.args.device)
        print("==================== ATTACK SETTINGS ====================")
        print("[ IMAGE ]:", self.args.source, "->", self.args.target)
        print("Attack Loss Metric:", self.args.att_metric)
        print("Noise Threshold (L2):", self.args.noise, f"(epsilon={self.args.epsilon})")
        print(f"{args.steps} Steps")
        print("=========================================================")
        self.net = coder.load_model(args, training=False).to(self.args.device)
        self.model_config = f"{args.model}_{args.quality}_{args.metric}_"
    
    def attack(self, image_file, crop=None):
        C = 3
        im_s, H, W = coder.read_image(image_file)
        im_s = im_s.to(self.args.device)

        # self.y_main_s = torch.round(self.net.g_a(im_s))
        self.y_main_s = self.net.g_a(im_s)
        
        # mean_s = torch.mean(torch.abs(y_main_s), dim=(0,2,3))
        image_dir = "./attack/kodak/"
        filename = image_dir + self.model_config + image_file.split("/")[-1][:-4]
        # im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi = attack_(im_s, self.net, args)
        im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi = attack_ifgsm(im_s, self.net, args)
        self.noise = im_adv - im_s + 0.5
        if self.args.target:
            print("%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(im_adv, "%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%s.png"%(filename, self.args.target))
            coder.write_image(output_adv, "%s_advout_%s.png"%(filename, self.args.target))
        if self.args.debug:
            coder.write_image(im_adv, "%s_advin_%0.8f.png"%(filename, mse_in.item()))
            coder.write_image(output_adv, "%s_advout_%0.8f.png"%(filename, mse_out.item()))
            coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%0.8f.png"%(filename, mse_out.item()))
            self.net.eval()

            with torch.no_grad():
                eval(im_adv, im_s, output_s, self.net, self.args)
                
                self.y_main_adv = self.net.g_a(im_adv) 
                # self.y_main_adv = torch.round(self.net.g_a(im_adv))     
                
                show_max_bar([self.y_main_s, self.y_main_adv], ["nature examples", "adversarial examples"], save_path="activations.pdf", sort=True) 
            if self.args.defend:
                y_main_adv_defend = defend(self.y_main_adv, self.args) 
                show_max_bar([self.y_main_s, self.y_main_adv, y_main_adv_defend], ["origin", "adv", "defend"], save_path="activations_defend.pdf", sort=True) 
                # mean_adv = torch.mean(torch.abs(y_main_adv), dim=(0,2,3))
                # show y_main and y_main_adv in bar
                # reorder max_adv with the same order as max_s
                # max_adv = torch.tensor([max_adv[i] for i in max_s.argsort(descending=True)])
                # max_s = torch.sort(max_s, descending=True)[0]
                # max_adv = torch.sort(max_adv, descending=True)[0]
                # plot_bar([mean_s, mean_adv], ["origin", "adv"], "means.pdf")
        return bpp_ori.item(), bpp.item(), vi

def batch_attack(args):    
    myattacker = attacker(args)
    images = sorted(glob(args.source))
    bpp_ori_, bpp_, vi_ = 0., 0., 0.
    if args.debug:
        # distribution visulization
        y_main_s, y_main_adv = [], []

    for i, image in enumerate(images):
        # evaluate time of each attack
        start = time.time()
        bpp_ori, bpp, vi = myattacker.attack(image, crop=None)
        end = time.time()
        if args.debug:
            y_main_s.append(myattacker.y_main_s)
            y_main_adv.append(myattacker.y_main_adv)
        print(image, bpp_ori, bpp, vi, "Time:", end-start)
        bpp_ori_ += bpp_ori
        bpp_ += bpp
        vi_ += vi
    bpp_ori, bpp, vi = bpp_ori_/len(images), bpp_/len(images), vi_/len(images)
    print("AVG:", args.quality, bpp_ori, bpp, (bpp-bpp_ori)/bpp_ori, vi)
    if args.debug:
        y_main_s = torch.mean(torch.abs(torch.cat(y_main_s, dim=0)), dim=0, keepdim=True)
        y_main_adv = torch.mean(torch.abs(torch.cat(y_main_adv, dim=0)), dim=0, keepdim=True)
        show_max_bar([y_main_s, y_main_adv-y_main_s], ["nature examples", "adversarial examples"], save_path="activations_kodak.pdf", sort=True, stack=True)

def attack_bitrates(args):
    if args.quality > 0:
        batch_attack(args)
    else:
        if args.model == "cheng2020":
            q_max = 7
        else:
            q_max = 9
        for q in range(1, q_max):
            args.quality = q
            batch_attack(args)

if __name__ == "__main__":
    args = coder.config()
    args = args.parse_args()
    # batch_attack(args)
    attack_bitrates(args)
    