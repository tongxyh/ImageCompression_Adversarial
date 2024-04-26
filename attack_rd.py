<<<<<<< HEAD
import os
import sys
import argparse
from glob import glob
from datetime import datetime

import lpips
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import coder
from anchors import balle
from utils import torch_msssim, ops


def attack(args, crop=None):
    dev_id = "cuda:0"
    TRAINING = False
    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0

    if args.log:
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        tensor_log_path = './logs'
        if not os.path.exists(tensor_log_path):
            os.mkdir(tensor_log_path)
        writer = SummaryWriter(tensor_log_path+'/'+TIMESTAMP)

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
            output_t, _, _, _, _ = image_comp(im_t, TRAINING)
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
        output_s, y_main_, y_hyper, p_main, p_hyper = image_comp(im_s, TRAINING)
        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
    
    if crop == None:
        lamb = args.lamb_attack
        LOSS_FUNC = args.att_metric
        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        print("[WARNINTG] Quantization Skipped!")
        noise_range = 0.5
        noise = torch.zeros(im_s.size())
        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=args.lr_attack)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
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
                if loss_i >= args.noise:
                    loss = loss_i
                else:
                    loss = loss_o

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log:
                writer.add_scalar('Loss/mse_all', loss.item(), i)
                writer.add_scalar('Loss/mse_in', loss_i.item(), i)
                writer.add_scalar('Loss/mse_out',  loss_o.item(), i)
                writer.add_scalar('Loss/bpp',  bpp.item(), i)
            
            if i%100 == 0:
                if args.mask_loc != None: 
                    print(i, "loss_rec(ALL/TAR/BKG):", loss_o.item(), loss_o_tar.item(), loss_o_bkg.item(), "loss_in(ALL/TAR/BKG):", loss_i.item(), loss_tar.item(), loss_bkg.item())
                else:
                    print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item())
                    
            if i%(args.steps//3) == 0:
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
                    img.save("./attack/kodak/%s_fake%d_in_%0.8f.png"%(filename, i, loss.item())) 

                    output, y_main, y_hyper, p_main, p_hyper = image_comp(im_, TRAINING)   
                    
                    bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                    bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                    bpp = bpp_main + bpp_hyper
                    print("bpp:", bpp.item())
                    output_ = torch.clamp(output, min=0., max=1.0)
                    out = output_.data[0].cpu().numpy()
                    out = np.round(out * 255.0)
                    out = out.astype('uint8')
                    out = out.transpose(1, 2, 0)
                    
                    img = Image.fromarray(out[:H, :W, :])
                    img.save("./attack/kodak/%s_fake%d_out_%0.4f_%0.8f.png"%(filename, i, bpp.item(), loss.item()))
    if args.log:
        writer.close()
    return bpp, 0


if __name__ == "__main__":
    args = coder.config()
    print("============================================================")
    print("[ IMAGE ]:", args.source, "->", args.target)
    bpp, psnr = attack(args, crop=None)
=======
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
from pytorch_msssim import ms_ssim

import coder
from anchors import balle
from utils import torch_msssim, ops
from anchors import model as models
from anchors.utils import layer_compare
from self_ensemble import eval, defend

torch.set_default_dtype(torch.float32)

def entropy(net, y_main, model="hyper"):
    z = net.h_a(y_main)
    z_hat, _ = net.entropy_bottleneck(z)
    params = net.h_s(z_hat)
    if model == "hyper":
        scales_hat = params
        means_hat = 0.
    if model == "context":
        ctx_params = net.context_prediction(y_main)
        gaussian_params = net.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
    # print(scales_hat.shape, means_hat.shape, y_main.shape)
    return means_hat, scales_hat

def clamp_feature_with_p(x, y_main, means_hat, scales_hat, epsilon=50):
    # clip by distribution
    scales_hat = torch.clamp(scales_hat, min=0.11)
    pred_error = (y_main-means_hat) / scales_hat
    print(torch.max(y_main), torch.max(pred_error), torch.max(scales_hat))
    pred_error = torch.where(pred_error < epsilon, pred_error, torch.tensor(epsilon, dtype=pred_error.dtype, device=pred_error.device))
    pred_error = torch.where(pred_error > - epsilon, pred_error, torch.tensor(epsilon, dtype=pred_error.dtype, device=pred_error.device))
    return pred_error * (scales_hat) + means_hat

def clamp_value_naive(args, y_main):
    # index_max = torch.amax(y_main, dim=(2,3), keepdim=True)
    # index_min = torch.amin(y_main, dim=(2,3), keepdim=True)
    # thres = 1.5
    # index_max, index_min = torch.load(f"{args.model}-{args.metric}-{args.quality}.pt")
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
    # y_main = torch.where(index_max < thres, y_main_, y_main)
    # y_main = torch.where(index_min > -thres, y_main_, y_main)

    # y_main = torch.where(y_main > index_max, index_max, y_main)
    # y_main = torch.where(y_main < index_min, index_min, y_main)
    return y_main

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
    # figsize = 3.5, 2

    fig, ax = plt.subplots(constrained_layout=True)
    # fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    # maxs = [torch.amax(torch.abs(i), dim=(0,2,3)) for i in data]
    maxs = [torch.amax(i, dim=(0,2,3)) for i in data]
    mins = [torch.amin(i, dim=(0,2,3)) for i in data]

    ax_x = np.arange(0, maxs[0].shape[0], 1)
    # maxs = [torch.mean(torch.clamp(i, min=0), dim=(0,2,3)) for i in data]
    # maxs[0] = torch.mean(torch.clamp(data[0], min=0), dim=(0,2,3))
    # maxs[1] = torch.mean(torch.clamp(data[1], min=0), dim=(0,2,3))

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
    
    fontsize = 16
    ax.grid(linewidth=0.1, linestyle="--")
    plt.ylim(ymin=-25, ymax=25)
    # plt.ylim(ymin=-12, ymax=12)
    ax.legend(prop={'size': fontsize})

    ax.set_xlabel("channel index (sorted)", fontsize=fontsize)
    ax.set_ylabel("activation magnitude", fontsize=fontsize)
    if vi < 10:
        ax.text(0.95, 0.05, f"$\Delta$PSNR={vi:0.2f}", fontsize=20, ha='right', va='bottom', color='g', transform=ax.transAxes)
    else:
        ax.text(0.95, 0.05, f"$\Delta$PSNR={vi:0.2f}", fontsize=20, ha='right', va='bottom', transform=ax.transAxes)
        
    plt.savefig(f"./logs/{save_path}")
    # ax.plot(ax_x, maxs[0[].detach().cpu().numpy(), label=labels[0], linewidth=0.5)
    # plot_bar(ax, ax_x, maxs, labels, save_path, stack)

def defend_(y_main, args):
    # if args.adv:
    #     profile = f"{args.model}-{args.metric}-{args.quality}-adv.pt"
    # else:
    #     profile = f"{args.model}-{args.metric}-{args.quality}.pt"
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
    return clamp_value_naive(args, y_main)
    # return clip_dead_channel(y_main, profile)

def crop(x, padding):
    return x[:,:,padding:-padding,padding:-padding]

@torch.no_grad()
def eval_(im_adv, im_s, output_s, net, args):
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
    msim_in = ms_ssim(im_, im_s, data_range=1., size_average=True).item()
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

    # if args.debug:
    #     layer_compare(net, im_, im_s)
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
    msim_out = ms_ssim(output_, output_s, data_range=1., size_average=True).item()
    if mse_in > 1e-20 and mse_out > 1e-20:
        vi = 10. * math.log10(mse_out/mse_in)
        if not args.adv:
            vi_msim = 10. * math.log10((1-msim_out)/(1-msim_in))
        else:
            vi_msim = None
    else:
        vi, vi_msim = None, None
        print(f"[!] Warning: mse_in ({mse_in}) or mse_out {mse_out} is zero")
    if args.debug and vi:
        print("loss_rec:", mse_out.item(), "loss_in:", mse_in.item(), "VI (mse):", vi)
        print("MS-SSIM results:", msim_in, msim_out, -10. * math.log10(1-msim_in), -10. * math.log10(1-msim_out))
    mse_results = {"mse_in": mse_in, "mse_out": mse_out}
    vi_results = {"vi": vi, "vi_msim": vi_msim}
    return im_, output_, bpp, mse_results, vi_results

def attack_onestep(im_s, net):
    # TODO: FreeAT
    pass
    # im_adv = im_s + torch.clamp(im_s - im_s_hat, min=0., max=1.0)
    # return im_adv

def attack_our(im_s, output_s, im_in, net, args):
    loss_i = torch.mean((im_s - im_in) ** 2)
    if loss_i > args.noise:
        if args.att_metric == "ms-ssim":
            loss = 1. - ms_ssim(im_s, im_in, data_range=1.0, size_average=True)
        if args.att_metric == "L2":
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
        LOSS_FUNC = args.att_metric
        if LOSS_FUNC == "ms-ssim":
            loss_o = ms_ssim(output_, output_s, data_range=1.0, size_average=True)
        if LOSS_FUNC == "L2":
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
            if args.clamp:
                output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
            else:
                output_s = result["x_hat"]
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
            
            # Freq
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

            # y_main_ = defend(y_main_s, args)
            # show_max_bar(y_main, y_main_, label_a="origin", label_b="downsampled", save_path="origin.pdf")
            # x_ = net.g_s(torch.round(y_main_))
            # psnr = -10*math.log10(torch.mean((x_ - im_s)**2)) 
            # print("PSNR after clipping:", psnr)
        
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
    epsilon = args.noise

    if args.model == "debug":
        noise = torch.Tensor(im_s.size()).uniform_(-epsilon**0.5,epsilon**0.5)
    else:
        noise = torch.zeros(im_s.size())

    if args.random > 1:
        noise = torch.Tensor(im_s.size()).uniform_(-1e-2,1e-2)
        
    noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise], lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
    net.train()
    c = args.lamb_attack
    for i in range(args.steps):
        noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
        
        # print("== only low freq perturbation ==")
        # noise_clipped = anti_noise(noise_clipped, scale=0.25)

        # if args.pad:
            # noise_clipped = torch.nn.functional.pad(noise_clipped, (args.pad,args.pad, args.pad, args.pad), mode='constant', value=0)
        if args.model == "debug":
            im_in = im_s + noise_clipped
        else:
            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

        if batch_attack:
            loss_i_batch = torch.mean((im_s - im_in) ** 2, (1,2,3))
            # TODO: add batch attack
        else:
            # if i < 0.0*args.steps:
            #     args.noise = 1
            # else:
            #     args.noise = epsilon
            loss, loss_i, loss_o = attack_our(im_s, output_s, im_in, net, args)
            # im_in = 0.5 * torch.tanh*(noise) + 1
            # loss, loss_i, loss_o = attack_cw(im_s, output_s, noise, net, args)

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
            # print("[WARNING] No learning rate decay")
            if args.debug:
                # print(im_s.shape, output_s.shape)
                _, _, bpp, mse, vi = eval(im_in, im_s, output_s, net, args)    
                net = net.train()
                # coder.write_image(output_, "%s_out_%d_%0.8f.png"%(filename, i, loss.item()), H, W)
    # noise = noise_clipped
    im_in = im_in.to(torch.get_default_dtype())
    im_s = im_s.to(torch.get_default_dtype())        
    net = net.to(torch.get_default_dtype())
    # net.eval()
    # output_s = net(im_s)["x_hat"]
    if args.model == "debug":
        noise = im_in - im_s        
        for i in np.arange(0, 10, 0.5):
            im_in = im_s + noise.to(torch.get_default_dtype()) / 2**(3*i)
            eval(im_in, im_s, output_s, net, args)
    else:
        im_adv, output_adv, bpp, mse_results, vi_results = eval(im_in, im_s, output_s, net, args) 

    return im_adv, output_adv, output_s, bpp_ori, bpp, mse_results, vi_results

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
        im_s = im_s.to(self.args.device).to(torch.get_default_dtype())

        if self.args.debug:
            # self.y_main_s = torch.round(self.net.g_a(im_s))
            self.y_main_s = self.net.g_a(im_s)
        
        # mean_s = torch.mean(torch.abs(y_main_s), dim=(0,2,3))
        image_dir = "./attack/major_tcsvt/"
        filename = image_dir + self.model_config + image_file.split("/")[-1][:-4]
        im_adv, output_adv, output_s, bpp_ori, bpp, mse_results, vi_results = attack_(im_s, self.net, args)
        self.noise = im_adv - im_s + 0.5
        vi_results["vi_anchor"] = math.log10(mse_results["mse_in"]) / math.log10(mse_results["mse_out"])
        # normed_err = torch.log(torch.mean((torch.abs(output_adv - output_s)+1e-9) / (torch.mean(torch.abs(im_adv - im_s))+1e-9), dim=1)).cpu()[0]
        # print(torch.max(normed_err))
        # plt.imshow(normed_err, vmin=0, vmax=10)
        # plt.colorbar()
        # plt.savefig("./attack/minor_tcsvt/err_normed.pdf")
    
        if self.args.target:
            # print("%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(im_adv, "%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%s.png"%(filename, self.args.target))
            coder.write_image(output_adv, "%s_advout_%s.png"%(filename, self.args.target))
        if self.args.debug:
            coder.write_image(im_adv, "%s_advin_%0.8f.png"%(filename, mse_results["mse_in"].item()))
            coder.write_image(output_adv, "%s_advout_%0.8f.png"%(filename, mse_results["mse_out"].item()))
            # coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%0.8f.png"%(filename, mse_out.item()))
            self.net.eval()
            
            # Channel-wise Activation Visualization
            with torch.no_grad():
                eval(im_adv, im_s, output_s, self.net, self.args)
                self.y_main_adv = self.net.g_a(im_adv)
                # # self.y_main_adv = torch.round(self.net.g_a(im_adv))
                show_max_bar([self.y_main_s, self.y_main_adv], ["natural example", "adversarial example"], save_path="activations.pdf", sort=True, vi=vi)
                
            if self.args.defend:
                _, x_in, x_out, _ = defend(net, x, method=args.method)
                
                # y_main_adv_defend = defend(self.y_main_adv, self.args) 
                # show_max_bar([self.y_main_s, self.y_main_adv, y_main_adv_defend], ["origin", "adv", "defend"], save_path="activations_defend.pdf", sort=True) 
                
                # mean_adv = torch.mean(torch.abs(y_main_adv), dim=(0,2,3))
                # show y_main and y_main_adv in bar
                # reorder max_adv with the same order as max_s
                # max_adv = torch.tensor([max_adv[i] for i in max_s.argsort(descending=True)])
                # max_s = torch.sort(max_s, descending=True)[0]
                # max_adv = torch.sort(max_adv, descending=True)[0]
                # plot_bar([mean_s, mean_adv], ["origin", "adv"], "means.pdf")
        
        return bpp_ori.item(), bpp.item(), vi_results

def batch_attack(args):    
    myattacker = attacker(args)
    images = sorted(glob(args.source))
    bpp_ori_, bpp_, vi_, vi_anchor_, vi_msim_, t_ = 0., 0., 0., 0., 0., 0.
    if args.debug:
        # distribution visulization
        y_main_s, y_main_adv = [], []

    for i, image in enumerate(images):
        # evaluate time of each attack
        start = time.time()

        vi_best = -1.
        for random_idx in range(args.random):
            bpp_tmp, bppadv_tmp, vi_results_ = myattacker.attack(image, crop=None)
            # print(random_idx, vi_results["vi"])
            if vi_results_["vi"] > vi_best: # select the best one
                vi_best = vi_results_["vi"]
                bpp_ori, bpp, vi_results = bpp_tmp, bppadv_tmp, vi_results_

        end = time.time()
        if args.debug:
            y_main_s.append(myattacker.y_main_s)
            y_main_adv.append(myattacker.y_main_adv)
        print(image, bpp_ori, bpp, vi_results["vi"], vi_results["vi_msim"], "Time:", end-start)
        bpp_ori_ += bpp_ori
        bpp_ += bpp
        vi_ += vi_results["vi"]
        vi_anchor_ += vi_results["vi_anchor"]
        
        if vi_results["vi_msim"] and vi_msim_:
            vi_msim_ += vi_results["vi_msim"]
        else:
            vi_msim_ = None
            
        t_ += end - start
    num_im = len(images)
    if vi_msim_:
        vi_msim = vi_msim_/num_im
    else:
        vi_msim = None
    bpp_ori, bpp, vi, vi_anchor, t = bpp_ori_/num_im, bpp_/num_im, vi_/num_im, vi_anchor_/num_im, t_/num_im
    print(f"AVG: {args.model}-{args.metric}-{args.quality}", bpp_ori, bpp, (bpp-bpp_ori)/bpp_ori, vi, "vi_anchor:", vi_anchor, vi_msim, t)
    if args.debug:
        # y_main_s = torch.mean(torch.abs(torch.cat(y_main_s, dim=0)), dim=0, keepdim=True)
        # y_main_adv = torch.mean(torch.abs(torch.cat(y_main_adv, dim=0)), dim=0, keepdim=True)

        # TODO: mean or max?
        # y_main_s = torch.amax(torch.cat(y_main_s, dim=0), dim=0, keepdim=True)
        # y_main_adv = torch.amax(torch.cat(y_main_adv, dim=0), dim=0, keepdim=True)
        y_main_s = torch.cat(y_main_s, dim=0)
        y_main_adv = torch.cat(y_main_adv, dim=0)

        torch.save([y_main_s, y_main_adv, vi], "./attack/data/temp_mse.pt")

def visualize_actication():
    # visualize only
    y_main_s, y_main_adv, vi = torch.load("./attack/data/temp_mse.pt")
    show_max_bar([y_main_s, y_main_adv], ["natural image", "adversarial example"], save_path="activations_kodak.pdf", sort=True, stack=True, vi=vi)

def main(args):
    if args.quality > 0:
        batch_attack(args)
        if args.debug:
            visualize_actication()
    else:
        q_max = 7 if args.model == "cheng2020" else 9
        for q in range(1, q_max):
            args.quality = q
            batch_attack(args)

if __name__ == "__main__":
    args = coder.config()
    args = args.parse_args()
    main(args)
    
>>>>>>> dev
