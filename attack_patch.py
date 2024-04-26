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
import torchvision
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
# from anchors.utils import layer_print


@torch.no_grad()
def eval(im_adv, im_s, output_s, net, args):
    net.eval()
    im_uint8 = torch.round(im_adv * 255.0)/255.0
    im_ =  torch.clamp(im_uint8, min=0., max=1.0)
    # save adverserial input
    # coder.write_image(im_, "%s_advin_%d_%0.8f.png"%(filename, i, loss.item()), H, W)
    if args.pad:
        result = net(F.pad(im_, (args.pad, args.pad, args.pad, args.pad), mode=args.padding_mode))
    else:
        result = net(im_)
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

    # if self.args.debug:
    #     layerout = layer_print(self.net, im_)
    #     # compare
    #     # print("Encoder:")
    #     # for layer, layer_s in zip(layerout["enc"], layerout_s["enc"]):
    #     #     mean_v, mean_after, mean_error = torch.mean((layer_s)**2).item(), torch.mean((layer)**2).item(), torch.mean((layer-layer_s)**2).item()
    #     #     print(mean_v, mean_after, mean_error, mean_error/mean_v)
    #     # print("Decoder:")
    #     index = 0    
    #     for layer, layer_s, f in zip(layerout["dec"], layerout_s["dec"], self.net.g_s._modules.values()):
    #         mean_v, mean_after, mean_error = torch.mean((layer_s)**2).item(), torch.mean((layer)**2).item(), torch.mean((layer-layer_s)**2).item()
    #         print(f"Layer-{index}:", mean_v**0.5, mean_after**0.5, mean_error, mean_error/mean_v)
    #         v_std = torch.mean((layer-(mean_after**0.5))**2)
    #         index += 1

    #         # if index in [2, 4, 6]:
    #         #     norm = layer/temp[0]
    #         #     norm_s = layer/temp[1]
    #         #     print("GDN norm:", torch.max(norm), torch.max(norm_s), torch.mean(norm), torch.mean(norm_s))

    #         temp = layer, layer_s
    #         if self.args.model == "debug":
    #             pass
    #         else:
    #             if index in [2, 4, 6]:
    #                 # gamma_mean = layerout_s["gdn_gamma_mean"][3:][index//2]
    #                 # beta_mean = layerout_s["gdn_beta_mean"][3:][index//2 ]
    #                 # print("GDN factor:", math.sqrt((mean_after+v_std)*128*gamma_mean+beta_mean))
    #                 beta = f.beta_reparam(f.beta)
    #                 gamma = f.gamma_reparam(f.gamma)
    #                 _, C, _, _ = temp[0].size()
    #                 gamma = gamma.reshape(C, C, 1, 1)
    #                 norm_ori = torch.sqrt(torch.nn.functional.conv2d(temp[1] ** 2, gamma, beta))
    #                 norm_adv = torch.sqrt(torch.nn.functional.conv2d(temp[0] ** 2, gamma, beta))
    #                 print("GDN norm:", torch.max(norm_ori), torch.max(norm_adv), torch.mean(norm_ori), torch.mean(norm_adv))
    
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
    return im_, output_, bpp, mse_in, mse_out, vi

def attack_onestep(im_s, net):
    # TODO: FreeAT
    pass
    # im_adv = im_s + torch.clamp(im_s - im_s_hat, min=0., max=1.0)
    # return im_adv

def psnr_partial(im_adv, output_adv, im_s, output_s):
    H, W = im_adv.shape[2], im_adv.shape[3]
    patch_adv = torch.nn.functional.unfold(im_adv, 64, stride=2).half()
    print(patch_adv.shape)
    patch_outadv = torch.nn.functional.unfold(output_adv, 64, stride=2).half()
    patch_s = torch.nn.functional.unfold(im_s, 64, stride=2).half()
    patch_outs = torch.nn.functional.unfold(output_s, 64, stride=2).half()
    new_h, new_w = int((H-64)/2)+1, int((W-64)/2)+1
    mse_out = torch.mean((patch_outs - patch_outadv)**2, dim=1).view(new_h, new_w)
    mse_in = torch.mean((patch_s - patch_adv)**2 , dim=1).view((new_h, new_w))
    print(H, W)
    print(mse_out.shape)
    patch_s = patch_s.view(1,3,64,64,new_h,new_w)
    patch_outs = patch_outs.view(1,3,64,64,new_h,new_w)
    patch_adv = patch_adv.view(1,3,64,64,new_h,new_w)
    patch_outadv = patch_outadv.view(1,3,64,64,new_h,new_w)

    vi = mse_out/mse_in
    vi[0:10, :] = 0
    vi[-10:, :] = 0
    vi[:, 0:10] = 0
    vi[:, -10:] = 0
    v, index_ = torch.max(vi, dim=1)
    v, index_h = torch.max(v, dim=0)
    print(v, index_h, index_[index_h])
    idx = index_h
    idy = index_[index_h]
    return patch_adv[:,:,:,:,idx,idy], patch_outadv[:,:,:,:,idx,idy], patch_s[:,:,:,:,idx,idy], patch_outs[:,:,:,:,idx,idy]
            
def attack_cw(im_s, net, args):
    pass

def attack_ifgsm(im_s, net, args):
    H, W = im_s.shape[2], im_s.shape[3]
    num_pixels = H * W

    # generate original output
    with torch.no_grad():
        net.eval()
        result = net(im_s)
        output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
        bpp_ori = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())

    scale = 1.
    batch_attack = False
    LOSS_FUNC = args.att_metric
    noise_range = args.epsilon/255.0
    # noise = torch.zeros(im_s.size())
    noise = torch.zeros(im_s.size())
    noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise], lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
    net.train()
    for i in range(args.steps):
        noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
        im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

        if batch_attack:
            loss_i_batch = torch.mean((im_s - im_in) ** 2, (1,2,3))
            # TODO: add batch attack
        else:
            loss_i = torch.mean((im_s - im_in) ** 2)
            if loss_i > args.noise:
                loss = loss_i
                loss_o = torch.Tensor([0.])
            else:
                y_main = net.g_a(im_in)
                if args.defend:
                    y_main = defend(y_main, args)
                x_ = net.g_s(y_main)
                # x_ = self.net(im_in)["x_hat"]
                if args.clamp:
                    output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
                else:
                    output_ = x_

                if LOSS_FUNC == "L2":
                    loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
                    # loss_o = 1. - (output_s - output_) ** 2
                if LOSS_FUNC == "L1":
                    loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))

                loss = loss_o

        optimizer.zero_grad()
        loss.backward()                
        optimizer.step()
        
        if i%100 == 0 and args.debug:
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", (1.-loss_o.item())/(loss_i.item()+1e-10))
                
        if i%(args.steps//3) == 0:
            lr_scheduler.step()
            if args.debug:
                print(im_s.shape, output_s.shape)
                _, _, bpp, mse_in, mse_out, vi = eval(im_in, im_s, output_s, net, args)    
                net.train()

    im_adv, output_adv, bpp, mse_in, mse_out, vi = eval(im_in, im_s, output_s, net, args)         
    return im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi

def attack_(im_s, net, args):
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
        result = net(im_s)
        output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
        bpp_ori = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())

    scale = 1.
    batch_attack = False
    LOSS_FUNC = args.att_metric
    noise_range = args.epsilon/255.0
    # noise = torch.zeros(im_s.size())
    noise = torch.Tensor(im_s.size()).uniform_(-1e-2,1e-2)
    noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise], lr=args.lr_attack)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
    net.train()
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
            loss_i = torch.mean((im_s - im_in) ** 2)
            if loss_i > args.noise:
                loss = loss_i
                loss_o = torch.Tensor([0.])
            else:
                if args.pad:
                    y_main = net.g_a(F.pad(im_in, padder, mode=args.padding_mode)) 
                else:
                    y_main = net.g_a(im_in)
                if args.defend:
                    y_main = defend(y_main, args)
                # if args.pad:
                #     y_main = torch.nn.functional.pad(y_main[:,:,padding_y:-padding_y,padding_y:-padding_y], padder_y, mode=args.padding_mode)    
                x_ = net.g_s(y_main)
                if args.pad:
                    x_ = crop(x_, padding)
                # x_ = self.net(im_in)["x_hat"]
                if args.clamp:
                    output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
                else:
                    output_ = x_

                # output_[:,:,H:,:] = 0.
                # output_[:,:,:,W:] = 0.

                if LOSS_FUNC == "L2":
                    loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
                    # loss_o = 1. - (output_s - output_) ** 2
                if LOSS_FUNC == "L1":
                    loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
                
                # if im_s.shape[0] == 1:
                # x_over = (x_ - output_)**2

                # loss = torch.mean(torch.where(x_over>0.0001, x_over, loss_o))
                # loss_o = torch.mean(loss_o)
                loss = loss_o

        optimizer.zero_grad()
        loss.backward()                
        optimizer.step()
        
        if i%100 == 0 and args.debug:
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", (1.-loss_o.item())/(loss_i.item()+1e-10))
                
        if i%(args.steps//3) == 0:
            lr_scheduler.step()
            if args.debug:
                print(im_s.shape, output_s.shape)
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

        # random crop
        # top = torch.randint(low=0, high=H-64, size=[1])[0]
        # left = torch.randint(low=0, high=W-64, size=[1])[0]
        # im_s = torchvision.transforms.functional.crop(im_s, top=top, left=left, height=64, width=64)

        image_dir = "./attack/patch/"
        filename = image_dir + self.model_config + image_file.split("/")[-1][:-4]
        im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi = attack_(im_s, self.net, args)
        if self.args.target:
            print("%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(im_adv, "%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(output_adv, "%s_advout_%s.png"%(filename, self.args.target))
            patch_adv, patch_outadv, patch_s, patch_outs = psnr_partial(im_adv, output_adv, im_s, output_s)
            
            filedir = image_dir + "/" + self.model_config + "/adv/"
            filename = filedir + image_file.split("/")[-1][:-4]
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            coder.write_image(patch_adv, "%s_advin_patch_%s.png"%(filename, self.args.target))
            coder.write_image(patch_outadv, "%s_advout_patch_%s.png"%(filename, self.args.target))

            filedir = image_dir + "/" + self.model_config + "/ori/"
            filename = filedir + image_file.split("/")[-1][:-4]
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            coder.write_image(patch_s, "%s_oriin_patch_%s.png"%(filename, self.args.target))
            coder.write_image(patch_outs, "%s_oriout_patch_%s.png"%(filename, self.args.target))

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
        for s in range(1):
            args.target = s+1
            bpp_ori, bpp, vi = myattacker.attack(image, crop=None)
            print(s, vi)

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
    