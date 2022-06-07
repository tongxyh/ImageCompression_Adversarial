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

def rotates(x, reverse=-1):
    if reverse == -1:
        x0 = torch.flip(x, [2])
        x1 = torch.flip(x, [3])
        x2 = torch.flip(x0, [3])

        x3 = torch.rot90(x, 1, [2,3])
        x4 = torch.flip(x3, [2])
        x5 = torch.flip(x3, [3])
        x6 = torch.flip(x4, [3])
        return x, x0, x1, x2, x3, x4, x5, x6
    else:
        cases = {
            0: x,
            1: torch.flip(x, [2]),
            2: torch.flip(x, [3]),
            3: torch.flip(torch.flip(x, [3]), [2]),
            4: torch.rot90(x, -1, [2,3]),
            5: torch.rot90(torch.flip(x, [2]), -1, [2,3]),
            6: torch.rot90(torch.flip(x, [3]), -1, [2,3]),
            7: torch.rot90(torch.flip(torch.flip(x, [3]), [2]), -1, [2,3]),
        }
        return cases[reverse]
        
def self_ensemble(net, x):
    xs = rotates(x)
    # output_ss = rotates(output_s)
    x_set0 = torch.cat(xs[:4], dim=0)
    x_set1 = torch.cat(xs[4:], dim=0) 
    best_mse, i = 1, 0

    result = net(x_set0)
    for x, x_hat in zip(xs[:4], result["x_hat"]):
        mse = torch.mean((x - x_hat)**2)
        # print(-10*math.log10(mse))
        if mse < best_mse:
            best_idx, best_mse, best_x, best_x_hat = i, mse, x, x_hat
        i = i + 1

    result = net(x_set1)
    for x, x_hat in zip(xs[4:], result["x_hat"]):
        mse = torch.mean((x - x_hat)**2)
        # print(-10*math.log10(mse))
        if mse < best_mse:
            best_idx, best_mse, best_x, best_x_hat = i, mse, x, x_hat
        i = i + 1
    return best_mse, best_x, torch.clamp(rotates(torch.unsqueeze(best_x_hat, dim=0), reverse=best_idx), min=0., max=1.)

def defend(net, x):
    return self_ensemble(net, x)

def crop(x, padding):
    return x[:,:,padding:-padding,padding:-padding]

@torch.no_grad()
def eval(im_adv, im_s, output_s, net, args):
    net.eval()
    im_uint8 = torch.round(im_adv * 255.0)/255.0
    im_ =  torch.clamp(im_uint8, min=0., max=1.0)

    result = net(im_)

    x_hat = result["x_hat"]
    
    mse_in = torch.mean((im_ - im_s)**2)
    if args.defend:
        _, x, output_ = defend(net, im_)
        result = net(x)
    else:
        if args.clamp:
            output_ = torch.clamp(x_hat, min=0., max=1.0)
        else:
            output_ = x_hat

    if args.debug:
        layer_compare(net, im_, im_s)
    
    num_pixels = (im_adv.shape[2]) * (im_adv.shape[3])
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())
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
        
        # TODO: attack self-ensemble
        # if args.defend:
        #     y_main = defend(y_main, args)

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

def attack_(im_s, net, args):

    # print("[*] Input size:", im_s.size())
    H, W = im_s.shape[2], im_s.shape[3]
    num_pixels = H * W

    # generate original output
    with torch.no_grad():
        net.eval()

        result = net(im_s)
        output_s = torch.clamp(result["x_hat"], min=0., max=1.0)
        bpp_ori = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())
        
        if args.defend:
            psnr = -10*math.log10(torch.mean((output_s - im_s)**2)) 
            print("Original PSNR:", psnr)
            best_mse, best_x, best_x_hat = defend(net, im_s)
            psnr = -10*math.log10(best_mse) 
            print("PSNR after defending:", psnr)

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
        im_adv, output_adv, output_s, bpp_ori, bpp, mse_in, mse_out, vi = attack_(im_s, self.net, args)
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

            # with torch.no_grad():
            #     eval(im_adv, im_s, output_s, self.net, self.args)
            #     self.y_main_adv = self.net.g_a(im_adv) 
   
            # if self.args.defend:
            #     best_mse, x,  = defend(self.net, im_adv) 
            #     vi = 10*math.log10(best_mse/mse_in)
            #     print("After Self Ensemble:", vi)

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
    