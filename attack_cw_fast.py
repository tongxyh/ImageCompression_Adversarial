import os
from pydoc import doc
import sys
import math
import argparse
from glob import glob
import time
from datetime import datetime
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

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

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

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
def attack_cw(im_s, output_s, im_in, net, c, noise_level):
    # TODO: input preprocess
    # c = args.lamb_attack
    loss_i = torch.mean((im_s - im_in) ** 2)
    
    # else:
    #     c1 = 0.
    #     c = 1.

    # if loss_i > args.noise:
    #     loss = loss_i
    #     # y_main = net.g_a(im_in)  
    #     # x_ = net.g_s(y_main)
    #     # output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)    
    #     # loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
    #     loss_o = torch.Tensor([0.])
    # else:
    
    y_main = net.g_a(im_in)  
    x_ = net.g_s(y_main)
    output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)    
    loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
    if 1. - loss_o > noise_level*1.1:
        c = 0
    loss = loss_i + c * loss_o
    return loss, loss_i, loss_o

def search_noise(im_s, output_s, net, noise_level):
    epsilon = noise_level
    noise_range = args.epsilon/255.0
    noise = torch.zeros(im_s.size())
    # noise = torch.Tensor(im_s.size()).uniform_(-1e-2,1e-2)
    noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise], lr=args.lr_attack)
    
    c_r = args.lamb_attack
    c_l = 0
    c = c_r
    # loss_i = 0
    loss_i = torch.Tensor([0.])
    # while abs(c_r - c_l) > 0.0001 and (abs(c_r - c_l) > 0.01 or torch.abs(loss_i - args.noise) > args.noise*0.01):    
    while abs(c_r - c_l) > 0.0001 and (abs(c_r - c_l) > 0.01 or torch.abs( 1 - loss_o - 0.99*noise_level) > noise_level*0.01): 
        for i in range(args.steps):
            noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
            # if args.pad:
                # noise_clipped = torch.nn.functional.pad(noise_clipped, (args.pad,args.pad, args.pad, args.pad), mode='constant', value=0)
            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
            # im_in = 0.5 * torch.tanh(noise) + 1
            # args.noise = 1.
            # c = 1
            loss, loss_i, loss_o = attack_cw(im_s, output_s, im_in, net, c, noise_level)
            # print(i, c,"loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", 10*math.log10((1.-loss_o.item())/(loss_i.item()+1e-10)))
            optimizer.zero_grad()
            loss.backward()                
            optimizer.step()
        if args.debug:
            print("\t", c,"loss_rec", 1 - loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", 10*math.log10((1.-loss_o.item())/(loss_i.item()+1e-10)))
        # if loss_i < 2 * epsilon:
        #     for i in range(args.steps):
        #         noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
        #         # if args.pad:
        #             # noise_clipped = torch.nn.functional.pad(noise_clipped, (args.pad,args.pad, args.pad, args.pad), mode='constant', value=0)
        #         im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
        #         # im_in = 0.5 * torch.tanh(noise) + 1
        #         args.noise = epsilon
        #         # c = 0
        #         loss, loss_i, loss_o = attack_cw(im_s, output_s, im_in, net, c)
        #         # print(i, c,"loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", 10*math.log10((1.-loss_o.item())/(loss_i.item()+1e-10)))
        #         optimizer.zero_grad()
        #         loss.backward()                
        #         optimizer.step()

            # print("Round 2", c,"loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", 10*math.log10((1.-loss_o.item())/(loss_i.item()+1e-10)))
        # print(c, c_l, c_r)
        # if loss_o > 1 - args.noise*100:
        # print(abs(c_r - c_l), 0.0001, torch.abs(loss_i - args.noise), args.noise*0.01)    
        # if loss_i < epsilon:
        if 1 - loss_o < 0.99*epsilon:
            c_l = c   
        else:
            c_r = c
        c = (c_r + c_l)/2
        # print(c, c_l, c_r)
    return loss_i, loss_o, im_in

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
    net.train()
    min_noise = args.noise
    max_noise = 0.1
    noise_level = max_noise
    loss_i = torch.Tensor([0.]).cuda()
    
    search_cnt = 0
    # while torch.abs(loss_i - args.noise) > args.noise * 0.01 and search_cnt < 20: # tcsvt-resubmit-version
    while search_cnt < args.search_steps:
        if args.debug:
            print(noise_level, loss_i.item())
        # print(noise_level, loss_i.item())
        loss_i_old = loss_i
        loss_i, loss_o, im_in = search_noise(im_s, output_s, net, noise_level)
        if torch.abs(loss_i - loss_i_old) < args.noise * 0.01 and torch.abs(loss_i - args.noise) < args.noise * 0.1:
            break
        
        if loss_i > args.noise:
            max_noise = noise_level
        else:
            min_noise = noise_level
        noise_level = (min_noise + max_noise)/2
        search_cnt += 1
        
    im_adv, output_adv, bpp, mse_in, mse_out, vi = eval(im_in, im_s, output_s, net, args) 
        
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

            with torch.no_grad():
                eval(im_adv, im_s, output_s, self.net, self.args)

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
        # if args.debug:
        #     y_main_s.append(myattacker.y_main_s)
        #     y_main_adv.append(myattacker.y_main_adv)
        print(image, bpp_ori, bpp, vi, "Time:", end-start)
        bpp_ori_ += bpp_ori
        bpp_ += bpp
        vi_ += vi
    bpp_ori, bpp, vi = bpp_ori_/len(images), bpp_/len(images), vi_/len(images)
    print("AVG:", args.quality, bpp_ori, bpp, (bpp-bpp_ori)/bpp_ori, vi)
    # if args.debug:
    #     y_main_s = torch.mean(torch.abs(torch.cat(y_main_s, dim=0)), dim=0, keepdim=True)
    #     y_main_adv = torch.mean(torch.abs(torch.cat(y_main_adv, dim=0)), dim=0, keepdim=True)
    #     show_max_bar([y_main_s, y_main_adv-y_main_s], ["nature examples", "adversarial examples"], save_path="activations_kodak.pdf", sort=True, stack=True)

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
    