import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

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
from pytorch_msssim import ms_ssim

import coder
from anchors import balle
from utils import torch_msssim, ops
from anchors import model as models
# from anchors.utils import layer_compare
# from attack_rd import eval

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True, warn_only=True)


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
        
def bitdepth_reduction(x, bits=6, inference=True):
    assert bits > 0
    scale = 2**bits - 1
    if inference:
        x = torch.round(x * scale)/scale
    else:
        # TODO: approximate quantization
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        x = (x * scale + noise)/scale
    return x

def random_resize(x, scale=0.5, random=False):
    # #up/down sample
    # print("resize bicubic")
    if random:
        scale = np.random.uniform(low=0.5, high=0.75)
    x_down = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False, antialias=True)
    # x_down = F.interpolate(x, scale_factor=scale, mode="nearest")
    x_up = F.interpolate(x_down, scale_factor=1/scale, mode="bicubic", align_corners=False, antialias=True)
    # x_res = x - x_up

    # coder.write_image(x_res+0.5, "./logs/resize_nearst_high.png", H=x.shape[2], W=x.shape[3])
    # coder.write_image(x_up, "./logs/resize_nearst_low.png", H=x.shape[2], W=x.shape[3])
    return x_up, scale
    #  return F.interpolate(x_down, scale_factor=1/scale, mode="bicubic", align_corners=False, antialias=True)

def self_ensemble(net, x):
    xs = rotates(x)
    # output_ss = rotates(output_s)
    x_set0 = torch.cat(xs[:4], dim=0)
    x_set1 = torch.cat(xs[4:], dim=0) 
    best_mse, i = float("inf"), 0
    likelihoods = [0, 0]
    if net.training:
        y_main = net.g_a(x_set0)
        x_hats = net.g_s(y_main)
    else:
        result = net(x_set0)
        x_hats = result["x_hat"]

    num_pixels = (x.shape[2]) * (x.shape[3])
    for x, x_hat in zip(xs[:4], x_hats):
        mse = torch.mean((x - x_hat)**2)
        if mse < best_mse:
            best_idx, best_mse, best_x, best_x_hat = i, mse, x, x_hat
            if not net.training:
                likelihoods = [likelihoods_[i-4] for likelihoods_ in result["likelihoods"].values()]
                # bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
                # print(i, bpp, -10*math.log10(mse))
        i = i + 1

    if net.training:
        y_main = net.g_a(x_set1)
        x_hats = net.g_s(y_main)
    else:
        result = net(x_set1)
        x_hats = result["x_hat"]

    for x, x_hat in zip(xs[4:], x_hats):
        mse = torch.mean((x - x_hat)**2)
        if mse < best_mse:
            best_idx, best_mse, best_x, best_x_hat = i, mse, x, x_hat
            if not net.training:
                likelihoods = [likelihoods_[i-4] for likelihoods_ in result["likelihoods"].values()]
                # bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
                # print(i, bpp, -10*math.log10(mse))
        i = i + 1
    if len(likelihoods) == 1:
        likelihoods = {'y':likelihoods[0]}
    else:
        likelihoods = {'y':likelihoods[0], 'z':likelihoods[1]}
    # print(best_idx, -10*math.log10(mse))
    return best_mse, best_x, torch.clamp(rotates(torch.unsqueeze(best_x_hat, dim=0), reverse=best_idx), min=0., max=1.), likelihoods

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

def defend(net, x, method="ensemble"):
    assert method in ["ensemble", "resize", "bitdepth"], f"{method} not in 'ensemble', 'resize'"
    if method == "ensemble":
        return self_ensemble(net, x)
    if method == "bitdepth":
        x_ = bitdepth_reduction(x, inference=False)
        x_r = bitdepth_reduction(x, inference=True)
        output_ = net(x_)["x_hat"]
        return x_r, x_, output_, None
    if method == "resize":
        # x_, scale = random_resize(x, scale=0.5, random=True)
        x_, scale = random_resize(x, scale=243/256, random=False)
        output_ = net(x_)["x_hat"]
        return scale, x_, output_, None
    if method == "clip":
        raise NotImplementedError

@torch.no_grad()
def eval(im_adv, im_s, output_s, net, args):
    net.eval()
    im_uint8 = torch.round(im_adv * 255.0)/255.0
    im_ =  torch.clamp(im_uint8, min=0., max=1.0)

    result = net(im_)

    x_hat = result["x_hat"]
    
    mse_in = torch.mean((im_ - im_s)**2)
    if args.defend:
        v_, x, x_hat, likelihood = defend(net, im_, method=args.method)
        if args.method == "ensemble":     
            result["likelihoods"] = likelihood
        else:
            if args.method == "resize":
                im_pre, _ = random_resize(im_s, scale=v_, random=False)
            if args.method == "bitdepth":
                x = v_ # actual quantization
                im_pre = bitdepth_reduction(im_s, inference=True)
            output_pre = torch.clamp(net(im_pre)["x_hat"], min=0., max=1.0)
        result = net(x)
        x_hat = result["x_hat"]
    # --------- resubmit version
    # else: 
    #     if args.clamp:
    #         output_ = torch.clamp(x_hat, min=0., max=1.0)
    #     else:
    #         output_ = x_hat
    
    # --------- major revision version
    if args.clamp:
        output_ = torch.clamp(x_hat, min=0., max=1.0)
    else:
        output_ = x_hat    

    num_pixels = (im_adv.shape[2]) * (im_adv.shape[3])
    bpp = sum((torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)) for likelihoods in result["likelihoods"].values())
    
    mse_out = torch.mean((output_ - output_s)**2)
    msim_out = ms_ssim(output_, output_s, data_range=1., size_average=True).item()
    vi_results = {}
    mse_results = {"mse_in": mse_in, "mse_out": mse_out}
    msim_in = ms_ssim(im_, im_s, data_range=1., size_average=True).item()
    if args.method == "resize" or args.method == "bitdepth":
        # mse_results["mse_in_pre"] = torch.mean((x - im_pre)**2) # x_adv after preprocess, im_s after preprocess
        mse_results["mse_pre"] = torch.mean((im_s - x)**2) # decoded x_adv after preprocess, decoded im_s after preprocess
        vi_results["vi_pre"] = 10. * math.log10(mse_results["mse_pre"]/mse_results["mse_in"])

        # mse_resize = torch.mean((im_s - im_up)**2)
        # mse_out_resize = torch.mean((output_ - output_up)**2)
    # print(-10*math.log10(mse_out), mse_in)
    if mse_in > 1e-20 and mse_out > 1e-20:
        vi_results["vi"] = 10. * math.log10(mse_out/mse_in)
        if not args.adv:
            if msim_in < 0.9999:
                vi_results["vi_msim"] =  10. * math.log10((1-msim_out)/(1-msim_in))
        else:
            vi_results["vi_msim"] =  None
    else:
        vi_results["vi"], vi_results["vi_msim"] = None, None
        print(f"[!] Warning: mse_in ({mse_in}) or mse_out {mse_out} is zero")
    if args.debug:
        print(mse_results, vi_results)
        print("MS-SSIM in/out:", msim_in, msim_out)
        # print("[EVAL] loss_rec (vs Origin):", mse_results["mse_out"].item(), "loss_rec (vs Resized Input)", mse_results["mse_out_resize"], "loss_in:", mse_results["mse_in"].item(), "VI (mse):", vi_results["vi"])
    # mse_results = {"mse_in": mse_in, "mse_out": mse_out, "mse_in_resize": mse_in_resize, "mse_out_resize":mse_out_resize}
    # vi_results = {"vi": vi, "vi_resize": vi_resize}
    return im_, output_, bpp, mse_results, vi_results

def attack_our(im_s, output_s, im_in, net, args):
    loss_i = torch.mean((im_s - im_in) ** 2)
    if loss_i > args.noise:
        loss = loss_i
        loss_o = torch.Tensor([0.])
    else:
        if args.adv: # TODO: attack self-ensembles
            best_mse, best_x, x_, _ = defend(net, im_in, method=args.method)
        else:
            y_main = net.g_a(im_in)
            x_ = net.g_s(y_main)

        if args.clamp: # default: False
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)
        else:
            output_ = x_

        loss_o = 1. - torch.mean((output_s - output_) * (output_s - output_)) # MSE(x_, y_)
        loss = loss_o

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
            # print("Original PSNR:", psnr)
            # best_mse, best_x, best_x_hat, _ = defend(net, im_s)
            # psnr = -10*math.log10(best_mse)
            # print("PSNR after defending:", psnr)

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
        im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)

        loss, loss_i, loss_o = attack_our(im_s, output_s, im_in, net, args)

        optimizer.zero_grad()
        loss.backward()                
        optimizer.step()
        
        if i%100 == 0 and args.debug:
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item(), "VI (mse):", (1.-loss_o.item())/(loss_i.item()+1e-10))
                
        if i%(args.steps//3) == 0:
            lr_scheduler.step()
            if args.debug:
                _, _, bpp, mse_results, vi_results = eval(im_in, im_s, output_s, net, args)    
                net = net.train()

    im_adv, output_adv, bpp, mse_results, vi_results = eval(im_in, im_s, output_s, net, args) 
    return im_adv, output_adv, output_s, bpp_ori, bpp, mse_results, vi_results

class defender:
    def __init__(self, args):
        self.args = args
        self.method = args.method

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
        if args.defend:
            print("==================== DEFENSE SETTINGS ====================")
            print("Defense Method:", self.args.method)
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
        im_adv, output_adv, output_s, bpp_ori, bpp, mse_results, vi = attack_(im_s, self.net, args)
        
        self.noise = im_adv - im_s + 0.5
        if self.args.target:
            print("%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(im_adv, "%s_advin_%s.png"%(filename, self.args.target))
            coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%s.png"%(filename, self.args.target))
            coder.write_image(output_adv, "%s_advout_%s.png"%(filename, self.args.target))
        if self.args.debug:
            coder.write_image(im_adv, "%s_advin_%0.8f.png"%(filename, mse_results["mse_in"].item()))
            coder.write_image(output_adv, "%s_advout_%0.8f.png"%(filename, mse_results["mse_out"].item()))
            coder.write_image(torch.clamp(im_adv-im_s + 0.5, min=0., max=1.), "%s_noise_%0.8f.png"%(filename, mse_results["mse_out"].item()))
            self.net.eval()

            # with torch.no_grad():
            #     eval(im_adv, im_s, output_s, self.net, self.args)
            #     self.y_main_adv = self.net.g_a(im_adv) 
   
            # if self.args.defend:
            #     best_mse, x,  = defend(self.net, im_adv) 
            #     vi = 10*math.log10(best_mse/mse_in)
            #     print("After Self Ensemble:", vi)

        return bpp_ori.item(), bpp.item(), vi, mse_results

def batch_attack(args):    
    myattacker = attacker(args)
    images = sorted(glob(args.source))
    bpp_ori_, bpp_, vi_ = 0., 0., 0.
    if args.method == "resize" or args.method == "bitdepth":
        vi_preprocess_ = 0.

    if args.debug:
        # distribution visulization
        y_main_s, y_main_adv = [], []

    for i, image in enumerate(images):
        # evaluate time of each attack
        start = time.time()
        bpp_ori, bpp, vi_results, mse_results = myattacker.attack(image, crop=None)
        vi = vi_results["vi"]
        if args.method == "resize" or args.method == "bitdepth":
            vi_pre = vi_results["vi_pre"]
            vi_preprocess_ += vi_pre

        end = time.time()
        if args.debug:
            y_main_s.append(myattacker.y_main_s)
            y_main_adv.append(myattacker.y_main_adv)
        print(image, bpp_ori, bpp, vi, "Time:", end-start)
        bpp_ori_ += bpp_ori
        bpp_ += bpp
        vi_ += vi
    bpp_ori, bpp, vi = bpp_ori_/len(images), bpp_/len(images), vi_/len(images)
    if args.method == "resize" or args.method == "bitdepth":
        vi_pre = vi_preprocess_/len(images)
        # info = f"AVG {args.quality} {}"
        print("AVG:", args.quality, bpp_ori, bpp, (bpp-bpp_ori)/bpp_ori, vi, vi_pre)
    else:
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
            q_max = 7
        for q in range(1, q_max):
            args.quality = q
            batch_attack(args)

if __name__ == "__main__":
    args = coder.config()
    args = args.parse_args()
    attack_bitrates(args)
    