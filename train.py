import argparse
import math
import os
import random
import shutil
import sys
from datetime import datetime
import time
from glob import glob

import lpips
import numpy as np
import torch
import torch.nn as nn
from compressai import datasets
from PIL import Image
from torchvision import transforms
from pytorch_msssim import MS_SSIM
from utils import ops


import coder
from attack_rd import attack_

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, metric="mse", lmbda=1e-2, device="cuda:0"):
        super().__init__()
        self.metric = metric
        self.mse = nn.MSELoss()
        self.mssim = MS_SSIM(data_range=1, size_average=True, channel=3)
        self.lmbda = lmbda
        ## about lambda: https://interdigitalinc.github.io/CompressAI/zoo.html 
        self.lpips = lpips.LPIPS(net='alex').to(device)
        # lpips_loss = torch.mean(self.lpips(batch_x, output))

    def forward(self, output, target, training=True):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W
        out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
        # output["x_hat"] = ops.Low_bound.apply(ops.Up_bound.apply(output["x_hat"],1.))
        if not training:
            output["x_hat"] = torch.clamp(output["x_hat"],min=0.,max=1.)
            out["mse_loss"] = self.mse(output["x_hat"], target)
            out["msim_loss"] = self.mssim(output["x_hat"], target)
            out["psnr"] = -10.*math.log10(out["mse_loss"])
            out["msim_dB"] = -10.*math.log10(1.-out["msim_loss"])

        else:
            # if args.adv:
            #         out["bpp_loss"] = out["bpp_loss"] * 0.
            lamb_r = 1
            if self.lmbda == 100 or 1:
                # print("[WARNING] Inf Mode")
                lamb_r = 0
            if self.metric == "mse":
                out["distortion_loss"] = self.mse(output["x_hat"], target)
                out["loss"] = self.lmbda * 255 ** 2 * out["distortion_loss"] + lamb_r * out["bpp_loss"]
            if self.metric == "ms-ssim":
                # crop to 128x128
                # target = target[:,:,64:192,64:192]
                # output["x_hat"] = output["x_hat"][:,:,64:192,64:192]
                out["distortion_loss"] = self.mssim(output["x_hat"], target)
                # out["distortion_loss"] = self.mssim(output["x_hat"], target)
                # out["distortion_loss"] = self.mssim(torch.clamp(output["x_hat"],min=0., max=1.), target)
                out["loss"] = self.lmbda * (1 - out["distortion_loss"]) + lamb_r * out["bpp_loss"]
            if self.metric == "lpips":
                out["distortion_loss"] = torch.mean(self.lpips(output["x_hat"], target))
                out["loss"] = self.lmbda * out["distortion_loss"] + out["bpp_loss"]
                
        return out

def load_data(train_data_dir, train_batch_size, crop=256):
    if crop:
        train_transform = transforms.Compose([
            #transforms.Resize(256),
            #transforms.RandomResizedCrop(size=112),
            transforms.RandomCrop(crop),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    # if args.adv:
    #     train_split = "train-tiny"
    # else:
    train_split = "train"
    print(f"Training Dataset: {train_data_dir}/{train_split}")
    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        split=train_split,
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        train_data_dir, 
        split="test",
        transform=test_transform
    )
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False,
        num_workers=1,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    return train_loader, test_loader

def load_multi_data(train_data_dir, attack_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    attack_dataset = datasets.ImageFolder(
        attack_data_dir, 
        transform=train_transform
    )
    im_dataset = torch.utils.data.ConcatDataset([train_dataset, attack_dataset])
    train_loader = torch.utils.data.DataLoader(
        im_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    return train_loader

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def test_epoch(epoch, test_dataloader, model, criterion, log_dir, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    msim_loss = AverageMeter()
    d_loss = AverageMeter()
    if args.adv:
        vi_loss = AverageMeter()

    for d in test_dataloader:
        d = d.to(device)
        if args.adv:
            noise = args.noise
            args.noise = 0.0001
            vi_loss.update(attack_(d, model, args)[-1])
            args.noise = noise
        else:
            with torch.no_grad():
                out_net = model(d.detach())
                # out_net["x_hat"] = output_adv
                out_criterion = criterion(out_net, d, training=True)
                aux_loss.update(model.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                d_loss.update(out_criterion["distortion_loss"])
    
    # TODO: write to log
    # log = f"Test epoch {epoch}: Average losses:\tLoss: {loss.avg:.4f} |\tMSE loss: {mse_loss.avg:.6f} |\tMS-SSIM loss: {msim_loss.avg:.4f} |\tBpp loss: {bpp_loss.avg:.3f}\n"
    log = f"Test epoch {epoch}: Average losses:\tLoss: {loss.avg:.4f} |\tMSE loss: {d_loss.avg:.6f} |\tBpp loss: {bpp_loss.avg:.3f}\n"
    with open(log_dir, "a") as f:
        f.write(log)
    
    if args.adv:
        print(f"Test Loss (VI): {vi_loss.avg:.4f}")
        return vi_loss.avg
    else:
        print(log)
        return loss.avg

def save_checkpoint(state, is_best, ckpt_dir, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f"{ckpt_dir}/best_loss.pth.tar")

def train(args):
    # print("Load Only g_s and g_a parameters")
    # print("No bpp loss")
    dev_id = "cuda:0"
    C = 3
    batch_size = args.batch_size
    lambs = {
                "mse": [0.0018, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483, 0.0932, 0.1800], 
                "ms-ssim": [2.40,4.58,8.73,16.64,31.73,60.50,115.37,220.00]
            }
    # model initalization
    image_comp, last_epoch, optimizer, aux_optimizer, lr_scheduler = coder.load_model(args, training=True)
    # image_comp.cuda()
    # image_comp = nn.DataParallel(image_comp, device_ids=[0])
    if args.lamb == None:
        lamb = lambs[args.metric][args.quality-1]
    else:
        lamb = args.lamb
    print("Lambda:", lamb)
    print("Learning rate (training):", args.lr_train)
    print("Learning rate (adversarial):", args.lr_attack)
    if lamb == 100 or lamb == 1:
        model_dir = f"{args.model}-Inf-{args.metric}"
    else:
        model_dir = f"{args.model}-{lamb}-{args.metric}"
    epochs_num = 200
    if args.adv:
        epochs_num = 100
        noise_range = args.noise
        model_dir += f"-{args.noise}-{args.steps}"
        ckpt_dir = f"./ckpts/adv/{model_dir}"
    else:
        ckpt_dir = f"./ckpts/anchor/{model_dir}"

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    train_data = '/workspace/dataset/vimeo'
    print("Save ckpts to:", ckpt_dir)
    print("Load train data:", train_data)
    # with open(f"{ckpt_dir}/log.txt", "w") as f:
        # f.write("=========================={model_dir}=======================\n")

    criterion = RateDistortionLoss(metric=args.metric, lmbda=lamb)

    # train_loader = load_data('/workspace/ct/datasets/datasets/div2k', batch_size)
    # train_loader = load_multi_data('/workspace/ct/datasets/datasets/div2k', f'/workspace/ct/datasets/attack/{model_dir}/iter-2', batch_size)
    
    best_loss = float("inf")
    if args.fintune:
        epochs_num = 10

    if args.adv:
        N_ADV = 0
        print(batch_size - N_ADV, "adv examples in all", batch_size)
    
    # confirm = input("Do you confirm the settings? (y or n)")
    # if confirm:
    #     pass
    # else:
    #     return

    t = time.time()
    train_dataloader, test_dataloader = load_data(train_data, batch_size, crop=256)
    print(f"Dataloader cost {time.time() - t}s") 

    for epoch in range(last_epoch, epochs_num):
        t = time.time()
        for step, batch_x in enumerate(train_dataloader):
            # optimizer.zero_grad()
            # aux_optimizer.zero_grad()

            batch_x = batch_x.to('cuda')
                        
            if args.adv:   
                im_s = batch_x[N_ADV:,:,:,:]
                batch_x = batch_x.detach()
                if step <=100:
                    args.noise = noise_range * step/100
                optimizer.zero_grad()
                aux_optimizer.zero_grad()
                batch_adv = attack_(im_s, image_comp, args)[0]
                batch_adv = batch_adv.detach()
                
                # print(torch.mean((batch_adv - im_s)**2))
                image_comp.train()
                batch_x[N_ADV:,:,:,:] = batch_adv
                # coder.write_image(batch_x[N_ADV:N_ADV+1], f"./logs/at_in_{step}.png")
                
                for i in range(1): 
                    result = image_comp(batch_x)
                    
                    out_criterion = criterion(result, batch_x)
                    # save batch_adv as image
                    # coder.write_image(batch_adv[0:], f"{ckpt_dir}/at_in_{i}.png")
                    # coder.write_image(torch.clamp(result["x_hat"][N_ADV:N_ADV+1],min=0.,max=1.), f"{ckpt_dir}/at_out_{step}_{i}.png")
                    optimizer.zero_grad()
                    aux_optimizer.zero_grad()
                    out_criterion["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 1.0)
                    # torch.nn.utils.clip_grad_value_(image_comp.parameters(), 1.0)
                    optimizer.step()
                    # print(i, out_criterion["distortion_loss"].item())
                    aux_loss = image_comp.aux_loss()
                    aux_loss.backward()
                    aux_optimizer.step()
            else:
                image_comp.train()
                result = image_comp(batch_x)
                out_criterion = criterion(result, batch_x)
                out_criterion["loss"].backward()
                torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 1.0)
                optimizer.step()
            
                aux_loss = image_comp.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()

            # with torch.no_grad():
            #     print("Before:", out_criterion["distortion_loss"].item())
            #     result = image_comp(batch_adv)
            #     out_criterion = criterion(result, batch_x)
            #     print("After:", out_criterion["distortion_loss"].item())

            if args.adv: 
                if step%10 == 0 and step > 0:
                    # for i in range(N_ADV, batch_size):
                        # coder.write_image(batch_x[i:i+1], f"./logs/at/in_{step}_{i}.png")
                        # coder.write_image(torch.clamp(result["x_hat"][i:i+1],min=0.,max=1.0), f"./logs/at/out_{step}_{i}.png")
                    print('step:', step, 'loss:', out_criterion["loss"].item(), "distortion:", out_criterion["distortion_loss"].item(), 'rate:', out_criterion["bpp_loss"].item(), f"lr: {optimizer.param_groups[0]['lr']}", "Epoch Time:", time.time() - t)
                    loss = test_epoch(epoch, test_dataloader, image_comp, criterion, f"{ckpt_dir}/log.txt", args)
                    lr_scheduler.step(loss)
                    is_best = loss < best_loss
                    print("New Best:", is_best)
                    best_loss = min(loss, best_loss)
                    if is_best or step%100 == 0:
                        save_checkpoint(
                        {
                            "epoch": epoch,
                            "step": step,
                            "state_dict": image_comp.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best, ckpt_dir, filename=f"{ckpt_dir}/ckpt-{epoch}-{step}.pth.tar"
                        )
                if step == 2000:
                    return
            elif step%10000 == 0:
                print('step:', step, 'loss:', out_criterion["loss"].item(), "distortion:", out_criterion["distortion_loss"].item(), 'rate:', out_criterion["bpp_loss"].item(), f"lr: {optimizer.param_groups[0]['lr']}", "Epoch Time:", time.time() - t)
                # torch.save(image_comp.module.state_dict(), os.path.join(ckpt_dir,'ae_%d_%d_%0.8f_%0.8f.pkl' % (epoch, step, loss_epoch/(step+1), bpp_epoch/(step+1))))
                # torch.save(image_comp.state_dict(), os.path.join(ckpt_dir,'ae_%d_%d_%0.4f_%0.8f_%0.8f.pkl' % (epoch, step, bpp_sum/(step+1), dloss_sum/(step+1), loss_sum/(step+1))))
                loss = test_epoch(epoch, test_dataloader, image_comp, criterion, f"{ckpt_dir}/log.txt", args)
                lr_scheduler.step(loss)

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": image_comp.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best, ckpt_dir, filename=f"{ckpt_dir}/ckpt-{epoch}.pth.tar"
                    )
        print("Epoch Time:", time.time() - t)
        if not args.adv:
            loss = test_epoch(epoch, test_dataloader, image_comp, criterion, f"{ckpt_dir}/log.txt", args)
            lr_scheduler.step(loss)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": image_comp.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best, ckpt_dir, filename=f"{ckpt_dir}/ckpt-{epoch}.pth.tar"
                )

if __name__ == "__main__":
    args = coder.config()
    args = args.parse_args()
    train(args)
