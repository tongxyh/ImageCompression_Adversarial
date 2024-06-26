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
import lpips
from anchors import balle
from datetime import datetime

import classifier

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda() # [n_out, n_in, k_x, k_y]


def attack(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):

    TRAINING = True
    dev_id = "cuda:0"
    # read image
    precise = 16
    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0
    # print('====> Encoding Image:', im_dir)

    if args.log:
        # Writer will output to ./runs/ directory by default
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

        tensor_log_path = './logs'
        if not os.path.exists(tensor_log_path):
            os.mkdir(tensor_log_path)
        writer = SummaryWriter(tensor_log_path+'/'+TIMESTAMP)
        # writer = SummaryWriter("./logs/")

    ## model initalization
    MODEL = args.model
    quality = args.quality
    arch_lists = ["factorized", "hyper", "context", "cheng2020", "nlaic", "elic"]
    assert MODEL in arch_lists, f"'{MODEL}' not in {arch_lists} for param '-m'"
    if MODEL == "elic":
        image_comp = model.ImageCompression(256)
        image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
        # image_comp.load_state_dict(torch.load(checkpoint_dir).state_dict())
        # torch.save(image_comp.state_dict(), "./checkpoints/elic-0.0.1/ae.pkl")
        image_comp.to(dev_id).eval()
        print("[ ARCH  ]:", MODEL) 

    # if MODEL == "nlaic":
    #     # index - [0-15]
    #     models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
    #       "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]
    #     model_index = args.quality
    #     M, N2 = 192, 128
    #     if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
    #         M, N2 = 256, 192
    #     image_comp = model.Image_coding(3, M, N2, M, M//2)
    #     ######################### Load Model #########################
    #     image_comp.load_state_dict(torch.load(
    #         os.path.join(checkpoint_dir, models[model_index] + r'.pkl'), map_location='cpu'))

    if MODEL in ["factorized", "hyper", "context", "cheng2020"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric, pretrained=args.download).to(dev_id)
        print("[ ARCH  ]:", MODEL, quality, args.metric)
        if args.download == False:
            print("[ CKPTS ]:", args.ckpt)
            image_comp.load_state_dict(torch.load(args.ckpt))
            image_comp.to(dev_id).train()
        else:
            print("[ CKPTS ]: Download from CompressAI Model Zoo", )
    # Gradient Mask
    gnet = Gradient_Net().to(dev_id)

    mnist_classifier = classifier.Classifier()
    mnist_classifier.load_state_dict(torch.load("xxx"))
    mnist_classifier = mnist_classifier.cuda()
    
    y = one_hot([5], num_classes=10)

    # img_s = Image.open(source_dir).resize((16,16))
    img_s = Image.open(args.source)
    filename = args.source.split("/")[-1][:-4]
    # img_s = np.array(img_s)/255.0/5.0+0.5
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
        # img_t = Image.open(args.target).resize((64,64))
        img_t = Image.open(args.target)
        # img_t = np.array(img_t)/255.0/10.0+0.5
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
            output_t, _, _, _, _ = image_comp(im_t, False, CONTEXT, POSTPROCESS)
            output_t[:,:,H:,:] = 0.
            output_t[:,:,:,W:] = 0.            
            out = torch.clamp(output_t, min=0., max=1.0)
            out = out.data[0].cpu().numpy()
            out = np.round(out * 255.0)
            out = out.astype('uint8')
            out = out.transpose(1, 2, 0)
            img = Image.fromarray(out[:H, :W, :])
            img.save("./attack/kodak/%s_target_out.png"%(filename))

        ## TODO: background masking
        mask = torch.zeros(1,C,H_PAD,W_PAD)
        print(H_PAD, W_PAD)
        # mask[:,:, PADDING:H+PADDING, PADDING:W+PADDING] = torch.zeros(1,C,H,W)
        
        if args.mask_loc != None:
            mask = torch.ones(1,C,H_PAD,W_PAD)
            # x0,y0 = 83,78
            # x1,y1 = 151,103
            # lamb_bkg = args.lamb_bkg
            lamb_tar = args.lamb_tar
            print(args.mask_loc, "lamb_bkg:", args.lamb_bkg_in, args.lamb_bkg_out, "lamb_tar:", lamb_tar)
            x0,x1,y0,y1 = args.mask_loc # W, H
            mask[:,:, y0:y1, x0:x1] = torch.zeros(1,C,y1-y0,x1-x0) #(y0, y1), (x0, x1)
        mask_bkg = mask.to(dev_id)
        mask_tar = 1. - mask_bkg

    mssim_func = torch_msssim.MS_SSIM(max_val=1).to(dev_id)
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        # mask = gnet(im_s)
        mask = torch.ones_like(im_s)

        output_s, y_main_, y_hyper, p_main, p_hyper = image_comp(im_s, False, CONTEXT, POSTPROCESS)
        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
        # print("Original PSNR:", ) 
        # print("Original MS-SSIM:", )
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        # LOSS_FUNC = "L2"
        LOSS_FUNC = args.att_metric

        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        # im = im_s.clone().detach().requires_grad_(True) + torch.randn(im_s.size()).cuda()
        noise = torch.zeros(im_s.size())
        # if args.target == None:
            # noise = torch.rand(im_s.size()) - 0.5

        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=args.lr_attack)
        
        # im = (im_s+noise/10.0).clone().detach().requires_grad_(True)

        # im = im_s.clone().detach().requires_grad_(True)
        # optimizer = torch.optim.Adam([im],lr=1e-3)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
        noise_range = 0.5
        print("[WARNINTG] Quantization Skipped!")
        for i in range(args.steps):
            # clip noise range
            # noise_clipped = torch.clamp(mask*noise, min=-noise_range, max=noise_range)
            noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
            # im_in = torch.clamp(im_s+noise_clipped, min=0., max=1.0)
            
            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
            # im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+mask_tar*noise_clipped, 0.), 1.)
            # print("noised:",im_in)
            # print("source:",im_s)
            # 1. NO PADDING
            
            im_in[:,:,H:,:] = 0.
            im_in[:,:,:,W:] = 0.

            # print("TEST 0:", torch.sum(im_in) - torch.sum(im_in[:,:,:H,:W]))
            # print(torch.sum(im_in[:,:,H:,:]).item(), torch.sum(im_in[:,:,:,W:]).item(), torch.sum(im_in[:,:,H:,W:]).item(), torch.sum(im_in[:,:,:H,:W]).item(), torch.sum(im_in).item())
            # output, y_main, y_hyper, p_main, p_hyper = image_comp(im_in, TRAINING, CONTEXT, POSTPROCESS)


            # output_ = torch.clamp(output, min=0., max=1.0)
            # output_ = ops.Up_bound.apply(ops.Low_bound.apply(output, 0.), 1.)
            y_main = image_comp.net.g_a(im_in)
            x_ = image_comp.net.g_s(y_main)
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(x_, 0.), 1.)

            
            loss_i = torch.mean((im_s - im_in) * (im_s - im_in))                
            
            # image classification loss
            # load classifier
            # load label
            loss_o = loss_func(mnist_classifier(output_), y)

            # with torch.no_grad():
                # att = torch.tanh((output_s - output_) * (output_s - output_) / (noise_clipped*noise_clipped+0.0001))
                # print(torch.mean(mask))
                # mask = 0.9999*mask + 0.0001*att
                # noise_range = 0.9999*noise_range + 0.0001*50/255
            # bpp attack
            # train_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
            # train_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
            # bpp = train_bpp_main + train_bpp_hyper
            # lambda decay
            # loss = lamb * mse_o + (0.25 * mse_padding_o) + (mse_i + 0.25 * mse_padding_i)
            # loss = lamb * (-mse_o) + mse_i
            # loss = (mse_i-1e-3)**2.0 / (lamb *(mse_o) + 1e-6)
            
            # target loss
            # loss = mse_i - lamb * bpp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.log:
                writer.add_scalar('Loss/mse_all', loss.item(), i)
                writer.add_scalar('Loss/mse_in', mse_i.item(), i)
                writer.add_scalar('Loss/mse_out',  mse_o.item(), i)
                writer.add_scalar('Loss/bpp',  bpp.item(), i)
            
            if i%100 == 0:
                if args.mask_loc != None: 
                    print(i, "loss_rec(ALL/TAR/BKG):", loss_o.item(), loss_o_tar.item(), loss_o_bkg.item(), "loss_in(ALL/TAR/BKG):", loss_i.item(), loss_tar.item(), loss_bkg.item())
                else:
                    print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item())
                    
            if i % (args.steps//3) == 0:
                # print(torch.mean(mask), torch.mean(att))
                print("step:", i, "loss_rec:", loss_o.item(), "loss_in:", loss_i.item())
                lr_scheduler.step()
                with torch.no_grad():
                    im_uint8 = torch.round(im_in * 255.0)/255.0
                    
                    # 1. NO PADDING
                    # im_uint8[:,:,H:,:] = 0.
                    # im_uint8[:,:,:,W:] = 0.
                    
                    # save adverserial input
                    im_ =  torch.clamp(im_uint8, min=0., max=1.0)
                    # print(torch.min(im_), torch.max(im_))
                    # im_ =  torch.clamp(im, min=0., max=1.0)
                    # torch.set_printoptions(threshold=1000000)
                    # print(im_)
                    fin = im_.data[0].cpu().numpy()
                    fin = np.round(fin * 255.0)
                    fin = fin.astype('uint8')
                    fin = fin.transpose(1, 2, 0)
                    img = Image.fromarray(fin[:H, :W, :])
                    # img = Image.fromarray(fin[PADDING:H+PADDING, PADDING:W+PADDING, :])
                    img.save("./attack/kodak/%s_fake%d_in_%0.8f.png"%(filename, i, loss.item())) 
                    # img.save("./attack/kodak/final_in.png")
                    # debug
                    # img_sc = Image.open("./mnist/8_0/fake%d_in_%0.8f.png"%(i, loss.item()))
                    # img_sc = np.array(img_sc)
                    # print(img_sc/255.0)
                    
                    # im_scc = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
                    # # print(np.mean((fin*1.0 - img_sc*1.0)**2))
                    # im_scc[:H,:W,:] = img_sc/255.0
                    # # print(im_scc[:,:,:H,:W] - fin[:,:H,:W])
                    # im_scc = torch.FloatTensor(im_scc)
                    # im_scc = im_scc.permute(2, 0, 1).contiguous()
                    # im_scc = im_scc.view(1, C, H_PAD, W_PAD).cuda()
                    # print(torch.sum(im_[:,:,H:,W:]))
                    # print(torch.sum(im_scc[:,:,H:,W:]))
                    # print(torch.mean((im_scc[:,:,:H,:W] - im_[:,:,:H,:W])**2))
                    # print(torch.mean((im_scc - im_)**2))

                    output, y_main, y_hyper, p_main, p_hyper = image_comp(im_, False, CONTEXT, POSTPROCESS)   
                    
                    bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                    bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                    bpp = bpp_main + bpp_hyper
                    print("bpp:", bpp.item())
                    output_ = torch.clamp(output, min=0., max=1.0)
                    # print("loss:", torch.mean((output_ - output_t)**2*mask_tar))
                    out = output_.data[0].cpu().numpy()
                    out = np.round(out * 255.0)
                    out = out.astype('uint8')
                    out = out.transpose(1, 2, 0)
                    
                    # img = Image.fromarray(out[PADDING:H+PADDING, PADDING:W+PADDING, :])
                    img = Image.fromarray(out[:H, :W, :])
                    img.save("./attack/kodak/%s_fake%d_out_%0.4f_%0.8f.png"%(filename, i, bpp.item(), loss.item()))
    if args.log:
        writer.close()
    
    # attacked recons
    # img.save("./attack/kodak/final_out.png")

    # # original recons
    # output_ = torch.clamp(output_s, min=0., max=1.0)
    # out = output_.data[0].cpu().numpy()
    # out = np.round(out * 255.0)
    # out = out.astype('uint8')
    # out = out.transpose(1, 2, 0)

    # img = Image.fromarray(out[:H, :W, :])
    # img.save("./attack/kodak/origin_out.png")

    return 0, 0


if __name__ == "__main__":
    args = coder.config()

    print("============================================================")
    print("[ IMAGE ]:", args.source, "->", args.target)

    checkpoint = None
    if args.model == "nonlocal":
        checkpoint = glob('./ckpts/%d_%s/ae_%d_*' %(int(args.lamb), args.job, args.ckpt_num))[0]
        print("[CONTEXT]:", args.context)
        print("==== Loading Checkpoint:", checkpoint, '====')
    
    # target = "/ct/code/LearnedCompression/attack/colorattacker.png"
    # source = "/ct/code/LearnedCompression/attack/colorchecker.png"

    # target = "/ct/code/LearnedCompression/attack/licenseplate/MZ8723_180x180.png" 
    # source = "/ct/code/LearnedCompression/attack/licenseplate/MZ2837_180x180.png"

    # target = "/home/tong/LearnedCompression/mnist/tmp/roof_angle.png"
    # source = "/home/tong/LearnedCompression/mnist/tmp/roof_psnr.png"
    checkpoint = args.ckpt
    bpp, psnr = attack(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))