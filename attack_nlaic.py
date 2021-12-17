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

# from utils import torch_msssim, ops
import Model.model as model
from Model.context_model import Weighted_Gaussian
from Util import ops
from datetime import datetime

class Gradient_Net(nn.Module):
  def __init__(self):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda() # [n_out, n_in, k_x, k_y]

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

    self.weight_x = nn.Parameter(data=kernel_x.repeat(1,3,1,1), requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y.repeat(1,3,1,1), requires_grad=False)

  def forward(self, x):
    grad_x = nn.functional.conv2d(x, self.weight_x, padding=1)
    grad_y = nn.functional.conv2d(x, self.weight_y, padding=1)
    gradient = torch.tanh(torch.abs(grad_x) + torch.abs(grad_y))
    return gradient


def add_noise(x):
    noise = np.random.uniform(-0.5, 0.5, x.size())
    noise = torch.Tensor(noise).cuda()
    return x + noise

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


    models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
        "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]
    model_index = args.quality
    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M//2)
    context = Weighted_Gaussian(M)
    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    context.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, models[model_index] + r'p.pkl'), map_location='cpu'))
    image_comp = image_comp.to(dev_id)
    context = context.to(dev_id)
    # Gradient Mask
    # gnet = Gradient_Net().to(dev_id)
    #msssim_func = msssim_func.cuda()

    # img_s = Image.open(source_dir).resize((16,16))
    img_s = Image.open(args.source)
    # img_s = np.array(img_s)/255.0/5.0+0.5
    img_s = np.array(img_s)/255.0
    filename = args.source.split("/")[-1][:-4]
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
            output_t, _, _, _, _ = image_comp(im_t, 2)
        
        ## TODO: background masking
        mask = torch.ones(1,C,H_PAD,W_PAD)
        # mask[:,:, PADDING:H+PADDING, PADDING:W+PADDING] = torch.zeros(1,C,H,W)
        # x0,y0 = 83,78
        # x1,y1 = 151,103
        x0,y0 = 0,0
        x1,y1 = W,H
        mask[:,:, y0:y1, x0:x1] = torch.zeros(1,C,y1-y0,x1-x0) #(y0, y1), (x0, x1)
        mask_bkg = mask.to(dev_id)
        mask_tar = 1. - mask_bkg

    with torch.no_grad():
        # mask = gnet(im_s)
        mask = torch.ones_like(im_s)

        output_s, _, p_hyper, y_main_q, hyper_dec = image_comp(im_s, 2) # 2 means round
        p_main, _ = context(y_main_q, hyper_dec)

        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
        # print("Original PSNR:", )
        # print("Original MS-SSIM:", )
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        lamb_bkg = 0.01
        LOSS_FUNC = "L2"
        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        # im = im_s.clone().detach().requires_grad_(True) + torch.randn(im_s.size()).cuda()
        noise = torch.zeros(im_s.size())
        # if args.target == None:
        #     noise = torch.rand(im_s.size()) - 0.5
 
        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=args.lr_attack)
        
        # im = (im_s+noise/10.0).clone().detach().requires_grad_(True)

        # im = im_s.clone().detach().requires_grad_(True)
        # optimizer = torch.optim.Adam([im],lr=1e-3)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
        noise_range = 0.5
        for i in range(args.steps):
            # clip noise range
            # noise_clipped = torch.clamp(mask*noise, min=-noise_range, max=noise_range)
            noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
            # im_in = torch.clamp(im_s+noise_clipped, min=0., max=1.0)
            im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
            if args.target != None:
                im_in = torch.clamp(im_s+noise, min=0., max=1.0)
            # print("noised:",im_in)
            # print("source:",im_s)
            # 1. NO PADDING
            im_in[:,:,H:,:] = 0.
            im_in[:,:,:,W:] = 0.
            
            output, _, _, _, _ = image_comp(im_in, TRAINING)

            # output_ = torch.clamp(output, min=0., max=1.0)
            output_ = ops.Up_bound.apply(ops.Low_bound.apply(output, 0.), 1.)
            if LOSS_FUNC == "L2":
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))                
                if args.target == None:
                    # loss_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(y_s, y_)
                    loss_o = 1. - torch.mean((im_s - output_) * (im_s - output_)) # MSE(x_, y_)
                else:
                    # loss_o = torch.mean((output_t - output_) * (output_t - output_)) # MSE(y_t, y_s)
                    loss_o = torch.mean((im_t - output_) * (im_t - output_)) # MSE(y_t, y_s)

            # L1 loss
            if LOSS_FUNC == "L1":
                loss_i = torch.mean(torch.abs(im_s - im_in))
                if args.target == None:
                    loss_o = 1.0 - torch.mean(torch.abs(im_s - output_))
                else:    
                    loss_o = torch.mean(torch.abs(output_t - output_))
            
            if LOSS_FUNC == "masked":
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in)) * mask_tar + lamb_bkg * torch.mean((im_s - im_in) * (im_s - im_in)) * mask_bkg
                loss_o = torch.mean((output_t - output_) * (output_t - output_)) * mask_tar

            # loss = loss_i + lamb * loss_o
            if loss_i >= 0.001: # PSNR=30dB
                loss = loss_i
            else:
                loss = loss_o
            
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

            # if i % (args.steps//30) == 0:
            if i % 1000 == 0:    
                # print(torch.mean(mask), torch.mean(att))
                print("step:", i, "overall loss:", loss.item(), "loss_rec:", loss_o.item(), "loss_in:", loss_i.item())
                if i % (args.steps/3) == 0:
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
                    img.save("./attack/kodak/final_in.png")

                    output, _, p_hyper, y_main_q, hyper_dec = image_comp(im_, 2)   
                    p_main, _ = context(y_main_q, hyper_dec)

                    bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                    bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                    bpp = bpp_main + bpp_hyper
                    print("bpp:", bpp.item())
                    output_ = torch.clamp(output, min=0., max=1.0)
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
    img.save("./attack/kodak/final_out.png")

    # original recons
    output_ = torch.clamp(output_s, min=0., max=1.0)
    out = output_.data[0].cpu().numpy()
    out = np.round(out * 255.0)
    out = out.astype('uint8')
    out = out.transpose(1, 2, 0)

    img = Image.fromarray(out[:H, :W, :])
    img.save("./attack/kodak/origin_out.png")

    return 0, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NIC config
    parser.add_argument("-cn", "--ckpt_num", type=int,
                        help="load checkpoint by step number")
    parser.add_argument("-l", "--lamb", type=float,
                        default=6400., help="lambda")
    parser.add_argument("-j", "--job", type=str, default="", help="job name")
    parser.add_argument('--ctx', dest='context', action='store_true')
    parser.add_argument('--no-ctx', dest='context', action='store_false')
    parser.add_argument('--post', dest='post', action='store_true')

    parser.add_argument('-itx',    dest='iter_x', type=int, default=0,          help="iter step updating x")
    parser.add_argument('-ity',    dest='iter_y', type=int, default=0,          help="iter step updating y")
    parser.add_argument('-m',      dest='model',  type=str, default="nlaic", help="compress model in 'factor','hyper','context','nonlocal'")
    parser.add_argument('-metric', dest='metric', type=str, default="ms-ssim",  help="mse or ms-ssim")
    parser.add_argument('-q',      dest='quality',type=int, default="2",        help="quality in [1-8]")
    
    # attack config
    parser.add_argument('-steps',dest='steps',      type=int,   default=10001,  help="attack iteration steps")
    parser.add_argument("-la",  dest="lamb_attack", type=float, default=0.2,    help="attack lambda")
    parser.add_argument("-lr",  dest="lr_attack",   type=float, default=0.001,  help="attack learning rate")
    parser.add_argument("-s",   dest="source",      type=str,   default=None,   help="source input image")
    parser.add_argument("-t",   dest="target",      type=str,   default=None,   help="target image")

    parser.add_argument('--log', dest='log', action='store_true')
    args = parser.parse_args()
    print("============================================================")
    print("[ IMAGE ]:", args.source, "->", args.target)

    checkpoint = "Weights/"
    # print("[CONTEXT]:", args.context)
    print("==== Loading Checkpoint:", checkpoint, '====')
    
    ## random select in 0
    # target = "/ct/code/mnist_png/testing/0/294.png"
    ## random select in [1-9]
    # source = "/home/tong/mnist_png/testing/9/281.png"
    # source = "/home/tong/mnist_png/testing/8/110.png"
    # source = "/home/tong/mnist_png/testing/7/411.png"
    # source = "/home/tong/mnist_png/testing/6/940.png"
    # source = "/home/tong/mnist_png/testing/5/509.png"
    # source = "/home/tong/mnist_png/testing/4/109.png"
    # source = "/home/tong/mnist_png/testing/3/1426.png"
    # source = "/ct/code/mnist_png/testing/2/72.png"
    # source = "/home/tong/mnist_png/testing/1/430.png"

    # target = "/ct/code/LearnedCompression/attack/colorattacker.png"
    # source = "/ct/code/LearnedCompression/attack/colorchecker.png"

    # target = "/ct/code/LearnedCompression/attack/licenseplate/MZ8723_180x180.png" 
    # source = "/ct/code/LearnedCompression/attack/licenseplate/MZ2837_180x180.png"

    # target = "/home/tong/LearnedCompression/mnist/tmp/roof_angle.png"
    # source = "/home/tong/LearnedCompression/mnist/tmp/roof_psnr.png"

    bpp, psnr = attack(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))