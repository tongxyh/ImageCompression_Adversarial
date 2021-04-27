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
from utils import torch_msssim
from anchors import balle
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

def test(args, checkpoint_dir, CONTEXT=True, POSTPROCESS=True, crop=None):

    TRAINING = True
    GPU = True
    # read image
    precise = 16
    # print('====> Encoding Image:', im_dir)

    # Writer will output to ./runs/ directory by default
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    tensor_log_path = './logs'
    if not os.path.exists(tensor_log_path):
        os.mkdir(tensor_log_path)
    writer = SummaryWriter(tensor_log_path+'/'+TIMESTAMP)
    # writer = SummaryWriter("./logs/")

    # img_s = Image.open(source_dir).resize((16,16))
    img_s = Image.open(args.source)
    # img_s = np.array(img_s)/255.0/5.0+0.5
    img_s = np.array(img_s)/255.0

    if len(img_s.shape) < 3:
        H, W = img_s.shape
        img_s = np.tile(img_s.reshape((H,W,1)), (1,1,3))
    else:
        H, W, _ = img_s.shape

    # img_t = Image.open(target_dir).resize((64,64))
    # img_t = Image.open(target_dir)
    # img_t = np.array(img_t)/255.0/10.0+0.5
    # img_t = np.array(img_t)/255.0

    # if len(img_t.shape) < 3:
    #     H, W = img_t.shape
    #     img_t = np.tile(img_t.reshape((H,W,1)), (1,1,3))
    # else:
    #     H, W, _ = img_t.shape

    num_pixels = H * W

    C = 3
    if crop == None:
        tile = 64.
    else:
        tile = crop * 1.0

    H_PAD = int(tile * np.ceil(H / tile))
    W_PAD = int(tile * np.ceil(W / tile))
    PADDING = 0
    im_s = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    im_s[PADDING:H+PADDING, PADDING:W+PADDING, :] = img_s[:, :, :3]
    im_s = torch.FloatTensor(im_s)

    # im_t = np.zeros([H_PAD, W_PAD, 3], dtype='float32')
    # im_t[PADDING:H+PADDING, PADDING:W+PADDING, :] = img_t[:, :, :3]
    # im_t = torch.FloatTensor(im_t)

    # model initalization
    MODEL = args.model
    quality = args.quality
    if MODEL == "nonlocal":
        image_comp = model.ImageCompression(256)
        image_comp.load_state_dict(torch.load(checkpoint_dir), strict=False)
        # image_comp.load_state_dict(torch.load(checkpoint_dir).state_dict())
        # torch.save(image_comp.state_dict(), "./checkpoints/elic-0.0.1/ae.pkl")
        image_comp.eval()
    if MODEL in ["factorized", "hyper", "context"]:
        image_comp = balle.Image_coder(MODEL, quality=quality, metric=args.metric)
        print(MODEL, quality, args.metric)
    # Gradient Mask
    gnet = Gradient_Net().cuda()
    if GPU:
        image_comp.cuda()
        #msssim_func = msssim_func.cuda()
        im_s = im_s.cuda()
        if args.target != None:
            im_t = im_t.cuda()

    im_s = im_s.permute(2, 0, 1).contiguous()
    im_s = im_s.view(1, C, H_PAD, W_PAD)
    if args.target != None:
        im_t = im_t.permute(2, 0, 1).contiguous()
        im_t = im_t.view(1, C, H_PAD, W_PAD)

    mssim_func = torch_msssim.MS_SSIM(max_val=1).cuda()
    with torch.no_grad():
        mask = gnet(im_s)
        output_s, y_main_, y_hyper, p_main, p_hyper = image_comp(im_s, False, CONTEXT, POSTPROCESS)  
        ori_bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
        ori_bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
        print("Original bpp:", ori_bpp_hyper + ori_bpp_main)
        # print("Original PSNR:", )
        # print("Original MS-SSIM:", )
    
    if crop == None:
        # rate
        lamb = args.lamb_attack
        LOSS_FUNC = "L2"
        print("using loss func:", LOSS_FUNC)
        print("Lambda:", lamb)
        # im = im_s.clone().detach().requires_grad_(True) + torch.randn(im_s.size()).cuda()
        noise = torch.randn(im_s.size())/10.0
        noise = noise.cuda().requires_grad_(True) # set requires_grad=True after moving tensor to device
        optimizer = torch.optim.Adam([noise],lr=1e-3)
        # im = (im_s+noise/10.0).clone().detach().requires_grad_(True)

        # im = im_s.clone().detach().requires_grad_(True)
        # optimizer = torch.optim.Adam([im],lr=1e-3)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
        for i in range(args.steps):
            
            im_in = torch.clamp(im_s+mask*noise, min=0., max=1.0)
            
            # 1. NO PADDING
            im_in[:,:,H:,:] = 0.
            im_in[:,:,:,W:] = 0.
            
            output, y_main, y_hyper, p_main, p_hyper = image_comp(im_in, TRAINING, CONTEXT, POSTPROCESS)
            output_ = torch.clamp(output, min=0., max=1.0)

            # mse_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(y_t, y_s)

            if LOSS_FUNC == "L2":
                # loss_o = torch.mean((output_s - output_) * (output_s - output_)) # MSE(y_s, y_t)
                loss_o = torch.mean((im_in - output_) * (im_in - output_)) # MSE(x_t, y_t)
                loss_i = torch.mean((im_s - im_in) * (im_s - im_in))
                loss = loss_i + lamb * (1-loss_o)
            
            # L1 loss
            if LOSS_FUNC == "L1":
                loss_o = torch.mean(torch.abs(im_s - output_))
                loss_i = torch.mean(torch.abs(im_s - im_in))
                loss = loss_i + lamb * (1-loss_o)

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

            writer.add_scalar('Loss/mse_all', loss.item(), i)
            # writer.add_scalar('Loss/mse_in', mse_i.item(), i)
            # writer.add_scalar('Loss/mse_out',  mse_o.item(), i)
            # writer.add_scalar('Loss/bpp',  bpp.item(), i)

            if i % (args.steps//3) == 0:
                print("step:", i, "overall loss:", loss.item(), "loss_rec:", loss_o.item(), "loss_in:", loss_i.item())
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
                    
                    img.save("./attack/kodak/fake%d_in_%0.8f.png"%(i, loss.item())) 
                    img.save("./attack/kodak/final_in.png")
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
                    out = output_.data[0].cpu().numpy()
                    out = np.round(out * 255.0)
                    out = out.astype('uint8')
                    out = out.transpose(1, 2, 0)
                    
                    # img = Image.fromarray(out[PADDING:H+PADDING, PADDING:W+PADDING, :])
                    img = Image.fromarray(out[:H, :W, :])
                    img.save("./attack/kodak/fake%d_out_%0.4f_%0.8f.png"%(i, bpp.item(), loss.item()))

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
    parser.add_argument('-m',      dest='model',  type=str, default="nonlocal", help="compress model in 'factor','hyper','context','nonlocal'")
    parser.add_argument('-metric', dest='metric', type=str, default="ms-ssim",  help="mse or ms-ssim")
    parser.add_argument('-q',      dest='quality',type=int, default="2",        help="quality in [1-8]")
    # attack config
    parser.add_argument('-step',dest='steps',       type=int,   default=10001,  help="attack iteration steps")
    parser.add_argument("-la",  dest="lamb_attack", type=float, default=0.2,    help="attack lambda")
    parser.add_argument("-s",   dest="source",      type=str,   default=None,   help="source input image")
    parser.add_argument("-t",   dest="target",      type=str,   default=None,   help="target image")

    args = parser.parse_args()

    checkpoints = glob('./ckpts/%d_%s/ae_%d_*' %(int(args.lamb), args.job, args.ckpt_num))
    if args.model == "nonlocal":
        print("CONTEXT:", args.context)
        print("==== Loading Checkpoint:", checkpoint, '====')
    # random select in 0
    # target = "/home/tong/LearnedCompression/mnist/tmp/roof_angle.png"
    # target = "/ct/code/mnist_png/testing/0/294.png"
    # target = "/ct/code/LearnedCompression/attack/colorattacker.png"
    # target = "/ct/code/LearnedCompression/attack/licenseplate/MZ8723_180x180.png"

    # random select in [1-9]
    # source = "/home/tong/mnist_png/testing/9/281.png"
    # source = "/home/tong/mnist_png/testing/8/110.png"
    # source = "/home/tong/mnist_png/testing/7/411.png"
    # source = "/home/tong/mnist_png/testing/6/940.png"
    # source = "/home/tong/mnist_png/testing/5/509.png"
    # source = "/home/tong/mnist_png/testing/4/109.png"
    # source = "/home/tong/mnist_png/testing/3/1426.png"
    # source = "/ct/code/mnist_png/testing/2/72.png"
    # source = "/home/tong/mnist_png/testing/1/430.png"
    # source = "/ct/code/LearnedCompression/attack/colorchecker.png"
    # source = "/ct/code/LearnedCompression/attack/licenseplate/MZ2837_180x180.png"
    # source = "/home/tong/LearnedCompression/mnist/tmp/roof_psnr.png"
    for checkpoint in checkpoints:
        bpp, psnr = test(args, checkpoint, CONTEXT=args.context, POSTPROCESS=args.post, crop=None)
    # print(checkpoint, "bpps:%0.4f, psnr:%0.4f" %(bpp, psnr))
