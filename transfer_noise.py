from glob import glob

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

torch.manual_seed(0)

import coder
from train import RateDistortionLoss
from attack_rd import attack_

@torch.no_grad()
def test(net, image, noise, crop=None):
    # criterion = RateDistortionLoss()
    im, H, W = coder.read_image(image)
    im = im.to(noise.device)
    output_s = net(im)["x_hat"]
    im_ = torch.clamp(im+noise, min=0., max=1.)
    mse_in = torch.mean((im_ - im)**2)
    output_adv = net(im_)["x_hat"]
    if args.debug:
        coder.write_image(im_, "transfer_in.png", H, W)
        coder.write_image(output_adv, "transfer_out_noclamp.png", H, W)
        coder.write_image(torch.clamp(output_adv, min=0., max=1.), "transfer_out.png", H, W)
    # return criterion(result, im, training=False)
    return mse_in, torch.clamp(output_adv, min=0., max=1.), torch.clamp(output_s, min=0., max=1.)

if __name__ == "__main__":
    parser = coder.config()
    parser.add_argument("-s2", "--source2", type=str, default="", help="source image")
    args = parser.parse_args()
    
    print("[Noise Transfer]:", args.source2, 'to' ,args.source)
    
    net = coder.load_model(args, training=False).to(args.device)
    # myattacker = attacker(args)

    # generate noise on source image A
    # bpp_ori, bpp, vi = myattacker.attack(args.source2)
    images_ = glob(args.source2)
    vis = torch.zeros(24,24)
    print(vis.shape)
    for i, source in enumerate(images_):
        im_s, H, W = coder.read_image(source)
        im_s = im_s.to(args.device)
        im_adv, output_adv, output_s, _, _, _, _, vi = attack_(im_s, net, args)
        noise = im_adv - im_s
        # mse_in = torch.mean(noise**2)
        # print(vi)
        if args.debug:
            coder.write_image(im_adv, "before_transfer_in.png")
            coder.write_image(output_adv, "before_transfer_out.png")
            coder.write_image(noise+0.5, "noise.png")
        # add noise to source image B
        images = glob(args.source)
        for j, image in enumerate(images):
            mse_in, output_adv, output_s = test(net, image, noise)
            mse_out = torch.mean((output_adv - output_s)**2)
            print(10*math.log10(mse_out/mse_in))
            vis[i,j] = 10*math.log10(mse_out/mse_in)
            # if args.debug:
            #     print(metrics)
        # print(metrics["bpp_loss"], metrics["mse_loss"], metrics["psnr"], metrics["msim_loss"], metrics["msim_dB"])
    torch.save(vis, "transfer.pkl")
    # vis = torch.load("transfer.pkl")
    # # plt.show(vis)
    # fig, ax = plt.subplots()
    # im = ax.imshow(vis)
    # for i in range(24):
    #     for j in range(24):
    #         text = ax.text(j, i, int(vis[i, j].item()),
    #                     ha="center", va="center", color="w", fontsize="xx-small")
    # plt.savefig("transfer.pdf")
    # print(vis)