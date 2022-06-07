from glob import glob

import numpy as np
import torch
import math

torch.manual_seed(0)

import coder
from anchors.utils import layer_compare
 

def test(args, image, net, crop=None):
    im, H, W = coder.read_image(image)

    ## generate random input noise
    scale = args.noise ** 0.5
    m = torch.distributions.normal.Normal(0.0, scale)
    # m = torch.distributions.uniform.Uniform(-args.noise, args.noise)

    noise = m.sample(im.size()).cuda()
    if args.debug:
        print("Noise level (L2):", torch.mean(noise ** 2))
    im = im.to(args.device)
    im_in = torch.clamp(im+noise, min=0., max=1.)

    ## save random noised input image
    # filename = args.source.split("/")[-1][:-4]
    # write_image(im_in, H, W, "./attack/random_noise_1e-3/"+filename+".png")    

    with torch.no_grad():
        x_hat = torch.clamp(net(im_in)["x_hat"], min=0., max=1.0)
        x_hat_ori = torch.clamp(net(im)["x_hat"], min=0., max=1.0)

        err_out = torch.mean((x_hat_ori - x_hat)**2).item()
        if args.debug:
            layer_compare(net, im_in, im)
            print("MSE (noise):", torch.mean(noise**2).item())
            print("MSE (out_ori - in_ori):", torch.mean((x_hat_ori-im)**2).item())
            print("MSE (out_adv - in_ori):", torch.mean((x_hat-im)**2).item())
            print("MSE (out_adv - out_ori):", err_out)
        if args.target:
            coder.write_image(x_hat, args.target, H, W)
    return 10.*math.log10(err_out/torch.mean(noise ** 2))

if __name__ == "__main__":
    parser = coder.config()
    args = parser.parse_args()
    print("[RANDOM NOISE]:", args.source)
    if args.quality > 0:
        net = coder.load_model(args, training=False).to(args.device)
        images = glob(args.source)
        print('AVG', sum(test(args, image, net, crop=None) for image in images)/len(images))
    else:
        if args.model == "cheng2020":
            quality_range = 7
        else:
            quality_range = 9
        for quality in range(1, quality_range):
            for noise in [1e-5,1e-4,1e-3,1e-2]:
                args.quality = quality
                args.noise = noise
                net = coder.load_model(args, training=False).to(args.device)
                images = glob(args.source)
                print("AVG", quality, noise, sum(test(args, image, net, crop=None) for image in images)/len(images))
