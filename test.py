from glob import glob

import numpy as np
import torch
from self_ensemble import defend
torch.manual_seed(0)

import coder
from train import RateDistortionLoss, AverageMeter

@torch.no_grad()
def test(args, net, crop=None):
    criterion = RateDistortionLoss()
    im, _, _ = coder.read_image(args.source)
    im = im.to(args.device)
    if args.defend:
        print("ssldjfak")
        _, x, output_, likelihood = defend(net, im)
        result = net(x)
        result['x_hat'] = output_
        result['likelihoods'] = likelihood
    else:    
        result = net(im)

    if args.target:
        coder.write_image(torch.clamp(result["x_hat"], min=0.0, max=1.0), args.target)
    return criterion(result, im, training=False)

if __name__ == "__main__":
    parser = coder.config()
    args = parser.parse_args()
    print("[Evaluate RD]:", args.source)
    net = coder.load_model(args, training=False).to(args.device)
    criterion = RateDistortionLoss()
    bpp, psnr, msim, msim_dB = [AverageMeter() for i in range(4)]
    images = glob(args.source)
    for image in images:
        # args.source = image
        im_ori = coder.read_image(image)[0].to(args.device)        
        if args.defend:
            with torch.no_grad():
                im, _, _ = coder.read_image(image)
                im = im.to(args.device)
                _, x, output_, likelihood = defend(net, im)
                result = net(x)
                result['x_hat'] = output_
                result['likelihoods'] = likelihood            
        else:
            result = coder.code(args, net, image, args.target)
        metrics = criterion(result, im_ori, training=False)
        if args.debug:
            print(metrics)
        # print(metrics["bpp_loss"], metrics["mse_loss"], metrics["psnr"], metrics["msim_loss"], metrics["msim_dB"])
        bpp.update(metrics["bpp_loss"])
        psnr.update(metrics["psnr"])
        msim.update(metrics["msim_loss"])
        msim_dB.update(metrics["msim_dB"])
    print("AVG:", bpp.avg.item(), psnr.avg, msim.avg.item(), msim_dB.avg)