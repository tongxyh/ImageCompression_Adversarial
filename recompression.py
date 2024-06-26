import os
import sys
from glob import glob
import time
import numpy as np
import math
import torch
from pytorch_msssim import ms_ssim

from utils.metrics import PSNR
import coder
import train
from self_ensemble import defend


# python recompression.py -s /workspace/ct/datasets/kodak/ -m hyper -q 3 -metric ms-ssim --download -steps 10
# python recompression.py -s /workspace/ct/datasets/kodak/ -m hyper -q 3 -metric ms-ssim -ckpt ./ckpts/attack/hyper-3/ae_199_0.08826580_0.44019613.pkl -steps 10

@torch.no_grad()
def test(args, repeat_times=50):
    # print(args.source)
    images = sorted(glob(args.source))
    # print(images)
    model = coder.load_model(args, training=False).to(args.device)
    criterion = train.RateDistortionLoss()
    bpps, psnrs, sims, sims_dB = [np.zeros((repeat_times, len(images))) for i in range(4)]
    for j, source in enumerate(images):
        print("[IMAGE]", source)
        index = source.split("/")[-1][:-4]
        ori = source
        im_ori = coder.read_image(ori)[0].to(args.device)
        for i in range(repeat_times):
            target = "./attack/kodak/{}_{}_out{}.png".format(args.log, index, i)
            if args.defend:
                im, _, _ = coder.read_image(source)
                im = im.to(args.device)
                _, x, output_, likelihood = defend(model, im)
                result = model(x)
                result['x_hat'] = output_
                result['likelihoods'] = likelihood 
                coder.write_image(torch.clamp(result["x_hat"], min=0.0, max=1.0), target)
            else:  
                result = coder.code(args, model, source, target)
            source = target
                
        ## only calculate result for the last repeat
        metrics = criterion(result, im_ori, training=False)
        print(metrics["bpp_loss"].item(), PSNR(torch.clamp(result["x_hat"], min=0, max=1), im_ori).item(), ms_ssim(torch.clamp(result["x_hat"], min=0, max=1), im_ori, data_range=1.0, size_average=True).item(), "Result")
        bpps[i, j], psnrs[i, j], sims[i,j] = metrics["bpp_loss"].item(), PSNR(torch.clamp(result["x_hat"], min=0, max=1), im_ori).item(), ms_ssim(torch.clamp(result["x_hat"], min=0, max=1), im_ori, data_range=1.0, size_average=True).item()
        sims_dB[i, j] = -10.0*math.log10(1 - sims[i,j])
            
            # print(bpp.item())
            # if args.download:
            #     # cmd = "python visual.py -m {} -metric {} -q {} -s {} -t {} --download > /dev/null 2>&1".format(args.model, args.metric, args.quality, source, target)
            # else:
            #     cmd = "python visual.py -m {} -metric {} -q {} -s {} -t {} -ckpt {}".format(args.model, args.metric, args.quality, source, target, args.ckpt)
            # os.system(cmd)
            # cmd = "python /workspace/ct/code/NIC/Util/CLIC/compare.py {} {}".format(ori, target)
            # os.system(cmd)
            
    # for i in range(repeat_times):
    #     print(i, np.mean(bpps, axis=1)[i], np.mean(psnrs, axis=1)[i], np.mean(sims, axis=1)[i], np.mean(sims_dB, axis=1)[i])
    print("Final:", np.mean(bpps, axis=1)[-1], np.mean(psnrs, axis=1)[-1], np.mean(sims, axis=1)[-1], np.mean(sims_dB, axis=1)[-1])

if __name__ == "__main__":
    args = coder.config().parse_args()
    if args.defend:
        print("Self Ensemble Applied!")
    test(args, repeat_times=args.steps)