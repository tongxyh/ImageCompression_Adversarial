import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

from glob import glob
import math
import coder
from anchors.utils import layer_compare
# from utils.motion_blur.blur_image import BlurImage 
# from utils.motion_blur.generate_PSF import PSF
# from utils.motion_blur.generate_trajectory import Trajectory
import torchvision.transforms as T


@torch.no_grad()
def test_deblur(image_blur, image_sharp, net, args, crop=None):
    print(image_blur, image_sharp)
    im_sharp = coder.read_image(image_sharp)[0].to(args.device)
    
    if image_blur is None:
        # params = [0.01, 0.009, 0.008, 0.007, 0.005, 0.003]
        # trajectory = Trajectory(canvas=64, max_len=60, expl=np.random.choice(params)).fit()
        # psf = PSF(canvas=64, trajectory=trajectory).fit() # point-separate function
        # im_blur = BlurImage(image_sharp, PSFs=psf, part=np.random.choice([1, 2, 3])).blur_image(save=False).result[0]
        # im_blur = torch.FloatTensor(im_blur).permute(2, 0, 1).contiguous().view(im_sharp.shape)

        blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.5))
        
        im_blur = blurrer(im_sharp)
        print(torch.mean((im_blur-im_sharp)**2))
    else:
        im_blur  = coder.read_image(image_blur)[0].to(args.device)

    num_pixels = im_blur.shape[2] * im_blur.shape[3]
    results = net(im_blur)
    bpp = sum(
                 (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                 for likelihoods in results["likelihoods"].values())
    y_blur = results["x_hat"]
    print(torch.mean((im_blur - im_sharp)**2))
    psnr_blur = - 10 * math.log10(torch.mean((im_blur - im_sharp)**2))
    psnr_sharp = - 10 * math.log10(torch.mean((y_blur - im_sharp)**2))
    print(psnr_blur, psnr_sharp)
    return psnr_blur - psnr_sharp, bpp.item(), psnr_sharp

def generate_blurimages(images):
    sigma = 1.0
    # blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma))
    for image in images:
        sigma = 5.0
        blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma))
        im_sharp = coder.read_image(image)[0].to(args.device)
        im_blur = blurrer(im_sharp)
        im_blur = torch.clamp(im_blur, min=0., max=1.)
        while torch.mean((im_blur-im_sharp)**2) > args.noise*1.01:
            sigma = sigma - 0.005
            blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma))
            im_blur = blurrer(im_sharp)
            im_blur = torch.clamp(im_blur, min=0., max=1.)
        print(sigma, torch.mean((im_blur-im_sharp)**2))
        coder.write_image(im_blur, "./attack/kodak/blur/%s.png" % (image.split("/")[-1].split(".")[0]))
        
    
def test(args, image, net, crop=None):
    im, H, W = coder.read_image(image)
    num_pixels = H * W
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
        results_ori = net(im)
        results = net(im_in)
        
        x_hat = torch.clamp(results["x_hat"], min=0., max=1.0)
        x_hat_ori = torch.clamp(results_ori["x_hat"], min=0., max=1.0)

        bpp_ori = sum(
                 (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                 for likelihoods in results_ori["likelihoods"].values()
             )
        bpp = sum(
                 (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                 for likelihoods in results["likelihoods"].values()
             )
        
        err_out = torch.mean((x_hat_ori - x_hat)**2).item()
        if args.debug:
            layer_compare(net, im_in, im)
            print("MSE (noise):", torch.mean(noise**2).item())
            print("MSE (out_ori - in_ori):", torch.mean((x_hat_ori-im)**2).item())
            print("MSE (out_adv - in_ori):", torch.mean((x_hat-im)**2).item())
            print("MSE (out_adv - out_ori):", err_out)
        if args.target:
            coder.write_image(x_hat, args.target, H, W)
    return 10.*math.log10(err_out/torch.mean(noise ** 2)), bpp.item(), -10.*math.log10(torch.mean((x_hat-im)**2).item())

def main(args):
    net = coder.load_model(args, training=False).to(args.device)
    images = sorted(glob(args.source))
    dpsnr_avg, bpp_avg, psnr_avg = 0., 0., 0.
    if args.degrade == "deblur":
        images_blur = images
        images_sharp = sorted(glob(args.target))
        if images_blur == []:
            images_blur = [None for _ in images_sharp]
            # sources = images_sharp
        else:
            pass
            # sources = zip(images_blur, images_sharp)
        for image_blur, image_sharp in zip(images_blur, images_sharp):
            dpsnr, bpp, psnr = test_deblur(image_blur, image_sharp, net, args)

            dpsnr_avg += dpsnr
            bpp_avg += bpp
            psnr_avg += psnr
        print('AVG', args.quality, dpsnr/len(images_sharp), bpp_avg/len(images_sharp), psnr_avg/len(images_sharp))
    else:
        for image in images:
            dpsnr, bpp, psnr = test(args, image, net, crop=None)
            dpsnr_avg += dpsnr
            bpp_avg += bpp
            psnr_avg += psnr
        print('AVG', args.quality, args.noise, dpsnr_avg/len(images_sharp), bpp_avg/len(images_sharp), psnr_avg/len(images_sharp))
            
if __name__ == "__main__":
    parser = coder.config()
    args = parser.parse_args()
    print(args.source)
    if args.source == "None":
        images = sorted(glob(args.target))
        generate_blurimages(images)
    else:
        print("[RANDOM NOISE]:", args.source)
        if args.quality > 0:
            main(args)
        else:
            if args.model == "cheng2020":
                quality_range = 7
            else:
                quality_range = 9
            for quality in range(1, quality_range):
                args.quality = quality
                if args.degrade == "deblur":
                    main(args)
                else:
                    for noise in [1e-5,1e-4,1e-3,1e-2]:
                        args.noise = noise
                        main(args)
