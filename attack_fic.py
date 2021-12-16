import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

import model_clic
import torch_msssim
from fast_context_model import Context4
import ops, coder

def main():
    dev_id = "cuda:0"
    args = coder.config()
    base_string = "029_0.00212_0.30019.pth"
    image_string = "./quant_image_" + base_string
    context_string = "./quant_context_" + base_string

    image_comp = model_clic.Image_coding(3, 32, 192, 42, 64).to(dev_id)
    image_comp.load_state_dict(torch.load(image_string))
    image_comp.eval()

    context = Context4().to(dev_id)
    context.load_state_dict(torch.load(context_string))
    context.eval()

    loss_func = torch_msssim.MS_SSIM(max_val=1.0).to(dev_id)

    img_name = args.source
    filename = args.source.split("/")[-1][:-4]
    img_array = np.array(Image.open(img_name))
    height, width, channel = img_array.shape
    
    input_data = torch.FloatTensor(img_array.transpose([2, 0, 1]).astype(float))
    im_s = Variable(input_data / 255.0, volatile=False).to(dev_id).view(1, channel, height, width)
    H, W = im_s.shape[2], im_s.shape[3]
    num_pixels = H * W

    # noise = torch.rand(im_s.size()) - 0.5
    noise = torch.zeros(im_s.size())
    noise = noise.to(dev_id).requires_grad_(True) # set requires_grad=True after moving tensor to device
    optimizer = torch.optim.Adam([noise],lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1,2,3], gamma=0.33, last_epoch=-1)
    noise_range = 0.5
    for i in range(args.steps):
        # clip noise range
        # noise_clipped = torch.clamp(mask*noise, min=-noise_range, max=noise_range)
        noise_clipped = ops.Up_bound.apply(ops.Low_bound.apply(noise, -noise_range), noise_range)
        # im_in = torch.clamp(im_s+noise_clipped, min=0., max=1.0)
        im_in = ops.Up_bound.apply(ops.Low_bound.apply(im_s+noise_clipped, 0.), 1.)
        im_in[:,:,H:,:] = 0.
        im_in[:,:,:,W:] = 0.
        
        x1, x2 = image_comp.encoder(im_in)

        fake = image_comp.decoder(x1)
        output_ = ops.Up_bound.apply(ops.Low_bound.apply(fake, 0.), 1.)

        loss_i = torch.mean((im_s - im_in) * (im_s - im_in))
        loss_o = 1. - torch.mean((im_s - output_) * (im_s - output_))
        
        if loss_i >= args.noise:
            loss = loss_i
        else:
            loss = loss_o

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:    
            print(i, "loss_rec", loss_o.item(), "loss_in", loss_i.item())

        if i % (args.steps//3) == 0:
            print("step:", i, "loss_rec:", loss_o.item(), "loss_in:", loss_i.item())
            lr_scheduler.step()
            with torch.no_grad():
                im_uint8 = torch.round(im_in * 255.0)/255.0
                
                # 1. NO PADDING
                # im_uint8[:,:,H:,:] = 0.
                # im_uint8[:,:,:,W:] = 0.
                
                # save adverserial input
                im_ =  torch.clamp(im_uint8, min=0., max=1.0)
                fin = im_.data[0].cpu().numpy()
                fin = np.round(fin * 255.0)
                fin = fin.astype('uint8')
                fin = fin.transpose(1, 2, 0)
                img = Image.fromarray(fin[:H, :W, :])
          
                img.save("./attack/kodak/fake%d_in_%0.8f.png"%(i, loss.item())) 
                img.save("./attack/kodak/final_in.png")

                #output, y_main, y_hyper, p_main, p_hyper = image_comp(im_, False, CONTEXT, POSTPROCESS)   
                x1, x2 = image_comp.encoder(im_)
                xq2, p_hyper = image_comp.factorized_entropy_func(x2, False)
                hyper_dec = image_comp.hyper_dec(xq2)

                xq1 = model_clic.RoundNoGradient.apply(x1)
                print("min: %.2f  |  max: %.2f" % (xq1.min(), xq1.max()))

                xp1 = image_comp.gaussin_entropy_func(xq1, hyper_dec)
                p_main, _ = context(xq1, hyper_dec)
                output = image_comp.decoder(x1)
        
                bpp_hyper = torch.sum(torch.log(p_hyper)) / (-np.log(2.) * num_pixels)
                bpp_main = torch.sum(torch.log(p_main)) / (-np.log(2.) * num_pixels)
                bpp = bpp_main + bpp_hyper
                print("bpp:", bpp.item())
                output_ = torch.clamp(output, min=0., max=1.0)
                out = output_.data[0].cpu().numpy()
                out = np.round(out * 255.0)
                out = out.astype('uint8')
                out = out.transpose(1, 2, 0)
                
                img = Image.fromarray(out[:H, :W, :])
                # img.save("./attack/kodak/fake%d_out_%0.4f_%0.8f.png"%(i, bpp.item(), loss.item()))
                img.save("./attack/kodak/%s_fake%d_out_%0.4f_%0.8f.png"%(filename, i, bpp.item(), loss.item()))
        # psnr = get_psnr(255 * origin.cpu().numpy(), 255 * fake.cpu().numpy())
        # print("%.3f \t| %.3f/%.2f \t| %.2f" % (bpp, msssim, -10 * np.log10(1 - msssim), psnr))


def get_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * np.log10(255.0 ** 2 / mse)


if __name__ == "__main__":
    main()

