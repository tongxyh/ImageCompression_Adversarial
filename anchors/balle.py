import sys
import math
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import compressai
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018, cheng2020_anchor
from anchors.utils import update_registered_buffers


class Image_coder(torch.nn.Module):
    def __init__(self, MODEL, quality, metric, pretrained=True):
        super(Image_coder, self).__init__()
        self.MODEL = MODEL
        if MODEL == "factorized":
            self.net = bmshj2018_factorized(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "hyper":
            self.net = bmshj2018_hyperprior(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "context":
            self.net = mbt2018(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "cheng2020":
            self.net = cheng2020_anchor(quality=quality, metric=metric, pretrained=pretrained)

    def forward(self, x, TRAINING, CONTEXT, POSTPROCESS):
        if TRAINING:
            self.net.train()
        else:
            self.net.eval()

        y = self.net.g_a(x)
        if self.MODEL == "factorized":
            y_hat, y_likelihoods = self.net.entropy_bottleneck(y)
            z_hat, z_likelihoods = 0, torch.Tensor([1.0])
            

        if self.MODEL == "hyper":
            z = self.net.h_a(torch.abs(y))
            z_hat, z_likelihoods = self.net.entropy_bottleneck(z)
            scales_hat = self.net.h_s(z_hat)
            y_hat, y_likelihoods = self.net.gaussian_conditional(y, scales_hat)

        if self.MODEL == "context" or self.MODEL == "cheng2020":
            z = self.net.h_a(y)
            z_hat, z_likelihoods = self.net.entropy_bottleneck(z)
            params = self.net.h_s(z_hat)

            y_hat = self.net.gaussian_conditional.quantize(y, "noise" if self.net.training else "dequantize")
            ctx_params = self.net.context_prediction(y_hat)
            gaussian_params = self.net.entropy_parameters(torch.cat((params, ctx_params), dim=1))
            scales_hat, means_hat = gaussian_params.chunk(2, 1)
            _, y_likelihoods = self.net.gaussian_conditional(y, scales_hat, means=means_hat)

        x_hat = self.net.g_s(y_hat)        
        return x_hat, y, z_hat, y_likelihoods, z_likelihoods

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.net.entropy_bottleneck,
            "net.entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        if self.MODEL == "hyper" or self.MODEL == "context" or self.MODEL == "cheng2020":
            update_registered_buffers(
                self.net.gaussian_conditional,
                "net.gaussian_conditional",
                ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                state_dict,
            )  
        super().load_state_dict(state_dict)
# TEST
# MODEL = sys.argv[2] #factorized, hyper, context
# quality = int(sys.argv[3])
# img = Image.open(sys.argv[1]).convert('RGB')
# x = transforms.ToTensor()(img).unsqueeze(0)

# image_comp = Image_coder(MODEL, quality)
# print(MODEL, quality)

# rec, y_main, y_hyper, p_main, p_hyper = image_comp(x, False)

# # save rec
# out = torch.clamp(rec, min=0., max=1.0)
# out = out.data[0].cpu().numpy()
# out = np.round(out * 255.0)
# out = out.astype('uint8')
# out = out.transpose(1, 2, 0)

# # img = Image.fromarray(out[PADDING:H+PADDING, PADDING:W+PADDING, :])
# img = Image.fromarray(out)
# img.save("./rec.png")

# pixels = x.size()[0] * x.size()[2] * x.size()[3]
# bits = torch.sum(torch.log(p_main)/-math.log(2)) + torch.sum(torch.log(p_hyper)/-math.log(2))
# print("bpp:", bits.item()/pixels)
# # PSNR
# print("PSNR:")