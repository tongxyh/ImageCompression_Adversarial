import numpy as np
import math
import torch
# from msssim import MultiScaleSSIM as msssim_
from utils import torch_msssim

def PSNR(im_a, im_b, max_val=1.0):
    mse = (im_a - im_b)**2.0
    mse = mse.mean()
    psnr = 10.0 * torch.log10(max_val*max_val/mse)
    return psnr

def torch_rgb2yuv444(img):
    # input:    torch tensor [N, 3, H, W]
    # output:   torch tensor [N, 3, H, W]
    ycbcr = torch.zeros_like(img, dtype=torch.float32)

    r = img[:, 0, :, :]
    g = img[:, 1, :, :]
    b = img[:, 2, :, :]

    # BT.601
    convert_mat = np.array([[0.299, 0.587, 0.114],
							[-0.1687, -0.3313, 0.5],
							[0.5, -0.4187, -0.0813]], dtype=np.float32)

    ycbcr[:, 0, :, :] = r * convert_mat[0, 0] + g * convert_mat[0, 1] + b * convert_mat[0, 2]
    ycbcr[:, 1, :, :] = r * convert_mat[1, 0] + g * convert_mat[1, 1] + b * convert_mat[1, 2] + 128./255.
    ycbcr[:, 2, :, :] = r * convert_mat[2, 0] + g * convert_mat[2, 1] + b * convert_mat[2, 2] + 128./255.

    return ycbcr

def mse_yuv444(yuv_0, yuv_1):
    psnr_weights = [6.0/8.0, 1.0/8.0, 1.0/8.0]

    mse = 1.
    for i in range(3):
        mse_w = torch.mean((yuv_1[:, i, :, :] - yuv_0[:, i, :, :])**2.0)
        mse = mse * (mse_w ** psnr_weights[i]) # psnr = 10.0 * torch.log10(255.*255./mse) * psnr_weights[i] + psnr
        # print(mse_w)
    return mse

class YUV_MSELoss(torch.nn.Module):
    def __init__(self):
        super(YUV_MSELoss,self).__init__()
    
    def forward(self, im0, im1):
        yuv_0 = torch_rgb2yuv444(im0)
        yuv_1 = torch_rgb2yuv444(im1)
        return mse_yuv444(yuv_0, yuv_1)

# def msssim_yuv444(yuv_0, yuv1):
#     msssim = msssim_(yuv0[:,:,0], yuv1[:,:,0])
#     ## TODO: (u,v)
#     return msssim