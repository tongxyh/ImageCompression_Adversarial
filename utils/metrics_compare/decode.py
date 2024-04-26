#!/usr/bin/env python3

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
import os
import model_clic
from glob import glob
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.contrib.coder.python.ops import coder_ops
import torch.nn.functional as F
import tensorflow.contrib.eager as tfe
import struct
tf.enable_eager_execution()

image_comp = model_clic.Image_coding(3,192,128,192,32)
image_comp.load_state_dict(torch.load(r'test1.pkl'))
os.system('mv ./encoded_files/* ./')
for i in glob('*h.dc'): 
  with open(i, "rb") as f:
    raw = f.read()
    H,W,min_v,max_v = struct.unpack('4i',raw[-16:])
    print ("Infor:",H,W,min_v,max_v)
    H_PAD = int(64.0 * np.ceil(H / 64.0))
    W_PAD = int(64.0 * np.ceil(W / 64.0))
    a = np.reshape(np.arange(min_v, max_v + 1), [1, 1, max_v - min_v + 1])
    a = np.tile(a, [128, 1, 1])
    ## using factorized model
    with torch.no_grad():
        lower = image_comp.factorized_entropy_func._logits_cumulative(torch.Tensor(a) - 0.5,
                                                                         stop_gradient=False)
        upper = image_comp.factorized_entropy_func._logits_cumulative(torch.Tensor(a) + 0.5,
                                                                         stop_gradient=False)
        sign = -torch.sign(torch.add(lower, upper))
        sign = sign.detach()
        p2 = torch.abs(F.sigmoid(sign * upper) - F.sigmoid(sign * lower))
    #print(p2)

    ## using range coder
    cdf2 = coder_ops.pmf_to_quantized_cdf(p2.numpy(), 16)
    cdf2 = tf.reshape(cdf2, [1, 128, -1])
    print (cdf2.shape)
    print (H_PAD,W_PAD)
    print (H_PAD//64*W_PAD//64, H_PAD//64, W_PAD//64)
    decoded = coder_ops.range_decode(raw[:-16], (H_PAD//64*W_PAD//64,128), cdf2, precision=16) + min_v
    decoded = tf.cast(decoded, dtype=tf.float32)
    xq2 = tf.reshape(decoded, [1, H_PAD//64, W_PAD//64, 128])
    print (decoded)

    # decode for main feature
    xq2 = torch.Tensor(xq2.numpy().transpose(0,3,1,2))
    with torch.no_grad():
        x3 = image_comp.hyper_dec(xq2)
        hyper_dec = image_comp.p(x3)

    with open(i[:-4]+'m.dc', "rb") as f:
        raw = f.read()
        min_v,max_v = struct.unpack('2i',raw[-8:])

        mean = hyper_dec[:, :192, :, :]
        scale1 = hyper_dec[:, 192:, :, :]

        N1, C1, H1, W1 = mean.size()

        scale1 = scale1.data[0].numpy().transpose(1, 2, 0)
        mean = mean.data[0].numpy().transpose(1, 2, 0)
        mean = tf.reshape(mean, [ H1, W1, C1, 1])
        scale1 = tf.reshape(scale1, [ H1, W1, C1, 1])

        arr = np.reshape(np.arange(min_v, max_v + 1), [ 1, 1, 1, max_v - min_v + 1])
        arr = np.tile(arr, [ H1, W1, C1, 1])

        upper = arr + 0.5
        lower = arr - 0.5
        print(mean.shape, scale1.shape, arr.shape)

        m1 = torch.distributions.normal.Normal(torch.Tensor(mean.numpy()), torch.Tensor(scale1.numpy()))
        p1 = torch.abs(m1.cdf(torch.Tensor(upper)) - m1.cdf(torch.Tensor(lower)))
        print(p1.size())

        cdf1 = coder_ops.pmf_to_quantized_cdf(p1.cpu().numpy(), 16)

        decoded = coder_ops.range_decode(raw[:-8], (H1,W1,192), cdf1, precision=16) + min_v
        decoded = tf.reshape(decoded, [N1, H1, W1, 192])
        decoded = tf.cast(decoded, dtype=tf.float32)
        print (decoded.shape)
        xq1 = torch.Tensor(decoded.numpy().transpose(0,3,1,2))
        with torch.no_grad():
            output = image_comp.decoder(xq1)
        output = torch.clamp(output, min=0, max=1.0)
        output = output.data[0].numpy()
        output = output.transpose(1, 2, 0) * 255.0
        output = output.astype('uint8')
        img = Image.fromarray(output[:H, :W, :])
        img.save('./decoded_images/' + i[:-4] + '.png')
