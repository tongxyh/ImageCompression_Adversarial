import torch
import torch.nn as nn
from compressai.zoo import bmshj2018_factorized, bmshj2018_hyperprior, mbt2018, cheng2020_anchor
from compressai.models import CompressionModel, FactorizedPrior
from compressai.layers import GDN
from .utils import conv, deconv

class ae_onelayer(FactorizedPrior):
    # Autoencoder with only 1 conv layer for both encoder & decoder
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, 3),
        )

class balle_relu(FactorizedPrior):
    # Autoencoder with only 1 conv layer for both encoder & decoder
    def __init__(self, N, M, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, N),
            nn.LeakyReLU(inplace=True),
            conv(N,N),
            nn.LeakyReLU(inplace=True),
            conv(N,N),
            nn.LeakyReLU(inplace=True),
            conv(N,M)
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            nn.LeakyReLU(inplace=True),
            deconv(N,N),
            nn.LeakyReLU(inplace=True),
            deconv(N,N),
            nn.LeakyReLU(inplace=True),
            deconv(N,3)
        )

def init_model(MODEL, quality, metric, pretrained=True):
        if MODEL == "debug":
            assert not pretrained, "No download-able model available!"
            if quality <= 5:
                net = balle_relu(N=128, M=192)
            else:
                net = balle_relu(N=192, M=320)
        if MODEL == "factorized":
            net = bmshj2018_factorized(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "hyper":
            net = bmshj2018_hyperprior(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "context":
            net = mbt2018(quality=quality, metric=metric, pretrained=pretrained)
        if MODEL == "cheng2020":
            net = cheng2020_anchor(quality=quality, metric=metric, pretrained=pretrained)
        return net

def compressor(x, net, MODEL):
    y = net.g_a(x)
    y_hat, z_hat, entropys = entropy_estimator(y, net, MODEL)
    x_hat = net.g_s(y_hat)        
    return {"x_hat":x_hat, "y_hat":y_hat, "z_hat":z_hat, "likelihoods":entropys}

def entropy_estimator(y, net, MODEL):
    if MODEL == "factorized":
        y_hat, y_likelihoods = net.entropy_bottleneck(y)
        z_hat, z_likelihoods = 0, torch.Tensor([1.0])
        
    if MODEL == "hyper":
        z = net.h_a(torch.abs(y))
        z_hat, z_likelihoods = net.entropy_bottleneck(z)
        scales_hat = net.h_s(z_hat)
        y_hat, y_likelihoods = net.gaussian_conditional(y, scales_hat)

    if MODEL == "context" or MODEL == "cheng2020":
        z = net.h_a(y)
        z_hat, z_likelihoods = net.entropy_bottleneck(z)
        params = net.h_s(z_hat)

        y_hat = net.gaussian_conditional.quantize(y, "noise" if net.training else "dequantize")
        ctx_params = net.context_prediction(y_hat)
        gaussian_params = net.entropy_parameters(torch.cat((params, ctx_params), dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = net.gaussian_conditional(y, scales_hat, means=means_hat)

    return y_hat, z_hat, {"y":y_likelihoods, "z":z_likelihoods}    

def probe(x, net, name="y_hat", MODEL="hyper"):
    if name == "y_hat":
        return net.g_a(x)
    if name == "z_hat":
        return net.h_a(net.g_a(x))
    if name == "scales_hat":
        if MODEL == "hyper":
            z_hat = net.h_a(net.g_a(x))
        return net.h_s(z_hat)
    if name == "means_hat":
        if MODEL == "context" or MODEL == "cheng2020":
            y = net.g_a(x)
            z = net.h_a(y)
            y_hat = net.gaussian_conditional.quantize(y, "noise" if net.training else "dequantize")
            z_hat, _ = net.entropy_bottleneck(z)
            params = net.h_s(z_hat)
            ctx_params = net.context_prediction(y_hat)
            gaussian_params = net.entropy_parameters(torch.cat((params, ctx_params), dim=1))
            _, means_hat = gaussian_params.chunk(2, 1)
            return means_hat
        else:
            return None