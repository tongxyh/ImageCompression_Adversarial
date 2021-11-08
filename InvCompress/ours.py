import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from .priors import JointAutoregressiveHierarchicalPriors
from .our_utils import *
from compressai.layers import *
from compressai.entropy_models import EntropyBottleneck
from .waseda import Cheng2020Anchor

class InvCompress(Cheng2020Anchor):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N)
        self.g_a = None
        self.g_s = None
        self.enh = EnhModule(64)
        self.inv = InvComp(M=N)
        self.attention = AttModule(N)
        N_hyper = 768
        self.context_prediction = MaskedConv2d(
            N_hyper, 2 * N_hyper, kernel_size=5, padding=2, stride=1
        )
        self.entropy_bottleneck = EntropyBottleneck(N_hyper)
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(N_hyper * 12 // 3, N_hyper * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N_hyper * 10 // 3, N_hyper * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N_hyper * 8 // 3, N_hyper * 6 // 3, 1),
        )
        self.h_a = nn.Sequential(
            conv3x3(N_hyper, N_hyper),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper, N_hyper),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper, N_hyper, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper, N_hyper),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper, N_hyper, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N_hyper, N_hyper),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N_hyper, N_hyper, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper, N_hyper * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N_hyper * 3 // 2, N_hyper * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N_hyper * 3 // 2, N_hyper * 2),
        )


    def g_a_func(self, x):
        # x = self.enh(x)
        x = self.inv(x)
        # x = self.attention(x)
        return x

    def g_s_func(self, x):
        # x = self.attention(x, rev = True)
        x = self.inv(x, rev=True)
        # x = self.enh(x, rev=True)
        return x

    def forward(self, x):
        y = self.g_a_func(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s_func(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        # N = state_dict["h_a.0.weight"].size(0)
        net = cls(N=128)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a_func(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:], "y": y}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = torch.zeros(
            (z_hat.size(0), 768, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        x_hat = self.g_s_func(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

