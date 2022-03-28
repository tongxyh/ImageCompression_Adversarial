import torch.nn as nn

from compressai.models import CompressModel, FactorizedPrior
from compressai.layers import GDN
from utils import conv, deconv, GDN


class ae_onelayer(FactorizedPrior):
    # Autoencoder with only 1 conv layer for both encoder & decoder
    def __init__(self, N, M, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)
        self.g_a = nn.Sequential(
            conv(3, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, 3),
        )