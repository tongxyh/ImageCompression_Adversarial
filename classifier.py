import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# https://docs.google.com/presentation/d/1TVixw6ItiZ8igjp6U17tcgoFrLSaHWQmMOwjlgQY9co/pub?slide=id.g1245051c73_0_2920
class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # TODO: dropout/convolution
        self.f1 = nn.Linear(3*784,200)
        self.act1 = nn.ReLU()
        self.f2 = nn.Linear(200,100)
        self.act2 = nn.ReLU()
        self.f3 = nn.Linear(100,60)
        self.act3 = nn.ReLU()
        self.f4 = nn.Linear(60,30)
        self.act4 = nn.ReLU()
        self.f5 = nn.Linear(30,10)
        self.act5 = nn.Softmax(dim=1)

    def forward(self,x):
        x1 = self.act1(self.f1(x))
        x2 = self.act2(self.f2(x1))
        x3 = self.act3(self.f3(x2))
        x4 = self.act4(self.f4(x3))
        return self.act5(self.f5(x4))