"""
VAE.py
VAE模型定义
input: VAEConfig
"""

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data
from config.ModelConfig import VAEConfig


class VAE(nn.Module):
    def __init__(self, conf: VAEConfig):
        super(VAE, self).__init__()
        self.conf = conf

        # Encoder
        self.econv1 = nn.Conv2d(3, conf.L1, kernel_size=conf.KernelSize, stride=conf.Stride)
        self.ebn1 = nn.BatchNorm2d(conf.L1, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.econv2 = nn.Conv2d(conf.L1, conf.L2, kernel_size=conf.KernelSize, stride=conf.Stride)
        self.ebn2 = nn.BatchNorm2d(conf.L2, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.econv3 = nn.Conv2d(conf.L2, conf.L3, kernel_size=conf.KernelSize, stride=conf.Stride)
        self.ebn3 = nn.BatchNorm2d(conf.L3, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.econv4 = nn.Conv2d(conf.L3, conf.L4, kernel_size=conf.KernelSize, stride=conf.Stride)
        self.ebn4 = nn.BatchNorm2d(conf.L4, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.efc1 = nn.Linear(conf.L4 * conf.FeatureRow * conf.FeatureCol, conf.L5)
        self.edrop1 = nn.Dropout(p=0.3, inplace=False)
        self.mu_z = nn.Linear(conf.L5, conf.LatentDim)
        self.logvar_z = nn.Linear(conf.L5, conf.LatentDim)
        # Decoder
        self.dfc1 = nn.Linear(conf.LatentDim, conf.L5)
        self.dfc2 = nn.Linear(conf.L5, conf.L4 * conf.FeatureRow * conf.FeatureCol)
        self.ddrop1 = nn.Dropout(p=0.3, inplace=False)
        self.dconv1 = nn.ConvTranspose2d(conf.L4, conf.L3, kernel_size=conf.KernelSize, stride=conf.Stride, padding=0,
                                         output_padding=0)
        self.dbn1 = nn.BatchNorm2d(conf.L3, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv2 = nn.ConvTranspose2d(conf.L3, conf.L2, kernel_size=conf.KernelSize, stride=conf.Stride, padding=0,
                                         output_padding=0)
        self.dbn2 = nn.BatchNorm2d(conf.L2, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv3 = nn.ConvTranspose2d(conf.L2, conf.L1, kernel_size=conf.KernelSize, stride=conf.Stride, padding=0,
                                         output_padding=1)
        self.dbn3 = nn.BatchNorm2d(conf.L1, eps=conf.EPS, momentum=0.1, affine=True, track_running_stats=True)
        self.dconv4 = nn.ConvTranspose2d(conf.L1, 3, kernel_size=conf.KernelSize, padding=0, stride=conf.Stride)
        #
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def encoder(self, x):
        eh1 = self.relu(self.ebn1(self.econv1(x)))
        eh2 = self.relu(self.ebn2(self.econv2(eh1)))
        eh3 = self.relu(self.ebn3(self.econv3(eh2)))
        eh4 = self.relu(self.ebn4(self.econv4(eh3)))
        eh5 = self.relu(self.edrop1(self.efc1(eh4.view(-1, self.conf.L4 * self.conf.FeatureRow * self.conf.FeatureCol))))
        mu_z = self.mu_z(eh5)
        logvar_z = self.logvar_z(eh5)
        return mu_z, logvar_z

    def reparam(self, mu_z, logvar_z):
        std = logvar_z.mul(0.5).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        eps = eps.to(self.conf.Device)
        return eps.mul(std).add_(mu_z)

    def decoder(self, z):
        dh1 = self.relu(self.dfc1(z))
        dh2 = self.relu(self.ddrop1(self.dfc2(dh1)))
        dh3 = self.relu(self.dbn1(self.dconv1(dh2.view(-1, self.conf.L4, self.conf.FeatureRow, self.conf.FeatureCol))))
        dh4 = self.relu(self.dbn2(self.dconv2(dh3)))
        dh5 = self.relu(self.dbn3(self.dconv3(dh4)))
        x = self.dconv4(dh5).view(-1, 3, self.conf.ImageSize)
        return self.sigmoid(x)

    def forward(self, x):
        mu_z, logvar_z = self.encoder(x)
        z = self.reparam(mu_z, logvar_z)
        return self.decoder(z), mu_z, logvar_z, z