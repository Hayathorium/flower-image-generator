import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, z_dim, 1, 1)
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (B, 3, 64, 64)
            nn.utils.spectral_norm(nn.Conv2d(3, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1)),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, True),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0))
        )
    def forward(self, x):
        return self.net(x)