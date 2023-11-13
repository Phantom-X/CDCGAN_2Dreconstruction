"""
@project:CDCGAN_2Dreconstruction
@Author: Phantom
@Time:2023/11/13 下午1:24
@Email: 2909981736@qq.com
"""
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.bn0 = nn.BatchNorm2d(102)
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.deconv1 = nn.ConvTranspose2d(102, 256, kernel_size=8, stride=1)  # out = s*in+k-2p-s
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 1, kernel_size=5, stride=2, padding=2, output_padding=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = self.bn0(x)
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.deconv4(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.bn0 = nn.BatchNorm2d(3)
        self.leaky_relu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=15, stride=1, padding=3)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)

    def forward(self, x):
        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        return x


if __name__ == '__main__':
    # g = Generator()
    # x = torch.rand([16, 102, 1, 1])
    # out = g(x)
    # print(out.shape)

    d = Discriminator()
    x = torch.rand([16, 3, 64, 64])
    out = d(x)
    print(out.shape)
