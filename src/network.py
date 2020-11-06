import torch
import torch.nn as nn


def get_same_padding(kernel_size):
    return int((kernel_size - 1) / 2)


class DownModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2):
        super(DownModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=get_same_padding(kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=get_same_padding(kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, scale_factor=2, last=False):
        super(UpModule, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=get_same_padding(kernel_size)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        if last:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=get_same_padding(kernel_size)),
                nn.Tanh()
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding=get_same_padding(kernel_size)),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class InpaintNetwork(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cnum: int = 32):
        super(InpaintNetwork, self).__init__()
        self.model = nn.Sequential(
            DownModule(in_channels, cnum),
            DownModule(cnum, 2*cnum),
            DownModule(2*cnum, 4*cnum),
            DownModule(4*cnum, 8*cnum),
            DownModule(8*cnum, 8*cnum),
            DownModule(8*cnum, 8*cnum),

            UpModule(8*cnum, 8*cnum),
            UpModule(8*cnum, 8*cnum),
            UpModule(8*cnum, 4*cnum),
            UpModule(4*cnum, 2*cnum),
            UpModule(2*cnum, cnum),
            UpModule(cnum, out_channels, last=True)
        )

    def forward(self, x):
        return self.model(x)




