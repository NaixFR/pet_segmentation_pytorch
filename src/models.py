import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling with MaxPool then DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upscaling then DoubleConv
    - bilinear=True: use bilinear upsampling + conv
    - bilinear=False: use transposed conv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: from decoder
        x2: skip connection from encoder
        """
        x1 = self.up(x1)

        # padding to handle odd sizes (robust)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                                   diff_y // 2, diff_y - diff_y // 2])

        # concat along channel dim
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation
    Input: (B, 3, H, W)
    Output: (B, 1, H, W) logits
    """
    def __init__(self, in_channels=3, out_channels=1, base_c=64, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_c = base_c
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16 + base_c * 8, base_c * 8, bilinear)
        self.up2 = Up(base_c * 8 + base_c * 4, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4 + base_c * 2, base_c * 2, bilinear)
        self.up4 = Up(base_c * 2 + base_c, base_c, bilinear)

        self.outc = OutConv(base_c, out_channels)

    def forward(self, x):
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # quick test
    model = UNet(base_c=32)  # lighter, faster
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("output shape:", y.shape)  # should be (2,1,256,256)
