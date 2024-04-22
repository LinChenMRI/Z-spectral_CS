import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c_in, c_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_out),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampling(nn.Module):
    def __init__(self, channels):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.Down(x)


class UpSampling(nn.Module):
    def __init__(self, channels):
        super(UpSampling, self).__init__()
        self.Up = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=2, padding=1)
        )
        self.Up_add_one = nn.Sequential(
            nn.ConvTranspose1d(in_channels=channels, out_channels=channels // 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x, r):
        res = x
        x = self.Up(x)
        if x.shape[2] != r.shape[2]:
            x = self.Up_add_one(res)
        return torch.cat((x, r), dim=1)


class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()

        # 4 downsampling
        self.conv1 = Conv(1, 64)
        self.Down1 = DownSampling(64)
        self.conv2 = Conv(64, 128)
        self.Down2 = DownSampling(128)
        self.conv3 = Conv(128, 256)
        self.Down3 = DownSampling(256)
        self.conv4 = Conv(256, 512)
        self.Down4 = DownSampling(512)
        self.conv5 = Conv(512, 1024)

        # 4 upsampling
        self.Up1 = UpSampling(1024)
        self.conv6 = Conv(1024, 512)
        self.Up2 = UpSampling(512)
        self.conv7 = Conv(512, 256)
        self.Up3 = UpSampling(256)
        self.conv8 = Conv(256, 128)
        self.Up4 = UpSampling(128)
        self.conv9 = Conv(128, 64)

        self.th = torch.nn.Sigmoid()
        self.predict = torch.nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        r1 = self.conv1(x)
        r2 = self.conv2(self.Down1(r1))
        r3 = self.conv3(self.Down2(r2))
        r4 = self.conv4(self.Down3(r3))
        r5 = self.conv5(self.Down4(r4))

        o1 = self.conv6(self.Up1(r5, r4))
        o2 = self.conv7(self.Up2(o1, r3))
        o3 = self.conv8(self.Up3(o2, r2))
        o4 = self.conv9(self.Up4(o3, r1))
        print(o4.shape)

        return self.th(self.predict(o4))