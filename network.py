import math
import torch
import torch.nn as nn

def conv1d_out_len(L_in, kernel_size, stride=1, padding=1, dilation=1):
    return math.floor((L_in + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)

class MixedConvBlock(nn.Module):
    def __init__(self, C_in: int, k1d: int, s1d: int, k2d: int = 3, padding1d: int = 1):
        super().__init__()
        self.C_in = C_in
        self.k1d = k1d
        self.s1d = s1d
        self.p1d = padding1d

        C_out = conv1d_out_len(C_in, kernel_size=k1d, stride=s1d, padding=padding1d, dilation=1)
        if C_out <= 0:
            raise ValueError(f"Invalid spectral length after 1D conv: C_in={C_in}, k={k1d}, s={s1d}, p={padding1d}")
        self.C_out = C_out

        self.spectral_conv = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=k1d, stride=s1d, padding=padding1d, bias=False
        )
        self.spectral_bn = nn.BatchNorm1d(1)
        self.spectral_act = nn.ReLU(inplace=True)

        self.depthwise_conv = nn.Conv2d(
            in_channels=C_out, out_channels=C_out,
            kernel_size=k2d, padding=k2d // 2,
            groups=C_out, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(C_out)
        self.depthwise_act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.size()
        assert C == self.C_in, f"Expected input channels {self.C_in}, got {C}"

        spec = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        spec = self.spectral_conv(spec)
        spec = self.spectral_bn(spec)
        spec = self.spectral_act(spec)
        C_out = spec.size(-1)
        spec = spec.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()

        out = self.depthwise_conv(spec)
        out = self.depthwise_bn(out)
        out = self.depthwise_act(out)
        return out


class NET(nn.Module):
    def __init__(self, band=184, num_classes=4, pw_out=128):

        super(NET, self).__init__()
        cfg = [
            {"k1d": 3, "s1d": 1},
            {"k1d": 5, "s1d": 2},
            {"k1d": 3, "s1d": 1},
            {"k1d": 3, "s1d": 1},
        ]

        C1_in = band
        C1_out = conv1d_out_len(C1_in, kernel_size=cfg[0]["k1d"], stride=cfg[0]["s1d"], padding=1)
        C2_out = conv1d_out_len(C1_out, kernel_size=cfg[1]["k1d"], stride=cfg[1]["s1d"], padding=1)
        C3_out = conv1d_out_len(C2_out, kernel_size=cfg[2]["k1d"], stride=cfg[2]["s1d"], padding=1)
        C4_out = conv1d_out_len(C3_out, kernel_size=cfg[3]["k1d"], stride=cfg[3]["s1d"], padding=1)

        self.block1 = MixedConvBlock(C_in=C1_in, k1d=cfg[0]["k1d"], s1d=cfg[0]["s1d"])
        self.bn1 = nn.BatchNorm2d(C1_out); self.act1 = nn.ReLU(inplace=True)

        self.block2 = MixedConvBlock(C_in=C1_out, k1d=cfg[1]["k1d"], s1d=cfg[1]["s1d"])
        self.bn2 = nn.BatchNorm2d(C2_out); self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3 = MixedConvBlock(C_in=C2_out, k1d=cfg[2]["k1d"], s1d=cfg[2]["s1d"])
        self.bn3 = nn.BatchNorm2d(C3_out); self.act3 = nn.ReLU(inplace=True)

        self.block4 = MixedConvBlock(C_in=C3_out, k1d=cfg[3]["k1d"], s1d=cfg[3]["s1d"])
        self.bn4 = nn.BatchNorm2d(C4_out); self.act4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.pointwise = nn.Conv2d(C4_out, pw_out, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(pw_out)
        self.pw_act = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))

        self.classifier = nn.Sequential(
            nn.Linear(pw_out * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x); x = self.bn1(x); x = self.act1(x)
        x = self.block2(x); x = self.bn2(x); x = self.act2(x); x = self.pool2(x)
        x = self.block3(x); x = self.bn3(x); x = self.act3(x)
        x = self.block4(x); x = self.bn4(x); x = self.act4(x); x = self.pool4(x)


        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.pw_act(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
