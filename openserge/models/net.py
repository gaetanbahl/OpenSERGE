from typing import Tuple, Dict, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from ..utils.utils import build_grid

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    """Residual block: Conv -> BN -> ReLU -> Conv -> BN -> Add residual -> ReLU"""
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(c_in, c_out, k, padding=p, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, k, padding=p, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)

        # Residual connection (1x1 conv if dimensions don't match)
        if c_in != c_out:
            self.residual = nn.Conv2d(c_in, c_out, 1, bias=False)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        identity = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class JunctionOffsetHead(nn.Module):
    """Three-branch head: junctionness (1 ch), offset (2 ch), node features (Nfeat)."""
    def __init__(self, c_in: int, nfeat: int = 256):
        super().__init__()
        # Each branch: initial projection + residual block + output conv
        self.jun = nn.Sequential(
            ConvBNReLU(c_in, nfeat),
            ResidualBlock(nfeat, nfeat),
            nn.Conv2d(nfeat, 1, 1)
        )
        self.off = nn.Sequential(
            ConvBNReLU(c_in, nfeat),
            ResidualBlock(nfeat, nfeat),
            nn.Conv2d(nfeat, 2, 1),
            nn.Tanh()
        )
        self.nfe = nn.Sequential(
            ConvBNReLU(c_in, nfeat),
            ResidualBlock(nfeat, nfeat),
            nn.Conv2d(nfeat, nfeat, 1)
        )

    def forward(self, f):
        j = self.jun(f)  # [B,1,H',W']
        o = 0.5 * self.off(f)  # constrain to [-0.5, 0.5]
        n = self.nfe(f)  # [B,Nfeat,H',W']
        return j, o, n

class Backbone(nn.Module):
    def __init__(self, name: str = 'resnet50', pretrained: bool = False):
        super().__init__()
        if name == 'resnet50':
            m = resnet50(weights=None if not pretrained else 'DEFAULT')
            c_out = 2048
        elif name == 'resnet18':
            m = resnet18(weights=None if not pretrained else 'DEFAULT')
            c_out = 512
        else:
            raise ValueError('Unsupported backbone')
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.body = nn.Sequential(m.layer1, m.layer2, m.layer3, m.layer4)
        self.c_out = c_out
        self.stride = 32  # typical for ResNet-50
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        return x  # [B, C, H/32, W/32]

class SingleShotRoadGraphNet(nn.Module):
    """Backbone + 3-branch head as in the paper."""
    def __init__(self, backbone='resnet50', nfeat=256):
        super().__init__()
        self.backbone = Backbone(backbone)
        self.head = JunctionOffsetHead(self.backbone.c_out, nfeat)
    def forward(self, x):
        f = self.backbone(x)
        j, o, n = self.head(f)
        return {'junction_logits': j, 'offset': o, 'node_feats_map': n, 'stride': self.backbone.stride}
