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

class FPN(nn.Module):
    """Feature Pyramid Network that aggregates multi-level features into a single output at stride 32."""
    def __init__(self, in_channels_list, out_channels=256):
        """
        Args:
            in_channels_list: List of channel counts for each level [C2, C3, C4, C5]
            out_channels: Output channel count for all FPN levels
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral connections (1x1 convs to reduce channels)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])

        # Output convs (3x3 convs after upsampling + addition)
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        """
        Args:
            features: List of feature maps [C2, C3, C4, C5] at strides [4, 8, 16, 32]
        Returns:
            Aggregated feature map at stride 32
        """
        # Apply lateral convolutions
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Top-down pathway with lateral connections
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample higher-level feature and add to current level
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode='bilinear', align_corners=False
            )

        # Apply output convolutions
        outputs = [self.output_convs[i](laterals[i]) for i in range(len(laterals))]

        # Aggregate all levels to stride 32 by downsampling
        # outputs[0] is at stride 4, outputs[1] at stride 8, etc.
        final_size = outputs[-1].shape[-2:]  # Size of stride-32 feature map

        aggregated = outputs[-1]  # Start with stride-32 features
        for i in range(len(outputs) - 1):
            # Downsample lower-level features to stride 32 and add
            aggregated = aggregated + F.interpolate(
                outputs[i], size=final_size, mode='bilinear', align_corners=False
            )

        return aggregated


class Backbone(nn.Module):
    def __init__(self, name: str = 'resnet50', pretrained: bool = False, use_fpn: bool = False):
        super().__init__()
        if name == 'resnet50':
            m = resnet50(weights=None if not pretrained else 'DEFAULT')
            c_out = 2048
            # Channel counts for [layer1, layer2, layer3, layer4]
            fpn_channels = [256, 512, 1024, 2048]
        elif name == 'resnet18':
            m = resnet18(weights=None if not pretrained else 'DEFAULT')
            c_out = 512
            # Channel counts for [layer1, layer2, layer3, layer4]
            fpn_channels = [64, 128, 256, 512]
        else:
            raise ValueError('Unsupported backbone')

        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        # Store layers separately for FPN
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

        self.use_fpn = use_fpn
        if use_fpn:
            self.fpn = FPN(fpn_channels, out_channels=256)
            self.c_out = 256
        else:
            self.c_out = c_out

        self.stride = 32  # Output is always at stride 32

    def forward(self, x):
        x = self.stem(x)

        if self.use_fpn:
            # Extract multi-level features
            c2 = self.layer1(x)   # stride 4
            c3 = self.layer2(c2)  # stride 8
            c4 = self.layer3(c3)  # stride 16
            c5 = self.layer4(c4)  # stride 32

            # Aggregate through FPN
            x = self.fpn([c2, c3, c4, c5])
        else:
            # Standard forward pass
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        return x  # [B, C, H/32, W/32]

class SingleShotRoadGraphNet(nn.Module):
    """Backbone + 3-branch head as in the paper."""
    def __init__(self, backbone='resnet50', nfeat=256, use_fpn=False):
        super().__init__()
        self.backbone = Backbone(backbone, use_fpn=use_fpn)
        self.head = JunctionOffsetHead(self.backbone.c_out, nfeat)
    def forward(self, x):
        f = self.backbone(x)
        j, o, n = self.head(f)
        return {'junction_logits': j, 'offset': o, 'node_feats_map': n, 'stride': self.backbone.stride}
