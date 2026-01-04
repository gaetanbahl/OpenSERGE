from typing import Tuple, Dict, Optional
import torch, torch.nn as nn, torch.nn.functional as F
import timm
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
    """
    Feature extraction backbone using timm library.

    Supports CNNs (ResNet, EfficientNet, etc.) and Vision Transformers.
    - CNNs: Uses features_only=True for FPN compatibility (multi-scale features)
    - ViTs: Reshapes patch tokens to spatial feature map (no FPN support)
    """
    def __init__(self, name: str = 'resnet50', use_fpn: bool = False):
        super().__init__()
        self.name = name
        self.use_fpn = use_fpn
        self.is_vit = 'vit' in name.lower()

        # Validate model exists and check if pretrained weights are available
        pretrained = self._check_pretrained_available(name)

        # Store normalization parameters for later use
        self.normalize_mean = None
        self.normalize_std = None

        if self.is_vit and use_fpn:
            raise ValueError("FPN is not supported with Vision Transformers. Set use_fpn=False.")

        if use_fpn:
            # Create feature extractor with multi-scale outputs
            # We need the last 4 feature levels at strides [4, 8, 16, 32]
            # First create with default indices to see how many levels the model has
            temp_model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True
            )

            # Get all available feature reductions
            all_reductions = temp_model.feature_info.reduction()

            # Select indices for strides [4, 8, 16, 32]
            # For most CNNs these are the last 4 features
            # For ResNets: [2, 4, 8, 16, 32] → we want indices (1, 2, 3, 4)
            # For EfficientNets: [2, 4, 8, 16, 32] → we want indices (1, 2, 3, 4)
            target_strides = [4, 8, 16, 32]
            out_indices = []

            for stride in target_strides:
                if stride in all_reductions:
                    out_indices.append(all_reductions.index(stride))
                else:
                    # If exact stride not found, try to find closest
                    # This shouldn't happen for standard models
                    raise ValueError(
                        f"Model {name} doesn't have feature at stride {stride}. "
                        f"Available strides: {all_reductions}"
                    )

            # Now create model with correct indices
            self.model = timm.create_model(
                name,
                pretrained=pretrained,
                features_only=True,
                out_indices=tuple(out_indices)
            )

            # Get feature channel information from the model
            fpn_channels = self.model.feature_info.channels()
            feature_strides = self.model.feature_info.reduction()

            # Verify we got 4 feature levels at correct strides
            if len(fpn_channels) != 4:
                raise ValueError(f"Model {name} returned {len(fpn_channels)} features, expected 4. "
                               f"Channels: {fpn_channels}")

            if feature_strides != target_strides:
                raise ValueError(f"Model {name} feature strides {feature_strides} don't match "
                               f"expected {target_strides}")

            # Create FPN to aggregate multi-scale features
            self.fpn = FPN(fpn_channels, out_channels=256)
            self.c_out = 256

        else:
            # Create single-scale feature extractor (last layer only)
            # For ViT, enable dynamic image size to support 512x512 inputs
            model_kwargs = {
                'pretrained': pretrained,
                'num_classes': 0,  # Remove classifier
                'global_pool': ''  # Remove global pooling
            }

            # Enable dynamic image size for Vision Transformers
            if self.is_vit:
                model_kwargs['dynamic_img_size'] = True

            self.model = timm.create_model(name, **model_kwargs)

            if self.is_vit:
                # Vision Transformer handling
                # ViT outputs [B, num_patches, embed_dim] where num_patches = (H/patch_size) * (W/patch_size)
                # We need to reshape to [B, embed_dim, H/patch_size, W/patch_size]

                # Get patch size and embed_dim from model
                if hasattr(self.model, 'patch_embed'):
                    self.patch_size = self.model.patch_embed.patch_size[0]
                    self.embed_dim = self.model.embed_dim
                else:
                    # Fallback for different ViT implementations
                    raise ValueError(f"Could not determine patch_size for ViT model {name}")

                # For 512x512 input with patch_size=16, we get 32x32 patches (stride 16)
                # We want stride 32, so we need to downsample by 2x
                # Add a conv layer to go from stride 16 -> stride 32
                self.vit_stride = self.patch_size  # e.g., 16 for patch16 models

                if self.vit_stride == 16:
                    # Downsample 2x to get stride 32
                    self.vit_projection = nn.Sequential(
                        nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
                    self.c_out = 256
                    self.stride = 32
                elif self.vit_stride == 32:
                    # Already at stride 32, just project channels
                    self.vit_projection = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
                    self.c_out = 256
                    self.stride = 32
                elif self.vit_stride == 14:
                    # DINOv2 models with patch14
                    # 512/14 ≈ 36.57, so we get ~37x37 patches
                    # Downsample to get closer to stride 32
                    self.vit_projection = nn.Sequential(
                        nn.Conv2d(self.embed_dim, 256, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True)
                    )
                    self.c_out = 256
                    self.stride = self.vit_stride  # Keep original stride
                else:
                    # Generic case: just project channels
                    self.vit_projection = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
                    self.c_out = 256
                    self.stride = self.vit_stride

                print(f"ViT backbone: patch_size={self.patch_size}, embed_dim={self.embed_dim}, "
                      f"output_stride={self.stride}, output_channels={self.c_out}")
            else:
                # CNN handling
                # Get output channels
                # For models with feature_info, use it; otherwise infer from model
                if hasattr(self.model, 'num_features'):
                    self.c_out = self.model.num_features
                elif hasattr(self.model, 'feature_info'):
                    self.c_out = self.model.feature_info.channels()[-1]
                else:
                    # Fallback: run a dummy forward pass to get output shape
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        dummy_output = self.model(dummy_input)
                        self.c_out = dummy_output.shape[1]

                self.stride = 32  # Output is always at stride 32 for CNNs

        # Extract normalization parameters from pretrained_cfg
        if hasattr(self.model, 'pretrained_cfg') and self.model.pretrained_cfg:
            self.normalize_mean = self.model.pretrained_cfg.get('mean', (0.485, 0.456, 0.406))
            self.normalize_std = self.model.pretrained_cfg.get('std', (0.229, 0.224, 0.225))
            print(f"Using normalization: mean={self.normalize_mean}, std={self.normalize_std}")
        else:
            # Default: no normalization (just [0, 1] scaling)
            self.normalize_mean = (0.0, 0.0, 0.0)
            self.normalize_std = (1.0, 1.0, 1.0)
            print(f"No pretrained_cfg found, using [0,1] scaling (no normalization)")

    def _check_pretrained_available(self, model_name: str) -> bool:
        """
        Check if model exists in timm and if pretrained weights are available.

        Returns:
            True if pretrained weights available, False otherwise

        Raises:
            ValueError if model doesn't exist at all
        """
        # Try to create model to check if it exists
        # This is the most reliable way since timm.list_models() uses patterns
        try:
            # Try with pretrained first
            timm.create_model(model_name, pretrained=True, num_classes=0)
            print(f"Loading {model_name} with pretrained weights")
            return True
        except RuntimeError as e:
            # Pretrained weights not available, try without
            if 'pretrained' in str(e).lower() or 'weight' in str(e).lower():
                try:
                    timm.create_model(model_name, pretrained=False, num_classes=0)
                    print(f"Loading {model_name} without pretrained weights (not available)")
                    return False
                except Exception:
                    pass
            # Model doesn't exist at all
            # Try to find similar models for helpful error message
            similar = timm.list_models(f'{model_name.split(".")[0]}*')[:5]
            if similar:
                raise ValueError(
                    f"Model '{model_name}' not found in timm. "
                    f"Did you mean one of these? {similar}"
                )
            else:
                raise ValueError(
                    f"Model '{model_name}' not found in timm. "
                    f"Use `timm.list_models()` to see available models, "
                    f"or check https://huggingface.co/timm"
                )
        except Exception as e:
            # Some other error - model likely doesn't exist
            similar = timm.list_models(f'{model_name.split(".")[0]}*')[:5]
            if similar:
                raise ValueError(
                    f"Model '{model_name}' not found or failed to load: {e}. "
                    f"Did you mean one of these? {similar}"
                )
            else:
                raise ValueError(
                    f"Model '{model_name}' not found in timm: {e}"
                )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Feature map [B, C, H/stride, W/stride]
        """
        B, _, H, W = x.shape

        if self.use_fpn:
            # Get multi-scale features
            features = self.model(x)  # List of [B, C_i, H_i, W_i]

            # Verify we got 4 features
            if len(features) != 4:
                raise RuntimeError(f"Expected 4 feature levels, got {len(features)}")

            # Aggregate through FPN
            x = self.fpn(features)
        else:
            # Single-scale forward pass
            x = self.model(x)

            if self.is_vit:
                # ViT outputs [B, num_patches+1, embed_dim] or [B, num_patches, embed_dim]
                # The +1 is for the [CLS] token which we need to remove

                if x.dim() == 3:
                    # [B, N, C] format (sequence of patches)
                    # Some models have CLS token, register tokens, or other prefix/suffix tokens
                    num_patches = (H // self.patch_size) * (W // self.patch_size)
                    N = x.shape[1]

                    if N == num_patches:
                        # No extra tokens
                        pass
                    elif N == num_patches + 1:
                        # Has 1 extra token (CLS), remove it
                        x = x[:, 1:, :]  # [B, num_patches, C]
                    elif N > num_patches:
                        # Has multiple extra tokens (CLS + registers, etc.)
                        # Assume extra tokens are at the beginning, take last num_patches
                        num_extra = N - num_patches
                        x = x[:, num_extra:, :]  # [B, num_patches, C]
                        print(f"Note: ViT has {num_extra} extra tokens (CLS/registers), removing them")
                    else:
                        # Fewer patches than expected
                        raise RuntimeError(
                            f"Expected at least {num_patches} patches, "
                            f"got {N} for input {H}x{W} with patch_size={self.patch_size}"
                        )

                    # Reshape from [B, num_patches, C] to [B, C, H_p, W_p]
                    h_p = H // self.patch_size
                    w_p = W // self.patch_size
                    x = x.transpose(1, 2).reshape(B, self.embed_dim, h_p, w_p)

                elif x.dim() == 4:
                    # Already in [B, C, H, W] format (some ViT variants)
                    pass
                else:
                    raise RuntimeError(f"Unexpected ViT output shape: {x.shape}")

                # Apply projection layer to get to desired stride and channels
                x = self.vit_projection(x)

        return x  # [B, C, H/stride, W/stride]

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
