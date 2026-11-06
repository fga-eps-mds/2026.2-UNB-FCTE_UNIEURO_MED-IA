"""Model architectures from Figure 2 / "Conv-Att model architecture" and "Self-attention".

Multi-input Conv-Att (the paper's proposed model):
  for each of the 3 drawings (clock, cube-copy, trail-making):
    VGG16 (ImageNet-pretrained) conv backbone -> (B, 512, 8, 8) feature map
    1x1 conv projects channels 512 -> D (hidden_dim)
    flatten to (B, 64, D), prepend a learned [CLS] token -> (B, 65, D)
    N stacked self-attention (transformer encoder) blocks, post-norm as in Fig. 2
    take the [CLS] position of the last block as that task's image representation
  concatenate the 3 representations -> Linear(3*D, 2) -> softmax

Also included: MultiInputVGGBaseline, the paper's baseline (VGG16 + global-average-pool
per tower, concat, FC+softmax) used for Table 1's VGG16 rows and as the model Grad-CAM
is applied to in Figure 1. Both single- and multi-input variants share `num_inputs`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

try:
    import timm
except ImportError:  # only needed for non-VGG16 backbones (e.g. convnextv2_*)
    timm = None


class VGGBackbone(nn.Module):
    """VGG16 conv feature extractor (no FC head), ImageNet-pretrained. 256x256 in -> (B,512,8,8)."""

    out_channels = 512

    def __init__(self, pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = vgg16(weights=weights).features  # (B,3,H,W) -> (B,512,H/32,W/32)
        if not train_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class TimmBackbone(nn.Module):
    """Any timm feature extractor (e.g. convnextv2_tiny), ImageNet-pretrained. Used in
    place of VGGBackbone when `model.backbone` names a non-"vgg16" architecture
    (IMPROVEMENTS.md's "different backbone" idea). 256x256 in -> (B,C,8,8) for the
    convnextv2_* family (same /32 downsampling as VGG16), C depends on variant size."""

    def __init__(self, name: str, pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for backbone != 'vgg16' (pip install timm)")
        self.model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(-1,))
        self.out_channels = self.model.feature_info.channels()[-1]
        if not train_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[-1]


def build_backbone(name: str, pretrained: bool = True, train_backbone: bool = True) -> nn.Module:
    """Factory: 'vgg16' (the paper's backbone) or any timm model name (e.g. convnextv2_tiny)."""
    if name == "vgg16":
        return VGGBackbone(pretrained=pretrained, train_backbone=train_backbone)
    return TimmBackbone(name, pretrained=pretrained, train_backbone=train_backbone)


class MultiHeadSelfAttention(nn.Module):
    """Attn(Q,K,V) = softmax(QK^T / sqrt(d_head)) V, returning weights for attention rollout."""

    def __init__(self, dim: int, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # (B, heads, N, N)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out), attn


class TransformerEncoderBlock(nn.Module):
    """Post-norm block: MHA -> Add&Norm -> FeedForward -> Add&Norm (matches Fig. 2)."""

    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_w = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x, attn_w


class SelfAttentionTower(nn.Module):
    """One task's CNN-backbone + self-attention stack, producing a single [CLS] vector."""

    def __init__(
        self,
        backbone: str = "vgg16",
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        train_backbone: bool = True,
        use_pos_embed: bool = False,
        num_patches: int = 64,
    ):
        super().__init__()
        self.backbone = build_backbone(backbone, pretrained=pretrained, train_backbone=train_backbone)
        self.proj = nn.Conv2d(self.backbone.out_channels, hidden_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            # Not in the paper: opt-in addition, see IMPROVEMENTS.md.
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, hidden_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(hidden_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        feat = self.backbone(x)  # (B, 512, H, W)
        feat = self.proj(feat)  # (B, D, H, W)
        B, D, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B, HW, D)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, HW+1, D)
        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
        attns = []
        for block in self.blocks:
            tokens, attn_w = block(tokens)
            if return_attn:
                attns.append(attn_w)
        cls_repr = tokens[:, 0]
        if return_attn:
            return cls_repr, attns, (H, W)
        return cls_repr


class MultiInputConvAtt(nn.Module):
    """The paper's proposed architecture (Figure 2). num_inputs=1 gives the single-input
    Conv-Att ablation (CDT only) when `tasks=["clock"]` is used in the data config."""

    def __init__(
        self,
        num_inputs: int = 3,
        backbone: str = "vgg16",
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 1,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        train_backbone: bool = True,
        use_pos_embed: bool = False,
    ):
        super().__init__()
        self.towers = nn.ModuleList(
            [
                SelfAttentionTower(
                    backbone=backbone,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    pretrained=pretrained,
                    train_backbone=train_backbone,
                    use_pos_embed=use_pos_embed,
                )
                for _ in range(num_inputs)
            ]
        )
        self.classifier = nn.Linear(hidden_dim * num_inputs, 2)

    def forward(self, inputs: list[torch.Tensor], return_attn: bool = False):
        reprs = []
        attn_maps = [] if return_attn else None
        for tower, x in zip(self.towers, inputs):
            if return_attn:
                r, attns, hw = tower(x, return_attn=True)
                attn_maps.append({"attns": attns, "grid_hw": hw})
            else:
                r = tower(x)
            reprs.append(r)
        logits = self.classifier(torch.cat(reprs, dim=1))
        if return_attn:
            return logits, attn_maps
        return logits


class MultiInputVGGBaseline(nn.Module):
    """Baseline: VGG16 + global-average-pool per tower, concat, FC+softmax (Table 1's
    VGG16 rows, and the model visualized with Grad-CAM in Figure 1)."""

    def __init__(self, num_inputs: int = 3, backbone: str = "vgg16", pretrained: bool = True, train_backbone: bool = True):
        super().__init__()
        self.backbones = nn.ModuleList(
            [build_backbone(backbone, pretrained=pretrained, train_backbone=train_backbone) for _ in range(num_inputs)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.backbones[0].out_channels * num_inputs, 2)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        feats = [self.pool(backbone(x)).flatten(1) for backbone, x in zip(self.backbones, inputs)]
        return self.classifier(torch.cat(feats, dim=1))


def build_model(cfg: dict) -> nn.Module:
    """Factory reading the `model` section of a config dict."""
    mcfg = cfg["model"]
    num_inputs = len(cfg["data"]["tasks"])
    if mcfg["type"] == "multi_input_conv_att":
        return MultiInputConvAtt(
            num_inputs=num_inputs,
            backbone=mcfg.get("backbone", "vgg16"),
            hidden_dim=mcfg.get("hidden_dim", 128),
            num_layers=mcfg.get("num_layers", 3),
            num_heads=mcfg.get("num_heads", 1),
            ff_dim=mcfg.get("ff_dim", 512),
            dropout=mcfg.get("dropout", 0.1),
            pretrained=mcfg.get("pretrained_backbone", True),
            train_backbone=mcfg.get("train_backbone", True),
            use_pos_embed=mcfg.get("use_pos_embed", False),
        )
    if mcfg["type"] == "vgg_baseline":
        return MultiInputVGGBaseline(
            num_inputs=num_inputs,
            backbone=mcfg.get("backbone", "vgg16"),
            pretrained=mcfg.get("pretrained_backbone", True),
            train_backbone=mcfg.get("train_backbone", True),
        )
    raise ValueError(f"Unknown model type: {mcfg['type']}")
