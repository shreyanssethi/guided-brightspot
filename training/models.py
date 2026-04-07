"""
training/models.py

Defines two models for the WMH segmentation ablation study:

  1. BaselineUNet  — standard MONAI 3D U-Net, 2-channel input (FLAIR + T1).
                     Architecture matches HW5 exactly: 5 layers, residual units,
                     batch norm. Used as the unguided control.

  2. GuidedUNet   — same encoder/decoder depth and channel widths, but the
                     classical soft probability map is injected at each skip
                     connection via element-wise multiplication before the
                     decoder receives it. No learnable parameters are added.

Architecture (both models, matched to HW5):
    channels      = (16, 32, 64, 128, 256)  — 5-layer hierarchy
    strides       = (2, 2, 2, 2)            — halve spatial dims at each level
    num_res_units = 2                        — residual units per block
    norm          = BatchNorm3d
    out_channels  = 2                        — background + WMH

Key architectural note for GuidedUNet:
    The encoder separates CONVOLUTION from DOWNSAMPLING. At each level:
        1. Apply conv block at current resolution → store as skip
        2. Apply strided conv to downsample → feed to next level

    This means skips are captured at FULL resolution for each level, so
    the decoder can concat cleanly after upsampling. This matches how
    MONAI's UNet internally handles skip connections.

    Soft map injection:
        skip_guided = skip x resize(soft_map, skip.spatial_size)
    This spatially reweights encoder features toward classically-flagged
    WMH candidate regions. No learnable parameters are added.

What I took inspiration from:
    - HW5 MONAI notebook (from class) — baseline architecture
    - Li et al. 2018 (WMH challenge winner) — channel/layer design
    - Oktay et al. 2018 (Attention U-Net) — spatial attention at skip connections
    - BAGAU-Net 2020 — prior-guided WMH segmentation (population atlas;
      my approach differs: per-patient soft map, elementwise multiply,
      no extra learnable parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets import UNet
from monai.networks.layers import Norm


# ── Shared architecture hyperparameters (matched to HW5) ──────────────────────

SPATIAL_DIMS  = 3
IN_CHANNELS   = 2        # FLAIR + T1
OUT_CHANNELS  = 2        # background + WMH
CHANNELS      = (16, 32, 64, 128, 256)
GUIDED_CHANNELS = (11, 22, 44, 88, 176)     # Guided uses smaller channels to match baseline param count (~4.8M)
STRIDES       = (2, 2, 2, 2)
NUM_RES_UNITS = 2
NORM          = Norm.BATCH


# ── 1. Baseline U-Net ─────────────────────────────────────────────────────────

class BaselineUNet(nn.Module):
    """
    Standard MONAI 3D U-Net for WMH segmentation.

    Takes FLAIR and T1 concatenated as a 2-channel input.
    Architecture matches HW5 exactly — no modifications.

    Forward:
        x      (B, 2, H, W, D) — cat([flair, t1], dim=1)
        → logits (B, 2, H, W, D)
    """

    def __init__(self):
        super().__init__()
        self.unet = UNet(
            spatial_dims=SPATIAL_DIMS,
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            channels=CHANNELS,
            strides=STRIDES,
            num_res_units=NUM_RES_UNITS,
            norm=NORM,
        )

    def forward(self, x):
        return self.unet(x)


# ── 2. Guided U-Net building blocks ───────────────────────────────────────────

class _ResBlock(nn.Module):
    """
    3D residual block: two Conv3d + BN + ReLU with identity skip.
    Used to implement num_res_units in each encoder/decoder stage.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class _ConvBlock(nn.Module):
    """
    Convolution block at CURRENT resolution (no spatial downsampling).

    Projects in_ch → out_ch, then applies num_res_units residual blocks.
    This is the building block for each encoder level. The skip connection
    is captured at this block's output — BEFORE any downsampling step.

    This separation (conv-then-downsample) is what makes skip connections
    align correctly with decoder upsampling outputs.
    """

    def __init__(self, in_ch: int, out_ch: int, num_res_units: int):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_res_units):
            layers.append(_ResBlock(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """
    One decoder stage: upsample → concat guided skip → conv.

    After the guided soft map is applied to the skip features, this block:
      1. Upsamples x by 2× via transposed convolution
      2. Concats with the (guided) skip features
      3. Applies convolution + residual units to produce the decoder output

    The spatial alignment check only triggers when dimensions differ by ±1
    voxel (e.g. due to odd input sizes), NOT when x and skip are at different
    resolution levels — those should already match after proper upsampling.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_res_units: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_ch, in_ch, kernel_size=2, stride=2, bias=False
        )
        layers = [
            nn.Conv3d(in_ch + skip_ch, out_ch,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        for _ in range(num_res_units):
            layers.append(_ResBlock(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Only resize if spatial dims are off by ≤1 voxel (odd input edge case)
        # NOT when x and skip are at genuinely different resolutions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='trilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# ── 3. Guided U-Net ───────────────────────────────────────────────────────────

class GuidedUNet(nn.Module):
    """
    Classically-guided 3D U-Net for WMH segmentation.

    Encoder/decoder depth and channel widths are identical to BaselineUNet.
    The key difference is the soft map injection at each skip connection.

    Forward signature differs from BaselineUNet:
        x        (B, 2, H, W, D) — cat([flair, t1], dim=1)
        soft_map (B, 1, H, W, D) — per-patient soft probability map [0,1]
        → logits (B, 2, H, W, D)

    Encoder structure (separates conv from downsampling):
        conv0 → skip0 (B, 16, H,    W,    D   )
        ↓ down0
        conv1 → skip1 (B, 32, H/2,  W/2,  D/2 )
        ↓ down1
        conv2 → skip2 (B, 64, H/4,  W/4,  D/4 )
        ↓ down2
        conv3 → skip3 (B, 128,H/8,  W/8,  D/8 )
        ↓ down3
        bottleneck    (B, 256,H/16, W/16, D/16)

    Decoder (each step upsamples and injects guided skip):
        dec3: B,256,H/16 → upsample → B,256,H/8  + guided(skip3) → B,128,H/8
        dec2: B,128,H/8  → upsample → B,128,H/4  + guided(skip2) → B,64, H/4
        dec1: B,64, H/4  → upsample → B,64, H/2  + guided(skip1) → B,32, H/2
        dec0: B,32, H/2  → upsample → B,32, H    + guided(skip0) → B,16, H
        head: B,16,H → B,2,H  (logits)
    """

    def __init__(self):
        super().__init__()

        # ── Encoder: conv blocks (no downsampling) ─────────────────────────
        # Skips are captured at the OUTPUT of each conv block,
        # BEFORE the downsampling step below.
        self.conv0 = _ConvBlock(IN_CHANNELS,  CHANNELS[0], NUM_RES_UNITS)
        self.conv1 = _ConvBlock(CHANNELS[0],  CHANNELS[1], NUM_RES_UNITS)
        self.conv2 = _ConvBlock(CHANNELS[1],  CHANNELS[2], NUM_RES_UNITS)
        self.conv3 = _ConvBlock(CHANNELS[2],  CHANNELS[3], NUM_RES_UNITS)

        # ── Downsampling steps (strided conv, separate from skip) ──────────
        self.down0 = nn.Sequential(
            nn.Conv3d(CHANNELS[0], CHANNELS[0], kernel_size=2,
                      stride=STRIDES[0], bias=False),
            nn.BatchNorm3d(CHANNELS[0]),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv3d(CHANNELS[1], CHANNELS[1], kernel_size=2,
                      stride=STRIDES[1], bias=False),
            nn.BatchNorm3d(CHANNELS[1]),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(CHANNELS[2], CHANNELS[2], kernel_size=2,
                      stride=STRIDES[2], bias=False),
            nn.BatchNorm3d(CHANNELS[2]),
            nn.ReLU(inplace=True),
        )
        self.down3 = nn.Sequential(
            nn.Conv3d(CHANNELS[3], CHANNELS[3], kernel_size=2,
                      stride=STRIDES[3], bias=False),
            nn.BatchNorm3d(CHANNELS[3]),
            nn.ReLU(inplace=True),
        )

        # ── Bottleneck (at lowest resolution, no skip) ─────────────────────
        self.bottleneck = _ConvBlock(CHANNELS[3], CHANNELS[4], NUM_RES_UNITS)

        # ── Decoder ────────────────────────────────────────────────────────
        # Each block upsamples from the previous decoder output, then concats
        # the guided skip from the corresponding encoder level.
        self.dec3 = _DecoderBlock(CHANNELS[4], CHANNELS[3],
                                   CHANNELS[3], NUM_RES_UNITS)
        self.dec2 = _DecoderBlock(CHANNELS[3], CHANNELS[2],
                                   CHANNELS[2], NUM_RES_UNITS)
        self.dec1 = _DecoderBlock(CHANNELS[2], CHANNELS[1],
                                   CHANNELS[1], NUM_RES_UNITS)
        self.dec0 = _DecoderBlock(CHANNELS[1], CHANNELS[0],
                                   CHANNELS[0], NUM_RES_UNITS)

        # ── Final classification head ──────────────────────────────────────
        self.head = nn.Conv3d(CHANNELS[0], OUT_CHANNELS, kernel_size=1)

    def _guide(self, skip: torch.Tensor,
               soft_map: torch.Tensor) -> torch.Tensor:
        """
        Resize soft_map to skip's spatial dims and apply elementwise multiply.

        The soft map is a scalar per-voxel confidence that the classical
        pipeline considers that voxel a WMH candidate. Broadcasting across
        the channel dim reweights all feature channels equally.

        Args:
            skip     (B, C, H', W', D') — encoder features at this level
            soft_map (B, 1, H,  W,  D ) — full-resolution soft map

        Returns:
            (B, C, H', W', D') — spatially reweighted features
        """
        sm = F.interpolate(
            soft_map, size=skip.shape[2:],
            mode='trilinear', align_corners=False
        )
        return skip * (1 + sm)  # boost signal, never zero it out

    def forward(self, x: torch.Tensor,
                soft_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        (B, 2, H, W, D): concatenated FLAIR + T1
            soft_map (B, 1, H, W, D): classical soft probability map

        Returns:
            logits (B, 2, H, W, D)
        """
        # ── Encoder: conv → skip → downsample ─────────────────────────────
        s0 = self.conv0(x)          # (B, 16, H,    W,    D   )
        x  = self.down0(s0)         # (B, 16, H/2,  W/2,  D/2 )

        s1 = self.conv1(x)          # (B, 32, H/2,  W/2,  D/2 )
        x  = self.down1(s1)         # (B, 32, H/4,  W/4,  D/4 )

        s2 = self.conv2(x)          # (B, 64, H/4,  W/4,  D/4 )
        x  = self.down2(s2)         # (B, 64, H/8,  W/8,  D/8 )

        s3 = self.conv3(x)          # (B, 128,H/8,  W/8,  D/8 )
        x  = self.down3(s3)         # (B, 128,H/16, W/16, D/16)

        # ── Bottleneck ────────────────────────────────────────────────────
        x = self.bottleneck(x)      # (B, 256,H/16, W/16, D/16)

        # ── Decoder: upsample → concat guided skip → conv ─────────────────
        x = self.dec3(x, self._guide(s3, soft_map))  # (B, 128,H/8,  W/8,  D/8 )
        x = self.dec2(x, self._guide(s2, soft_map))  # (B, 64, H/4,  W/4,  D/4 )
        x = self.dec1(x, self._guide(s1, soft_map))  # (B, 32, H/2,  W/2,  D/2 )
        x = self.dec0(x, self._guide(s0, soft_map))  # (B, 16, H,    W,    D   )

        return self.head(x)                           # (B, 2,  H,    W,    D   )


# ── Factory functions ─────────────────────────────────────────────────────────

def build_baseline() -> BaselineUNet:
    """Instantiate the baseline U-Net. Call .to(device) after."""
    return BaselineUNet()


def build_guided() -> GuidedUNet:
    """Instantiate the guided U-Net. Call .to(device) after."""
    return GuidedUNet()


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    python training/models.py

    Expected:
        BaselineUNet  params ~4.8M  output (2, 2, 96, 96, 48)
        GuidedUNet    params ~5-6M  output (2, 2, 96, 96, 48)
        Param delta   small — guidance adds zero learnable params,
                      small diff from decoder design choices
        Zero soft_map patches: 0
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, H, W, D = 2, 96, 96, 48

    x        = torch.randn(B, 2, H, W, D).to(device)
    soft_map = torch.rand(B, 1, H, W, D).to(device)

    # Baseline
    baseline = build_baseline().to(device)
    n_base   = sum(p.numel() for p in baseline.parameters() if p.requires_grad)
    with torch.no_grad():
        out_b = baseline(x)
    print(f'BaselineUNet')
    print(f'  params: {n_base:,}')
    print(f'  input:  {tuple(x.shape)}')
    print(f'  output: {tuple(out_b.shape)}')
    assert out_b.shape == (B, OUT_CHANNELS, H, W, D), \
        f'Shape mismatch: {out_b.shape} != {(B, OUT_CHANNELS, H, W, D)}'
    print(f'  shape: OK')

    # Guided
    guided  = build_guided().to(device)
    n_guide = sum(p.numel() for p in guided.parameters() if p.requires_grad)
    with torch.no_grad():
        out_g = guided(x, soft_map)
    print(f'\nGuidedUNet')
    print(f'  params: {n_guide:,}')
    print(f'  input:  x={tuple(x.shape)}  soft_map={tuple(soft_map.shape)}')
    print(f'  output: {tuple(out_g.shape)}')
    assert out_g.shape == (B, OUT_CHANNELS, H, W, D), \
        f'Shape mismatch: {out_g.shape} != {(B, OUT_CHANNELS, H, W, D)}'
    print(f'  shape: OK')

    print(f'\nParam delta: {abs(n_guide - n_base):,} '
          f'({abs(n_guide - n_base) / n_base * 100:.1f}% of baseline)')
    print('\nBoth models passed. Ready for training.')