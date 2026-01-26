from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import torch
import math
import einops as eo

from torch.nn.utils.parametrizations import weight_norm

from .resnet import Upsample, Downsample

def checkpoint_gpu(fn, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(fn, *args, **kwargs)

def WeightNormConv3d(*args, **kwargs):
    return weight_norm(nn.Conv3d(*args, **kwargs))

def causal_pad(x, p = 1, k = 3, do_causal_pad=True):
    # x is [b,c,t,h,w]
    # Spatial padding: replicate (constant works too)
    # Temporal padding: replicate to avoid zero boundary at first frame
    b,c,t,h,w = x.shape

    # Pad spatially first (h, w dimensions)
    x = F.pad(x, (p, p, p, p, 0, 0), mode="replicate")

    # Pad temporally (causal - only left/past side)
    # Replicate first frame backward instead of zeros
    temporal_pad = math.ceil(k/2) if do_causal_pad else 0
    x = F.pad(x, (0, 0, 0, 0, temporal_pad, 0), mode="replicate")

    return x

class CausConv3d(nn.Module):
    def __init__(self, fi, fo, k, s, p):
        super().__init__()

        self.conv = WeightNormConv3d(fi, fo, k, s, 0)
        self.k = k
        self.p = p

        self.cache_id = None
    
    def forward(self, x, feat_cache : 'ConvCache' = None):
        do_pad = True
        if feat_cache is not None:
            x, do_pad = feat_cache.update(self.cache_id, x)

        x = causal_pad(x, self.p, self.k, do_causal_pad=do_pad)
        x = self.conv(x)
        return x

class ResBlock3D(nn.Module):
    def __init__(self, ch, total_res_blocks):
        super().__init__()

        self.conv1 = CausConv3d(ch, 2*ch, 3, 1, 1)
        self.conv2 = CausConv3d(2*ch, 2*ch, 3, 1, 1)
        self.conv3 = CausConv3d(2*ch, ch, 3, 1, 1)

        self.act1 = nn.SiLU(inplace=False)
        self.act2 = nn.SiLU(inplace=False)

          # Fix up init
        scaling_factor = total_res_blocks ** -.25

        nn.init.kaiming_uniform_(self.conv1.conv.weight)
        nn.init.zeros_(self.conv1.conv.bias)
        self.conv1.conv.weight.data *= scaling_factor

        nn.init.kaiming_uniform_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)
        self.conv2.conv.weight.data *= scaling_factor

        nn.init.zeros_(self.conv3.conv.weight)
  
    def forward(self, x, feat_cache = None):
        def _forward(x):
            x = self.conv1(x, feat_cache)
            x = self.act1(x)
            x = self.conv2(x, feat_cache)
            x = self.act2(x)
            x = self.conv3(x, feat_cache)
            return x

        res = x
        x = checkpoint_gpu(_forward, x)
        x = x + res
        return x

class TemporalUpsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, ch_out*2, 3, 1, 1)

    def forward(self, x, feat_cache = None):
        x = self.proj(x, feat_cache) # [b, c*2, t, h, w]
        x = eo.rearrange(x, 'b (c two) t h w -> b c (t two) h w', two = 2)
        return x

class SpatialUpsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv3d(ch_in, ch_out, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.proj(x)
        x = video_interpolate(x, scale_factor=2, mode='bilinear')
        return x

class TemporalDownsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, ch_out//2, 3, 1, 1)

    def forward(self, x, feat_cache = None):
        x = self.proj(x, feat_cache) # [b, c//2, t, h, w]
        x = eo.rearrange(x, 'b c (t two) h w -> b (c two) t h w', two = 2)
        return x

class SpatialDownsample(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = WeightNormConv3d(ch_in, ch_out, 1, 1, 0, bias=False)

    def forward(self, x):
        x = video_interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.proj(x)
        return x

class UpBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        self.up = SpatialUpsample(ch_in, ch_out)
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock3D(ch_out, total_blocks))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, feat_cache = None):
        x = self.up(x)
        for block in self.blocks:
            x = block(x, feat_cache)
        return x

class DownBlock3D(nn.Module):
    def __init__(self, ch_in, ch_out, num_res, total_blocks):
        super().__init__()

        self.down = SpatialDownsample(ch_in, ch_out)
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock3D(ch_in, total_blocks))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, feat_cache = None):
        for block in self.blocks:
            x = block(x, feat_cache)
        x = self.down(x)
        return x

def video_interpolate(x, size=None, scale_factor=None, mode='bicubic'):
    """
    Interpolate video spatially only (per-frame), preserving temporal dimension.
    x: [b, c, t, h, w]
    size: (h_new, w_new) for target spatial size
    scale_factor: float or (h_scale, w_scale) for spatial scaling
    """
    b, c, t, h, w = x.shape
    x = eo.rearrange(x, 'b c t h w -> (b t) c h w')
    x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    x = eo.rearrange(x, '(b t) c h w -> b c t h w', b=b, t=t)
    return x

def find_nearest_square(h, w):
    # Assuming h,w are 9:16, find nearest square 1:1 that is a power of 2
    # Keep area roughly the same
    area = h * w
    side = round((area)**0.5)
    # Find nearest power of 2
    power = round(math.log2(side))
    side = 2**power
    return side, side

def find_nearest_landscape(h, w):
    # Assuming h,w are 1:1, find nearest landscape 9:16
    # Only use common 9:16 resolutions
    area = h * w

    # Define common 9:16 resolutions
    resolutions = [
        (45, 80),
        (90, 160),
        (180, 320),
        (360, 640),
        (720, 1280),
        (1080, 1920),
        (2160, 3840)
    ]

    # Find closest resolution by area
    min_diff = float('inf')
    h_new, w_new = resolutions[0]

    for res_h, res_w in resolutions:
        res_area = res_h * res_w
        diff = abs(area - res_area)
        if diff < min_diff:
            min_diff = diff
            h_new, w_new = res_h, res_w

    return h_new, w_new

class LandscapeToSquare3D(nn.Module):
    def __init__(self, ch, ch_out = None):
        super().__init__()

        if ch_out is None: ch_out = ch
        self.proj = CausConv3d(ch, ch_out, 3, 1, 1)

    def forward(self, x, feat_cache = None):
        # x is [b, c, t, h, w] where h:w is 9:16
        b, c, t, h, w = x.shape
        target_h, target_w = find_nearest_square(h, w)
        x = self.proj(x, feat_cache)
        # Interpolate spatially only per-frame
        x = video_interpolate(x, size=(target_h, target_w), mode='bilinear')
        return x

class SquareToLandscape3D(nn.Module):
    def __init__(self, ch, ch_out = None):
        super().__init__()

        if ch_out is None: ch_out = ch
        self.proj = CausConv3d(ch, ch_out, 3, 1, 1)

    def forward(self, x, feat_cache = None):
        # x is [b, c, t, h, w] where h:w is 1:1
        b, c, t, h, w = x.shape
        target_h, target_w = find_nearest_landscape(h, w)
        # Interpolate FIRST (spatially only per-frame)
        x = video_interpolate(x, size=(target_h, target_w), mode='bilinear')
        # Then learned conv to fix artifacts
        x = self.proj(x, feat_cache)
        return x