import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import einops as eo

from .resnet_3d import CausConv3d, WeightNormConv3d

def flatten_pixel_shuffle(x): # upsampling
    # x is [b,c,t,h,w]
    b = x.shape[0]
    x = eo.rearrange(x, 'b c t h w -> (b t) c h w')
    x = F.pixel_shuffle(x, 2)
    x = eo.rearrange(x, '(b t) c h w -> b c t h w', b = b)
    return x

def flatten_pixel_unshuffle(x): # downsampling
    # x is [b,c,t,h,w]
    b = x.shape[0]
    x = eo.rearrange(x, 'b c t h w -> (b t) c h w')
    x = F.pixel_unshuffle(x, 2)
    x = eo.rearrange(x, '(b t) c h w -> b c t h w', b = b)
    return x

class SpaceToChannel3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, ch_out // 4, 3, 1, 1)

    def forward(self, x):
        x = self.proj(x)
        x = flatten_pixel_unshuffle(x)
        return x

class ChannelToSpace3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, 4 * ch_out, 3, 1, 1)
    
    def forward(self, x):
        x = self.proj(x)
        x = flatten_pixel_shuffle(x)
        return x

class ChannelAverage3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, ch_out, 3, 1, 1)
        self.grps = ch_in // ch_out
    
    def forward(self, x):
        res = x
        x = self.proj(x)
        res = res.view(res.shape[0], self.grps, res.shape[1] // self.grps, res.shape[2], res.shape[3], res.shape[4])
        res = res.mean(dim=1)
        return res + x

class ChannelDuplication3D(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.proj = CausConv3d(ch_in, ch_out, 3, 1, 1)
        self.reps = ch_out // ch_in
    
    def forward(self, x):
        res = x
        x = self.proj(x)
        res = res.repeat_interleave(self.reps, dim=1)
        return res + x