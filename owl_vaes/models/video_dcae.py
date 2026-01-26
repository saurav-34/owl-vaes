import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from ..nn.resnet import (
    LandscapeToSquare, SquareToLandscape,
    WeightNormConv2d
)
from ..nn.resnet_3d import (
    DownBlock3D, UpBlock3D, CausConv3d,
    TemporalDownsample, TemporalUpsample,
    LandscapeToSquare3D, SquareToLandscape3D
)
from ..nn.sana_3d import (
    ChannelToSpace3D, SpaceToChannel3D,
    ChannelAverage3D, ChannelDuplication3D,
    flatten_pixel_shuffle, flatten_pixel_unshuffle
)

from copy import deepcopy

def is_landscape(config):
    sample_size = config.sample_size
    if isinstance(sample_size, int):
        return False
    sample_size = (int(sample_size[0]), int(sample_size[1]))
    if sample_size[0] < sample_size[1]: # width > height
        return True
    return False

def swap_tc(x):
    # b,t,c,h,w -> b,c,t,h,w
    b,t,c,h,w = x.shape
    x = x.permute(0, 2, 1, 3, 4)
    return x

def swap_ct(x):
    # b,c,t,h,w -> b,t,c,h,w
    b,c,t,h,w = x.shape
    x = x.permute(0, 2, 1, 3, 4)
    return x

def batch(x):
    b,t,c,h,w = x.shape
    x = x.reshape(b*t, c, h, w)
    return x

def unbatch(x, b):
    _, c, h, w = x.shape
    x = x.reshape(b, -1, c, h, w)
    return x

def latent_ln(z, eps=1e-6):
    # z: [b, c, t, h, w]
    mean = z.mean(dim=(1, 2, 3, 4), keepdim=True)
    var  = z.var(dim=(1, 2, 3, 4), keepdim=True, unbiased=False)
    return (z - mean) / torch.sqrt(var + eps)

class PermissiveIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class Encoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.config = config
        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.latent_channels = config.latent_channels
        self.is_landscape = is_landscape(config)
        self.encoder_chunk_size = getattr(config, 'encoder_chunk_size', 4)

        # Landscape to square transformation (keeps channels same, just reshapes spatially)
        self.l2s = LandscapeToSquare3D(config.channels, config.channels) if self.is_landscape else PermissiveIdentity()

        self.conv_in = CausConv3d(config.channels * 4, ch_0, 3, 1, 1)

        blocks = []
        residuals = []
        temp_downs = []
        ch = ch_0

        blocks_per_stage = config.encoder_blocks_per_stage
        total_blocks = sum(blocks_per_stage)

        for i, block_count in enumerate(blocks_per_stage):
            next_ch = min(ch*2, ch_max)

            is_temporal = (i < 2)

            blocks.append(DownBlock3D(ch, next_ch, block_count, total_blocks))
            residuals.append(SpaceToChannel3D(ch, next_ch))
            temp_downs.append(TemporalDownsample(next_ch, next_ch) if is_temporal else PermissiveIdentity())

            ch = next_ch

        self.blocks = nn.ModuleList(blocks)
        self.residuals = nn.ModuleList(residuals)
        self.temp_downs = nn.ModuleList(temp_downs)

        self.conv_out = ChannelAverage3D(ch, config.latent_channels)

    def forward(self, x, feat_cache = None):
        """
        btchw in -> btchw out
        """
        b = x.shape[0]
        x = swap_tc(x) # -> bcthw
        x = self.l2s(x, feat_cache) # landscape -> square if needed
        x = flatten_pixel_unshuffle(x)
        x = self.conv_in(x, feat_cache)

        for block, shortcut, temp_down in zip(self.blocks, self.residuals, self.temp_downs):
            x = block(x, feat_cache) + shortcut(x)
            x = temp_down(x, feat_cache)

        mu = self.conv_out(x, feat_cache)
        mu = latent_ln(mu)
        mu = swap_ct(mu)
        return mu

class Decoder(nn.Module):
    def __init__(self, config : 'ResNetConfig'):
        super().__init__()

        self.config = config
        self.is_landscape = is_landscape(config)

        size = config.sample_size
        latent_size = config.latent_size
        ch_0 = config.ch_0
        ch_max = config.ch_max

        self.conv_in = ChannelDuplication3D(config.latent_channels, ch_max)

        blocks = []
        residuals = []
        temp_ups = []

        ch = ch_0

        blocks_per_stage = config.decoder_blocks_per_stage
        total_blocks = sum(blocks_per_stage)

        for i, block_count in enumerate(blocks_per_stage):
            next_ch = min(ch*2, ch_max)

            is_temporal = (i < 2)

            temp_ups.append(TemporalUpsample(next_ch, next_ch) if is_temporal else PermissiveIdentity())
            blocks.append(UpBlock3D(next_ch, ch, block_count, total_blocks))
            residuals.append(ChannelToSpace3D(next_ch, ch))

            ch = next_ch

        self.blocks = nn.ModuleList(list(reversed(blocks)))
        self.residuals = nn.ModuleList(list(reversed(residuals)))
        self.temp_ups = nn.ModuleList(list(reversed(temp_ups)))

        self.act_out = nn.SiLU()
        self.conv_out = CausConv3d(ch_0, config.channels*4, 3, 1, 1)

        # Square to landscape transformation (keeps channels same, just reshapes spatially)
        self.s2l = SquareToLandscape3D(config.channels, config.channels) if self.is_landscape else PermissiveIdentity()
        

    def forward(self, x, feat_cache = None):
        b = x.shape[0]
        x = swap_tc(x)
        x = self.conv_in(x, feat_cache)

        for block, shortcut, temp_up in zip(self.blocks, self.residuals, self.temp_ups):
            x = temp_up(x, feat_cache)
            x = block(x, feat_cache) + shortcut(x)

        x = self.act_out(x)
        x = self.conv_out(x, feat_cache)
        x = flatten_pixel_shuffle(x)
        x = self.s2l(x, feat_cache) # square -> landscape if needed
        x = swap_ct(x)
        return x

class VideoDCAE(nn.Module):
    """
    Video DCAE based autoencoder
    """
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.config = config
        
    def forward(self, x, feat_cache = None):
        mu = self.encoder(x, feat_cache)
        rec = self.decoder(mu, feat_cache)
        return rec, mu

def test_video_dcae():
    from dataclasses import dataclass
    @dataclass
    class VideoDCAEConfig:
        sample_size = (360, 640)
        channels = 3
        latent_size = 16
        latent_channels = 16
        ch_0 = 64
        ch_max = 512
        encoder_blocks_per_stage = [2, 2, 4]
        decoder_blocks_per_stage = [2, 2, 4]
    
    config = VideoDCAEConfig()
    model = VideoDCAE(config)
    model = model.cuda().bfloat16()

    with torch.no_grad():
        x = torch.randn(1, 16, config.channels, config.sample_size[0], config.sample_size[1]).cuda().bfloat16()
        z = model.encoder(x)
        print(z.shape)
        rec = model.decoder(z)
        print(rec.shape)

if __name__ == "__main__":
    test_video_dcae()
    

