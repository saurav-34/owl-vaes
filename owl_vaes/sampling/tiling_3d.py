"""
Tiled encode/decode for 3D videos.
Only does temporal tiling to be clear
"""

import torch
import einops as eo
from tqdm import tqdm

from ..nn.conv_cache import ConvCache

@torch.no_grad()
def tiled_encode(encoder, x, tile_size = 8, cache_frames = 2):
    """
    Encode video with temporal tiling using feature caching.

    :param encoder: Encoder module
    :param x: Input video [b, t, c, h, w]
    :param tile_size: Number of frames per tile (must be divisible by temporal downsampling factor)
    :param cache_frames: Number of frames to cache (kernel_size - 1, typically 2 for k=3)
    :return: Encoded latents [b, t', c', h', w']
    """
    # x is [b,t,c,h,w]
    x_tiles = x.split(tile_size, dim = 1)

    # Create and associate cache with encoder
    enc_cache = ConvCache(frames=cache_frames)
    enc_cache.associate(encoder)

    z_tiles = []
    for tile in tqdm(x_tiles, desc="Encoding tiles"):
        z = encoder(tile, feat_cache=enc_cache)
        z_tiles.append(z)

    z = torch.cat(z_tiles, dim = 1)
    return z

@torch.no_grad()
def tiled_decode(decoder, z, tile_size = 8, cache_frames = 2):
    """
    Decode latents with temporal tiling using feature caching.

    :param decoder: Decoder module
    :param z: Input latents [b, t, c, h, w]
    :param tile_size: Number of latent frames per tile (must be divisible by temporal upsampling factor)
    :param cache_frames: Number of frames to cache (kernel_size - 1, typically 2 for k=3)
    :return: Decoded video [b, t', c', h', w']
    """
    # z is [b,t,c,h,w]
    z_tiles = z.split(tile_size, dim = 1)

    # Create and associate cache with decoder
    dec_cache = ConvCache(frames=cache_frames)
    dec_cache.associate(decoder)

    rec_tiles = []
    for tile in tqdm(z_tiles, desc="Decoding tiles"):
        rec = decoder(tile, feat_cache=dec_cache)
        rec_tiles.append(rec)

    rec = torch.cat(rec_tiles, dim = 1)
    return rec

@torch.no_grad()
def tiled_rec(encoder, decoder, x, tile_size = 8, cache_frames = 2):
    """
    Reconstruct video with temporal tiling using feature caching.

    :param encoder: Encoder module
    :param decoder: Decoder module
    :param x: Input video [b, t, c, h, w]
    :param tile_size: Number of frames per tile (must be divisible by temporal downsampling factor)
    :param cache_frames: Number of frames to cache (kernel_size - 1, typically 2 for k=3)
    :return: Reconstructed video [b, t, c, h, w]
    """
    # x is [b,t,c,h,w]
    x_tiles = x.split(tile_size, dim = 1)

    # Create and associate separate caches for encoder and decoder
    enc_cache = ConvCache(frames=cache_frames)
    enc_cache.associate(encoder)

    dec_cache = ConvCache(frames=cache_frames)
    dec_cache.associate(decoder)

    rec_tiles = []
    for tile in tqdm(x_tiles, desc="Reconstructing tiles"):
        z = encoder(tile, feat_cache=enc_cache)
        rec = decoder(z, feat_cache=dec_cache)
        rec_tiles.append(rec)

    rec = torch.cat(rec_tiles, dim = 1)
    return rec