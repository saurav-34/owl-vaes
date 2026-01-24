"""
Tiled encode/decode for 3D videos.
Only does temporal tiling to be clear
"""

import torch
import einops as eo
from tqdm import tqdm

@torch.no_grad()
def tiled_rec(encoder, decoder, x, tile_size = 4):
    # x is [b,t,c,h,w]
    x_tiles = x.split(tile_size, dim = 1)

    rec_tiles = [decoder(encoder(tile)) for tile in tqdm(x_tiles)]
    rec = torch.cat(rec_tiles, dim = 1)
    return rec