"""
Feature caching for convolutional layers
"""

import torch
from torch import nn
import uuid

from .resnet_3d import CausConv3d


class ConvCache:
    """
    Feature cache for CausConv3d layers.
    :param frames: Should be kernel_size - 1
    """
    def __init__(self, frames = 2):
        self.cache = {}
        self.frames = frames

    def associate(self, module):
        """
        Give unique 16-character hex IDs to all causal conv layers
        """
        for name, submodule in module.named_modules():
            if isinstance(submodule, CausConv3d):
                cache_id = uuid.uuid4()
                submodule.cache_id = cache_id

    def update(self, key, x):
        """
        Updates cache and returns x with cache values appended.
        If cache had something, don't pad the output.
        If cache didn't have something, pad the output
        If nothing was appended (i.e. cache empty) no padding will be needed.
        If something was appended, padding is not needed.
        Boolean value indicates if appending happened (i.e. it's False => padding is needed)
        """
        # assume x is [b,c,t,h,w]
        assert x.shape[2] >= self.frames, "Input when using cache did not contain enough frames to update cache. Note that number of frames in a tiled encode job must be a multiple of downsampling factor."
        do_pad = True
        if key in self.cache:
            do_pad = False
            x = torch.cat([self.cache[key].clone(), x], dim = 2)
        
        self.cache[key] = x[:,:,-self.frames:].clone() # Last N frames
        return x, do_pad

