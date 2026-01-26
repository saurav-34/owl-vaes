import torch
import torch.nn.functional as F
import einops as eo

from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

def charbonnier_loss_fn(x, y, eps = 1.0e-3):
    diff = (x - y).float()
    # Mean over channels
    charb_per_pixel = (diff.pow(2) + eps**2).sqrt().mean(2 if x.ndim == 5 else 1)
    charb_per_pixel = charb_per_pixel.view(charb_per_pixel.shape[0], -1) # Flatten
    charb_loss = charb_per_pixel.mean()
    
    return charb_loss

def sample_tiles(x, y, n_tiles=2, tile_size=None):
    if tile_size is None:
        h = x.shape[-2]
        if h < 256:
            tile_size = 128
        else:
            tile_size = 256                                                                                                                                                                                                  
    if x.ndim == 5:                                                                                                                                                                                                                                
        x = eo.rearrange(x, 'b n c h w -> (b n) c h w')                                                                                                                                                                                            
        y = eo.rearrange(y, 'b n c h w -> (b n) c h w')                                                                                                                                                                                            
                                                                                                                                                                                                                                                    
    B, C, H, W = x.shape                                                                                                                                                                                                                           
    device = x.device                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                    
    # Pad if needed                                                                                                                                                                                                                                
    pad_h = max(tile_size - H, 0)                                                                                                                                                                                                                  
    pad_w = max(tile_size - W, 0)                                                                                                                                                                                                                  
    if pad_h > 0 or pad_w > 0:                                                                                                                                                                                                                     
        x = F.pad(x, (0, pad_w, 0, pad_h), value=-1.0)                                                                                                                                                                                             
        y = F.pad(y, (0, pad_w, 0, pad_h), value=-1.0)                                                                                                                                                                                             
        _, _, H, W = x.shape                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                    
    # Random starting positions: (B, n_tiles)                                                                                                                                                                                                      
    start_y = torch.randint(0, max(H - tile_size + 1, 1), (B, n_tiles), device=device)                                                                                                                                                             
    start_x = torch.randint(0, max(W - tile_size + 1, 1), (B, n_tiles), device=device)                                                                                                                                                             
                                                                                                                                                                                                                                                    
    # Create offset grids for tile extraction                                                                                                                                                                                                      
    offset_y = torch.arange(tile_size, device=device)  # [0, 1, ..., tile_size-1]                                                                                                                                                                  
    offset_x = torch.arange(tile_size, device=device)                                                                                                                                                                                              
                                                                                                                                                                                                                                                    
    # Compute all indices: (B, n_tiles, tile_size, tile_size)                                                                                                                                                                                      
    y_idx = start_y[:, :, None, None] + offset_y[None, None, :, None]                                                                                                                                                                              
    x_idx = start_x[:, :, None, None] + offset_x[None, None, None, :]                                                                                                                                                                              
                                                                                                                                                                                                                                                    
    # Expand for gathering                                                                                                                                                                                                                         
    batch_idx = torch.arange(B, device=device)[:, None, None, None]                                                                                                                                                                                
                                                                                                                                                                                                                                                    
    # Gather tiles: (B, n_tiles, C, tile_size, tile_size)                                                                                                                                                                                          
    x_nhwc = x.permute(0, 2, 3, 1)  # (B, H, W, C)                                                                                                                                                                                                 
    y_nhwc = y.permute(0, 2, 3, 1)                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                    
    tiles_x = x_nhwc[batch_idx, y_idx, x_idx]  # (B, n_tiles, tile_size, tile_size, C)                                                                                                                                                             
    tiles_y = y_nhwc[batch_idx, y_idx, x_idx]                                                                                                                                                                                                      
                                                                                                                                                                                                                                                    
    # Reshape to (B*n_tiles, C, tile_size, tile_size)                                                                                                                                                                                              
    tiles_x = tiles_x.permute(0, 1, 4, 2, 3).reshape(B * n_tiles, C, tile_size, tile_size)                                                                                                                                                         
    tiles_y = tiles_y.permute(0, 1, 4, 2, 3).reshape(B * n_tiles, C, tile_size, tile_size)                                                                                                                                                         
                                                                                                                                                                                                                                                    
    return tiles_x, tiles_y

def ssim_loss_(x, y, data_range=2.0, max_k=11):
    if x.ndim == 5: # video
        x = eo.rearrange(x, 'b n c h w -> (b n) c h w')
        y = eo.rearrange(y, 'b n c h w -> (b n) c h w')

    x = x.float()
    y = y.float()

    h,w = x.shape[-2:]
    k = int(min(h,w,max_k))
    if k % 2 == 0:
        k = max(1, k - 1)

    ssim_score = ssim_fn(preds=x, target=y, data_range=data_range, kernel_size=k)
    return 1 - ssim_score

def ssim_loss_fn(x, y, **tiling_kwargs):
    tiles_x, tiles_y = sample_tiles(x, y, **tiling_kwargs)
    ssim_loss = ssim_loss_(tiles_x, tiles_y)
    return ssim_loss

