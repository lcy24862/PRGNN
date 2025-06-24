# 2022.06.17-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# relative position embedding
# References: https://arxiv.org/abs/2009.13658
# --------------------------------------------------------
def get_3d_relative_pos_embed(embed_dim, grid_size):
    """
    embed_dim: channel (64개)
    grid_size: int of the grid height and width (n=196일 때 14가 들어옴옴)
    return:
    pos_embed: [grid_size*grid_size, grid_size*grid_size]
    """
    pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_size)
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]
    return relative_pos


# --------------------------------------------------------
# 3d sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    
    grid = [[[ 0,  1,  2, ..., 13],  # X-coordinates (first row: column indices)
         [ 0,  1,  2, ..., 13],  
         [ 0,  1,  2, ..., 13],  
         ...,
         [ 0,  1,  2, ..., 13]],  # Last row

        [[ 0,  0,  0, ...,  0],  # Y-coordinates (first column: row indices)
         [ 1,  1,  1, ...,  1],  
         [ 2,  2,  2, ...,  2],  
         ...,
         [13, 13, 13, ..., 13]]]  # Last column
         
    """
    grid_x = np.arange(grid_size, dtype=float) # [0, 1, ... 13]
    grid_y = np.arange(grid_size, dtype=float)
    grid_z = np.arange(grid_size, dtype=float)
    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing='ij')  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H*W*Z, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (H*W*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (H*W*Z, D/3)

    emb = np.concatenate([emb_h, emb_w, emb_z], axis=1) # (H*W*Z, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position (ex. 32)
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    omega = np.arange(embed_dim // 2, dtype=float) # divide into two arrays for sin and cosine [0, 1, 2, ..., 15]
    omega /= embed_dim / 2. # 16
    omega = 1. / 10000**omega  # (D/3,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/6)
    emb_cos = np.cos(out) # (M, D/6)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D/3)
    return emb

import torch
import math

def get_roi_sincos_pos_embed(embed_dim, roi_centers, temperature=10000):
    """
    Compute sinusoidal position embeddings for ROI centroids.
    
    Args:
        embed_dim (int): The output embedding dimension. Must be divisible by 3 and such that embed_dim/3 is even.
        roi_centers (Tensor): A tensor of shape [batch, num_rois, 3] containing the (x, y, z) coordinates
                              of the ROI centroids. Coordinates can be in any consistent scale (normalized or absolute).
        temperature (float): Temperature parameter controlling the frequency scale.
    
    Returns:
        pos_embed (Tensor): Positional embeddings with shape [batch, embed_dim, num_rois].
    """
    assert embed_dim % 3 == 0, "Embedding dimension must be divisible by 3."
    embed_each = embed_dim // 3
    assert embed_each % 2 == 0, "Per-axis embedding dimension must be even."

    # Number of frequency bands per axis
    dim_half = embed_each // 2

    # Create the frequency vector for each axis.
    freq_seq = torch.arange(dim_half, dtype=torch.float32, device=roi_centers.device)

    freq_seq = freq_seq / dim_half
    inv_freq = 1.0 / (temperature ** freq_seq)  # [dim_half]

    pos = roi_centers.unsqueeze(-1)  # [batch, num_rois, 3, 1]
    pos_enc = pos * inv_freq 

    sin_embed = torch.sin(pos_enc)  # [batch, num_rois, 3, dim_half]
    cos_embed = torch.cos(pos_enc)  # [batch, num_rois, 3, dim_half]

    axis_embed = torch.cat([sin_embed, cos_embed], dim=-1)  # [batch, num_rois, 3, embed_each]

    # Concatenate the embeddings for the three axes along the channel dimension.
    pos_embed = axis_embed.view(roi_centers.shape[0], roi_centers.shape[1], embed_dim)
    pos_embed = pos_embed.permute(0, 2, 1)
    return pos_embed
