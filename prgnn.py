# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer
import nibabel as nib
import numpy as np

from timm.layers import DropPath, to_2tuple, trunc_normal_
from utils import ResNetFeatures


def obtain_contribution_maps(opt, feature_map_shape, original_mask, num_rois):
    """
    Track and aggregate features from the original mask through downsampling for batched 3D data.

    Args:
        feature_map (torch.Tensor): Feature map of shape (B, C', H', W', D').
        original_mask (torch.Tensor): ROI mask of shape (num_rois, H, W, D).
        num_rois (int): Number of unique ROIs.

    Returns:
        list[dict]: A list of dictionaries for each batch where keys are ROI labels and 
                    values are aggregated features (mean pooling).
    """
    # Get dimensions
    B, C_prime, H_prime, W_prime, D_prime = feature_map_shape
    _, H, W, D = original_mask.shape

    # Initialize contribution maps for each ROI
    contribution_maps = torch.zeros((B, num_rois, H_prime, W_prime, D_prime), device=opt.device)

    # Generate binary masks for each ROI and downsample
    for roi in range(1, num_rois):  # Assuming ROI labels are 1-indexed
        binary_mask = original_mask[roi].float().unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W, D)
        # Downsample the binary mask to match the feature map's spatial dimensions
        # F.interpolate 는 input shape 가 (N, C, d1, d2, d3) 이여야함...
        downsampled_mask = F.interpolate(
            binary_mask, 
            size=(H_prime, W_prime, D_prime), 
            mode='trilinear', 
            align_corners=False
        )
        contribution_maps[:, roi] = downsampled_mask.squeeze(1)  # Remove channel dimension

    # Normalize contribution maps to sum to 1 for each pixel in the feature map
    contribution_maps = contribution_maps / (contribution_maps.sum(dim=1, keepdim=True) + 1e-8)  # Avoid division by zero

    return contribution_maps


class Prediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Prediction, self).__init__()
        # 1D convolution with kernel size 1 to reduce channels to num_classes.
        # self.conv1d = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        self.conv1d = nn.Conv1d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, channels, num_rois]
        # Transform channels to num_classes for each ROI.
        logits = self.conv1d(x).squeeze()  # shape becomes [batch, num_classes, 1]
        probs = F.softmax(logits, dim=1)
        return probs


class AveragePool1d(nn.Module):
    def __init__(self):
        super(AveragePool1d, self).__init__()

    def forward(self, x):
        pooled = x.mean(dim=1, keepdim=True).permute(0, 2, 1)  # Shape: [Batch, num_rois, 1]
        return pooled


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv1d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm1d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv1d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm1d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        drop_path = opt.drop_path
        relative_pos = opt.relative_pos
        self.pool = opt.pool

        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        in_channels = opt.in_channels # [48, 96, 144, 336, 624]
        channels = opt.channels # [48, 48, 96, 240, 384]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k
        max_dilation = 49 // max(num_knn)
        
        self.features = ResNetFeatures('basic', [1, 1, 1], opt.channels[1:])
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 53))
        
        ###########################################
        # One_hot_mask 만들어서 stem 에 곱해줄거임 #
        ROI_mask = nib.load(opt.one_hot_mask).get_fdata()
        ROI_mask = torch.Tensor(ROI_mask)
        labels = np.unique(ROI_mask)
        num_rois = len(labels)

        one_hot_mask = torch.zeros((num_rois, 91, 109, 91))
        for label_new, label_orig in enumerate(labels):
            # exclude zero??
            if label_new == 0:
                continue
            one_hot_mask[label_new] = (ROI_mask == label_orig).float()  # Binary mask for each ROI

        roi_centers = self.get_normalized_roi_centers(one_hot_mask, opt.batch_size)
        self.contribution_maps = {
            'stage0': obtain_contribution_maps(opt, (1, channels[0], 46, 55, 46), one_hot_mask, one_hot_mask.shape[0]), # TODO: 나중에 1이 아니라 batch size 로 바꿔야함
            'stage1': obtain_contribution_maps(opt, (1, channels[0], 23, 28, 23), one_hot_mask, one_hot_mask.shape[0]),
            'stage2': obtain_contribution_maps(opt, (1, channels[1], 12, 14, 12), one_hot_mask, one_hot_mask.shape[0]),
            'stage3': obtain_contribution_maps(opt, (1, channels[2], 6, 7, 6), one_hot_mask, one_hot_mask.shape[0]),
        }
        ###########################################

        self.backbone = nn.ModuleDict()
        for i in range(len(channels)):
            backbone = nn.ModuleList([])
            for j in range(blocks[i]):
                if j == 0:
                    backbone += [Seq(Grapher(in_channels[i], channels[i], num_knn[i], min(i // 4 + 1, max_dilation), 
                                        conv, act, norm,
                                        bias, stochastic, epsilon, 1, n=192, drop_path=dpr[i],
                                        relative_pos=relative_pos, roi_centers=roi_centers, downsample=True),
                                     FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[i]))]
                else:
                    backbone += [Seq(Grapher(channels[i], channels[i], num_knn[i], min(i // 4 + 1, max_dilation), 
                                        conv, act, norm,
                                        bias, stochastic, epsilon, 1, n=192, drop_path=dpr[i],
                                        relative_pos=relative_pos, roi_centers=roi_centers),
                                     FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[i]))]

            self.backbone[f'stage{i}'] = Seq(*backbone)
            
        self.pool = AveragePool1d()
        self.prediction = nn.Conv1d(num_rois, opt.n_classes, 1, bias=True)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def obtain_node_embedding(self, feature_map, contribution_maps):
        """
        Obtains node embeddings for each ROI by performing weighted sum of feature map with contribution maps

        Args:
            feature_map (torch.Tensor): Feature map of shape (B, C', H', W', D').
            contribution_maps (torch.Tensor): ROI mask of shape (1, num_rois, H', W', D').

        Returns:
            node_embedding (torch.Tensor): Node embedding of shape (B, num_rois, C')
        """
        
        B, C_prime, H_prime, W_prime, D_prime = feature_map.shape
        _, num_rois, _, _, _ = contribution_maps.shape

        # Expand contribution maps to match the feature map shape for broadcasting
        contribution_maps = contribution_maps.expand(B, -1, -1, -1, -1)  # Shape: (B, num_rois, H', W', D')

        # Add a channel dimension to contribution maps to align with feature_map for broadcasting
        contribution_maps = contribution_maps.unsqueeze(1)  # Shape: (B, 1, num_rois, H', W', D')

        # Add an ROI dimension to the feature map for broadcasting
        feature_map = feature_map.unsqueeze(2)  # Shape: (B, C', 1, H', W', D')

        # Compute weighted feature map for all ROIs in one operation
        weighted_features = feature_map * contribution_maps  # Shape: (B, C', num_rois, H', W', D')

        # Sum over spatial dimensions (H', W', D') to get total weighted features per ROI
        weighted_sum = weighted_features.sum(dim=(3, 4, 5))  # Shape: (B, C', num_rois)

        # Compute total contribution per ROI for normalization
        total_contributions = contribution_maps.sum(dim=(3, 4, 5))  # Shape: (B, 1, num_rois)
        # Normalize weighted sum by total contributions
        node_embeddings = weighted_sum / (total_contributions + 1e-8)  # Shape: (B, C', num_rois)

        return node_embeddings
    
    def forward(self, inputs):
        
        features = self.features(inputs)

        stage0_node_emb = self.obtain_node_embedding(features[0], self.contribution_maps[f'stage0'])
        stage0_node_emb = self.backbone[f'stage0'](stage0_node_emb)
        stage0_node_emb = stage0_node_emb + self.pos_embed
        
        stage1_node_emb = self.obtain_node_embedding(features[1], self.contribution_maps[f'stage1'])
        stage1_node_emb = torch.concat([stage0_node_emb, stage1_node_emb], dim=1)
        stage1_node_emb = self.backbone[f'stage1'](stage1_node_emb)
        
        stage2_node_emb = self.obtain_node_embedding(features[2], self.contribution_maps[f'stage2'])
        stage2_node_emb = torch.concat([stage1_node_emb, stage2_node_emb], dim=1)
        stage2_node_emb = self.backbone[f'stage2'](stage2_node_emb)
        
        stage3_node_emb = self.obtain_node_embedding(features[3], self.contribution_maps[f'stage3'])
        stage3_node_emb = torch.concat([stage2_node_emb, stage3_node_emb], dim=1)
        stage3_node_emb = self.backbone[f'stage3'](stage3_node_emb)
        
        pooled = self.pool(stage3_node_emb)    
        preds = self.prediction(pooled).squeeze(2)
        return preds 
            
            
    def get_normalized_roi_centers(self, one_hot_mask, batch_size):
        """
        Compute the normalized ROI centroids.

        Args:
            one_hot_mask (Tensor): Binary mask of shape [num_rois, 91, 109, 91] indicating the ROIs.
            batch_size (int): Number of batches for output consistency.

        Returns:
            roi_centers (Tensor): Normalized ROI centroids of shape [batch_size, num_rois, 3].
        """
        num_rois, D, H, W = one_hot_mask.shape  # D=91, H=109, W=91
        
        # Create coordinate grids for each dimension
        z_coords = torch.arange(D, device=one_hot_mask.device).view(D, 1, 1).expand(D, H, W)
        y_coords = torch.arange(H, device=one_hot_mask.device).view(1, H, 1).expand(D, H, W)
        x_coords = torch.arange(W, device=one_hot_mask.device).view(1, 1, W).expand(D, H, W)
        
        # Compute weighted sum of coordinates (centroids)
        sum_x = (one_hot_mask * x_coords).sum(dim=(1, 2, 3))  # [num_rois]
        sum_y = (one_hot_mask * y_coords).sum(dim=(1, 2, 3))
        sum_z = (one_hot_mask * z_coords).sum(dim=(1, 2, 3))

        # Get the number of voxels per ROI to compute mean coordinates
        roi_voxel_count = one_hot_mask.sum(dim=(1, 2, 3)).clamp(min=1)  # Avoid division by zero
        
        centroid_x = sum_x / roi_voxel_count  # [num_rois]
        centroid_y = sum_y / roi_voxel_count
        centroid_z = sum_z / roi_voxel_count
        
        # Normalize centroids to [0, 1] range
        centroid_x /= W
        centroid_y /= H
        centroid_z /= D
        
        # Stack centroids and expand for batch dimension
        roi_centers = torch.stack([centroid_x, centroid_y, centroid_z], dim=-1)  # [num_rois, 3]
        roi_centers = roi_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_rois, 3]

        return roi_centers
    
    
    def get_pooled_features(self, inputs):
        
        stage0, stage1, stage2, stage3 = self.features(inputs)
        # print(stage0.shape, stage1.shape, stage2.shape, stage3.shape, stage4.shape)
        # features = self.features(inputs)
        
        stage0_node_emb = self.obtain_node_embedding(stage0, self.contribution_maps[f'stage0'])
        stage0_node_emb = self.backbone[f'stage0'](stage0_node_emb)
        stage0_node_emb = stage0_node_emb + self.pos_embed
        
        stage1_node_emb = self.obtain_node_embedding(stage1, self.contribution_maps[f'stage1'])
        stage1_node_emb = torch.concat([stage0_node_emb, stage1_node_emb], dim=1)
        stage1_node_emb = self.backbone[f'stage1'](stage1_node_emb)
        
        stage2_node_emb = self.obtain_node_embedding(stage2, self.contribution_maps[f'stage2'])
        stage2_node_emb = torch.concat([stage1_node_emb, stage2_node_emb], dim=1)
        stage2_node_emb = self.backbone[f'stage2'](stage2_node_emb)
        
        stage3_node_emb = self.obtain_node_embedding(stage3, self.contribution_maps[f'stage3'])
        stage3_node_emb = torch.concat([stage2_node_emb, stage3_node_emb], dim=1)
        stage3_node_emb = self.backbone[f'stage3'](stage3_node_emb)
        
        pooled = self.pool(stage3_node_emb)    
        preds = self.prediction(pooled).squeeze(2)
        
        return pooled, preds