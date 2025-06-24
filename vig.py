# 2022.10.31-Changed for building ViG model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from gcn_lib import Grapher, act_layer

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


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


class Stem(nn.Module):
    """ Image to Visual Word Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, in_dim=1, out_dim=96, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_dim, out_dim//8, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim//8),
            act_layer(act),
            nn.Conv3d(out_dim//8, out_dim//4, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim//4),
            act_layer(act),
            nn.Conv3d(out_dim//4, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim//2),
            act_layer(act),
            nn.Conv3d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm3d(out_dim),
            act_layer(act),
            nn.Conv3d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm3d(out_dim),
        )
        
        
    def forward(self, x):
        x = self.convs(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        channels = opt.n_filters
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        self.n_blocks = opt.n_blocks
        drop_path = opt.drop_path
        
        self.stem = Stem(out_dim=channels, act=act)

        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 
        print('dpr', dpr)
        num_knn = [int(x.item()) for x in torch.linspace(k, 2*k, self.n_blocks)]  # number of knn's k
        print('num_knn', num_knn)
        max_dilation = 196 // max(num_knn)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, channels, 23, 28, 23))

        if opt.use_dilation:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], min(i // 4 + 1, max_dilation), conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])
        else:
            self.backbone = Seq(*[Seq(Grapher(channels, num_knn[i], 1, conv, act, norm,
                                                bias, stochastic, epsilon, 1, drop_path=dpr[i]),
                                      FFN(channels, channels * 4, act=act, drop_path=dpr[i])
                                     ) for i in range(self.n_blocks)])

        self.prediction = Seq(nn.Conv1d(channels, 1024, 1, bias=True),
                              nn.BatchNorm1d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv1d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        x = self.stem(inputs).flatten(start_dim=2)
        
        for i in range(self.n_blocks):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool1d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)

