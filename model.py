

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, InstanceNorm2d
from utils import ST_BLOCK_0  

"""
the parameters:
x-> [batch_num,in_channels,num_nodes,tem_size],
输入x-> [ batch数, 通道数, 节点数, 时间],
"""

class STGCN_block(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(STGCN_block, self).__init__()
        self.block1 = ST_BLOCK_0(c_in, c_out, num_nodes, tem_size, K, Kt)
        self.block2 = ST_BLOCK_0(c_out, c_out, num_nodes, tem_size, K, Kt)
        self.final_conv = Conv2d(tem_size, 12, kernel_size=(1, c_out), padding=(0, 0),
                                 stride=(1, 1), bias=True)
        self.w = Parameter(torch.zeros(num_nodes, 12), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

    
    def forward(self, x, supports):
        x, _, _ = self.block1(x, supports)
        x, d_adj, t_adj = self.block2(x, supports)
        x = x.permute(0, 3, 2, 1)
        x = self.final_conv(x).squeeze().permute(0, 2, 1)  # b,n,12
        x = x * self.w
        return x, d_adj, t_adj



class STGCN(nn.Module):
    
    def __init__(self, c_in, c_out, num_nodes, week, day, recent, K, Kt):
        super(STGCN, self).__init__()

        self.block_w = STGCN_block(c_in, c_out, num_nodes, week, K, Kt)
        
        self.block_d = STGCN_block(c_in, c_out, num_nodes, day, K, Kt)
        
        self.block_r = STGCN_block(c_in, c_out, num_nodes, recent, K, Kt)
      
        self.bn = BatchNorm2d(c_in, affine=False)


    def forward(self, x_w, x_d, x_r, supports):
        x_w = self.bn(x_w)
        x_d = self.bn(x_d)
        x_r = self.bn(x_r)
        x_w, _, _ = self.block_w(x_w, supports)
        x_d, _, _ = self.block_d(x_d, supports)
        x_r, d_adj_r, t_adj_r = self.block_r(x_r, supports)
        out = x_w + x_d + x_r
        return out, d_adj_r, t_adj_r
