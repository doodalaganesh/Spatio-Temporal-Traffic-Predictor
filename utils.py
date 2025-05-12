
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter, LayerNorm, BatchNorm1d


class SATT(nn.Module):
   
    def __init__(self, c_in, num_nodes, tem_size):
      
        super(SATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(tem_size, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)

   
        self.w = nn.Parameter(torch.rand(tem_size, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
       

        self.b = nn.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=True)
        

        self.v = nn.Parameter(torch.rand(num_nodes, num_nodes), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)
   

  
    def forward(self, seq):

        c1 = seq
        f1 = self.conv1(c1).squeeze(1)  

        c2 = seq.permute(0, 3, 1, 2)  
        f2 = self.conv2(c2).squeeze(1) 

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
     
        logits = torch.matmul(self.v, logits)
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
       
        return coefs



class cheby_conv_ds(nn.Module):
    def __init__(self, c_in, c_out, K):
        super(cheby_conv_ds, self).__init__()
        c_in_new = (K) * c_in
        self.conv1 = Conv2d(c_in_new, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.K = K

    def forward(self, x, adj, ds):
        nSample, feat_in, nNode, length = x.shape # 32, 3/64 , 307 , 24/12
        Ls = []
        L0 = torch.eye(nNode).cuda()
        L1 = adj

        L = ds * adj
        I = ds * torch.eye(nNode).cuda()
        Ls.append(I)
        Ls.append(L)
        for k in range(2, self.K):
            L2 = 2 * torch.matmul(adj, L1) - L0
            L0, L1 = L1, L2
            L3 = ds * L2
            Ls.append(L3)

        Lap = torch.stack(Ls, 1)  # [B, K,nNode, nNode]
        # print(Lap)
        Lap = Lap.transpose(-1, -2)
        x = torch.einsum('bcnl,bknq->bckql', x, Lap).contiguous()
        x = x.view(nSample, -1, nNode, length)
        out = self.conv1(x)
        # print('out=', out.shape)
        return out

  
class TATT(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT, self).__init__()
        self.conv1 = Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, seq):
       
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
       
        f1 = self.conv1(c1).squeeze(1) 

        c2 = seq.permute(0, 2, 1, 3)  
        f2 = self.conv2(c2).squeeze(1)  

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        
        logits = torch.matmul(self.v, logits)
        
        a, _ = torch.max(logits, 1, True)
        logits = logits - a
        coefs = torch.softmax(logits, -1)
       
        return coefs



class ST_BLOCK_0(nn.Module):
    def __init__(self, c_in, c_out, num_nodes, tem_size, K, Kt):
        super(ST_BLOCK_0, self).__init__()

        self.conv1 = Conv2d(c_in, c_out, kernel_size=(1, 1),
                            stride=(1, 1), bias=True)
        self.TATT = TATT(c_in, num_nodes, tem_size)
        self.SATT = SATT(c_in, num_nodes, tem_size)
        self.dynamic_gcn = cheby_conv_ds(c_in, c_out, K)
        self.K = K
        self.time_conv = Conv2d(c_out, c_out, kernel_size=(1, Kt), padding=(0, 1),
                                stride=(1, 1), bias=True)
    
        self.bn = LayerNorm([c_out, num_nodes, tem_size])

    def forward(self, x, supports):
        x_input = self.conv1(x)
        T_coef = self.TATT(x)
        T_coef = T_coef.transpose(-1, -2)
        x_TAt = torch.einsum('bcnl,blq->bcnq', x, T_coef)
        S_coef = self.SATT(x)  # B x N x N
      
        spatial_gcn = self.dynamic_gcn(x_TAt, supports, S_coef)
        spatial_gcn = torch.relu(spatial_gcn)
        time_conv_output = self.time_conv(spatial_gcn)
        out = self.bn(torch.relu(time_conv_output + x_input))
      
        return out, S_coef, T_coef
