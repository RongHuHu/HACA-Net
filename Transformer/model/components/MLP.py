import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Mlp(nn.Module):
    def __init__(self, in_channels, output_size, layer_num, mode):
        super(Mlp, self).__init__()
        input_size = (1, in_channels)
        # input_test = torch.ones(input_size)
        self.layer_seq = nn.Sequential()
        for i in range(layer_num - 1):  # minus 1 for output_layer
            self.layer_seq.add_module(f'lmlp_{i}', nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.LayerNorm(in_channels),
                nn.LeakyReLU()))

        if mode == 'classification':
            self.layer_seq.add_module('out_mlp', nn.Sequential(
                nn.Linear(in_channels, output_size),
                nn.Softmax()))
        elif mode == 'regression':
            self.layer_seq.add_module(
                'out_mlp', nn.Linear(in_channels, output_size))

        # print("output_tensor size : ",self.layer_seq(input_test))

    def forward(self, x):
        x = self.layer_seq(x)
        return x

class Times_Channel_Squeeze(nn.Module):
    def __init__(self, in_channels, output_size, layer_num, mode, reduction=16):
        super(Times_Channel_Squeeze, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.layer_seq = nn.Sequential()
        for i in range(layer_num):  
            self.layer_seq.add_module(f'lmlp_{i}', nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.LayerNorm(in_channels // reduction),
                nn.Dropout(p=0.1),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                #nn.LayerNorm(in_channels),
                nn.LeakyReLU())
            )         
        self.fc = nn.Linear(in_channels, output_size)

    def forward(self, x):
        y = x.permute(0,2,1)
        b, c, l = y.size() # (B,C,L)
        y = self.avg_pool(y) # (B,C,L) 通过avg=》 (B,C,1)
        #print('y1',y.size()) # [128, 96, 1]
        y = y.view(b, c)
        #print('y2',y.size()) # [128, 96]
        y = self.layer_seq(y)
        #print('y3',y.size()) # [128, 96]
        y = y.unsqueeze(-2)
        #print('y4',y.size()) # [128, 1, 96]
        x = x * y
        #print('x',x.size()) # [128, 1, 96]
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



