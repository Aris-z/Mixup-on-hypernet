import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Hypernetwork import HyperNet
from CNNBlock import CNNBlock
import random
from utils import *


class Embedding(nn.Module):
    def __init__(self, z_num, z_dim, device_ids):
        super(Embedding, self).__init__()
        self.z_num = z_num
        self.z_dim = z_dim
        self.z_list = nn.ParameterList()

        self.out_num_z, self.in_num_z = self.z_num

        for i in range(self.out_num_z):
            for j in range(self.in_num_z):
                self.z_list.append(Parameter(torch.randn(self.z_dim).cuda(device=device_ids[0])))

    def forward(self, hypernet):
        w_one_layer = []
        for i in range(self.out_num_z):
            w_slice = []
            for j in range(self.in_num_z):
                w_slice.append(hypernet(self.z_list[i*self.in_num_z + j]))
            w_one_layer.append(torch.cat(w_slice, dim=1))
        return torch.cat(w_one_layer, dim=0)


class Weight_Generation(nn.Module):
    def __init__(self, device_ids, z_dim=64, input_size=16, output_size=16, filter_size=3):
        super(Weight_Generation, self).__init__()
        self.z_dim = z_dim

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.hypernet = HyperNet(input_size=input_size, output_size=output_size,
                                 filter_size=filter_size, z_dim=self.z_dim, device_ids=device_ids)

        self.z_num_list = \
            [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
             [2, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2],
             [4, 2], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4], [4, 4]]

        self.filter_size_list = \
            [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16],
             [16, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32],
             [32, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]

        self.CNNnet = nn.ModuleList()
        for i in range(18):
            self.CNNnet.append(CNNBlock(self.filter_size_list[i][0], self.filter_size_list[i][1]))

        self.hyper = nn.ModuleList()
        for i in range(36):
            self.hyper.append(Embedding(self.z_num_list[i], self.z_dim, device_ids))

        self.avg = nn.AvgPool2d(32)
        self.output = nn.Linear(64, 10)
        self.mix_flag = None

    def forward(self, x, mixup_alpha=0.1, layer_mix=None):
        # 3 channels to 16 channels
        x = F.relu(self.batchnorm1(self.conv1(x)))
        weight = []
        for i in range(18):
            weight1 = self.hyper[2 * i](self.hypernet)
            # weight1.shape =
            weight2 = self.hyper[2 * i + 1](self.hypernet)
            # weight2.shape =
            weight.append(weight1)
            weight.append(weight2)
        '''
        if not layer_mix:
            layer_mix = random.randint(0, 2)

        if layer_mix == 0:
            self.mix_flag = True
            mix_x = manifold_mixup_data_demo(x1, x2, mixup_alpha)
            for i in range(18):
                weight1 = self.hyper[2*i](self.hypernet)
                # weight1.shape =
                weight2 = self.hyper[2*i+1](self.hypernet)
                # weight2.shape =
                mix = self.CNNnet[i](x1, weight1, weight2)

        if layer_mix == 1:
            pass
        for i in range(6, 12):
            pass

        if layer_mix == 2:
            pass
        for i in range(12, 18):
            pass

        if layer_mix == 3:
            pass
            
#        print(x.shape)
        x1 = self.output(self.avg(x1).reshape(-1, 64))
        这个模块挪到train函数里面去
        '''
        return weight, x
