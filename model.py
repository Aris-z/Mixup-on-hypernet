import torch
import torch.nn as nn
from torch.nn import functional as F
from CNNBlock import CNNBlock
from WeightGeneration import Weight_Generation
import random
from utils import *


class model(nn.Module):
    def __init__(self, weight_net1, weight_net2, mix_alpha, *args):
        super(model, self).__init__()
        self.mix_alpha = mix_alpha
        self.filter_size_list = \
            [[16, 16], [16, 16], [16, 16], [16, 16], [16, 16], [16, 16],
             [16, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32],
             [32, 64], [64, 64], [64, 64], [64, 64], [64, 64], [64, 64]]

        self.CNNnet = nn.ModuleList()
        for i in range(18):
            self.CNNnet.append(CNNBlock(self.filter_size_list[i][0], self.filter_size_list[i][1]))
        self.lam = torch.randn(1)
        
        self.weight_net1 = weight_net1
        self.weight_net2 = weight_net2
        self.avg = nn.AvgPool2d(32)
        self.output = nn.Linear(64, 10)

    def forward(self, x1, x2, batch_size, layer_mix=None, *args):
        weight_list_1, x1 = self.weight_net1(x1)  # 先让model只返回权重参数列表，然后在这个地方继续进行后面的mix操作
        weight_list_2, x2 = self.weight_net2(x2)
        mix_weight_list = []
        # layer_mix每一个epoch随机但是同一个epoch内是固定的
        if not layer_mix:
            layer_mix = random.randint(0, 2)
            
        if layer_mix == 0:
            mixed_x, self.lam[0] = manifold_mixup_data_demo(x1, x2, self.mix_alpha)
            for k in range(36):
                mixed_weight = manifold_mixup_weight(weight_list_1[k], weight_list_2[k], self.lam[0].item())
                mix_weight_list.append(mixed_weight)
            for i in range(0, 18):
                mixed_x = self.CNNnet[i](mixed_x, mix_weight_list[2 * i], mix_weight_list[2 * i + 1])

        if layer_mix == 1:
            for m in range(0, 6):
                x1 = self.CNNnet[m](x1, weight_list_1[2 * m], weight_list_2[2 * m + 1])
                x2 = self.CNNnet[m](x2, weight_list_2[2 * m], weight_list_2[2 * m + 1])
            mixed_x, self.lam[0] = manifold_mixup_data_demo(x1, x2, self.mix_alpha)
            for k in range(12, 36):
                mixed_weight = manifold_mixup_weight(weight_list_1[k], weight_list_2[k], self.lam[0].item())
                mix_weight_list.append(mixed_weight)
            for i in range(6, 18):
                mixed_x = self.CNNnet[i](mixed_x, mix_weight_list[2 * (i - 6)], mix_weight_list[2 * (i - 6) + 1])

        if layer_mix == 2:
            for m in range(0, 12):
                x1 = self.CNNnet[m](x1, weight_list_1[2 * m], weight_list_2[2 * m + 1])
                x2 = self.CNNnet[m](x2, weight_list_2[2 * m], weight_list_2[2 * m + 1])
            mixed_x, self.lam[0] = manifold_mixup_data_demo(x1, x2, self.mix_alpha)
            for k in range(24, 36):
                mixed_weight = manifold_mixup_weight(weight_list_1[k], weight_list_2[k], self.lam[0].item())
                mix_weight_list.append(mixed_weight)
            for i in range(12, 18):
                mixed_x = self.CNNnet[i](mixed_x, mix_weight_list[2 * (i - 12)], mix_weight_list[2 * (i - 12) + 1])
        
        outputs = self.output(self.avg(mixed_x).reshape(-1, 64))
        self.lam = self.lam.cuda()
        return outputs, self.lam
