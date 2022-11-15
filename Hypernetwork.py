import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class HyperNet(nn.Module):
    def __init__(self, input_size, output_size, z_dim, filter_size, device_ids):
        super(HyperNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.z_dim = z_dim
        self.filter_size = filter_size

        # parameter in 1st layer
        self.w1 = Parameter(torch.randn(self.z_dim, self.z_dim * self.input_size).cuda(device=device_ids[0]))
        self.b1 = Parameter(torch.randn(self.z_dim * self.input_size).cuda(device=device_ids[0]))

        # parameter in 2nd layer
        self.w2 = Parameter(torch.randn(self.z_dim, self.output_size * self.filter_size *
                                        self.filter_size).cuda(device=device_ids[0]))
        self.b2 = Parameter(torch.randn(self.output_size * self.filter_size * self.filter_size).cuda(device=device_ids[0]))

    # hypernet input z in some layer.
    def forward(self, z):
        output_1 = torch.matmul(z, self.w1) + self.b1
        output_1 = output_1.reshape(self.input_size, self.z_dim)
        output_2 = torch.matmul(output_1, self.w2) + self.b2
        return output_2.reshape(self.output_size, self.input_size, self.filter_size, self.filter_size)


