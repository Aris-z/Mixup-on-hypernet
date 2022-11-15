import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # 暂时先不加残差块...用一个浅一点的CNN

        self.batchnorm1 = nn.BatchNorm2d(self.output_size)
        self.batchnorm2 = nn.BatchNorm2d(self.output_size)

    # Artificial parameter CNN
    def forward(self, x, conv_w1, conv_w2):
        output1 = F.relu(self.batchnorm1(F.conv2d(x, conv_w1, padding=1)))
        output2 = F.relu(self.batchnorm2(F.conv2d(output1, conv_w2, padding=1)))

        return output2
