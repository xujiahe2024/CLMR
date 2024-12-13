import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        """
        通用基础模型类，定义了初始化方法和权重初始化。
        """
        super(Model, self).__init__()

    def initialize(self, m):
        """
        初始化模型权重
        """
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def __init__(self):
        """
        Identity 模块，不做任何变换，直接返回输入
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x
