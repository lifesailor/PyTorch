"""
1. 데이터를 불러와서 batch 단위로 생성할 수 있고,
2. PyTorch에서 제공하는 여러가지 함수로 모델을 만들 수 있고,
3. 모델의 최적 파라미터를 학습시킬 수 있다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils


class MyModel(nn.Module):
    def __init__(self, X_dim, y_dim):
        super(MyModel, self).__init__()
        layer1 = nn.Linear(X_dim, 128)
        activation1 = nn.ReLU()
        layer2 = nn.Linear(128, y_dim)
        self.module = nn.Sequential(
            layer1,
            activation1,
            layer2
        )

    def forward(self, x):
        out = self.module(x)
        result = F.softmax(out, dim=1)
        return result


criterion = nn.CrossEntropyLoss()
learninig_rate = 1e-5
optimizer = optim.SGD(model.parameters())