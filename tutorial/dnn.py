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

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        h1 = nn.Linear(len(X_features), 50)
        h2 = nn.Linear(50, 35)
        h3 = nn.Linear(35, 1)
        self.hidden = nn.Sequential(
            h1,
            nn.Tanh(),
            h2,
            nn.Tanh(),
            h3,
        )
        if use_cuda:
            self.hidden = self.hidden.cuda()

    def forward(self, x):
        o = self.hidden(x)
        return o.view(-1)


if __name__ == "__main__":

    # 1. data
    trn = pd.read_csv("./dnn_data/trn.tsv", sep='\t')
    val = pd.read_csv("./dnn_data/val.tsv", sep='\t')

    X_features = ["feature_1", "feature_2", "feature_3", "feature_4",
                  "feature_5", "feature_6", "feature_7", "feature_8"]
    y_feature = ["y"]

    trn_X_pd, trn_y_pd = trn[X_features], trn[y_feature]
    val_X_pd, val_y_pd = val[X_features], val[y_feature]

    trn_X = torch.from_numpy(trn_X_pd.astype(np.float32).values)
    trn_y = torch.from_numpy(trn_y_pd.astype(np.float32).values)
    trn_y = trn_y.view(-1)

    val_X = torch.from_numpy(val_X_pd.astype(np.float32).values)
    val_y = torch.from_numpy(val_y_pd.astype(np.float32).values)
    val_y = val_y.view(-1)

    # 2. train
    batch_size = 64
    trn = data_utils.TensorDataset(trn_X, trn_y)
    trn_loader = data_utils.DataLoader(trn, batch_size=batch_size, shuffle=True)

    val = data_utils.TensorDataset(val_X, val_y)
    val_loader = data_utils.DataLoader(val, batch_size=batch_size, shuffle=False)

    # 3. parameter
    use_cuda = torch.cuda.is_available()

    model = MLPRegressor()
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 10
    num_batches = len(trn_loader)

    # 4. train
    # 4. train
    trn_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        trn_loss_summary = 0.0
        for i, trn in enumerate(trn_loader):
            trn_X, trn_y = trn[0], trn[1]
            trn_X.float
            if use_cuda:
                trn_X, trn_y = trn_X.cuda(), trn_y.cuda()
            optimizer.zero_grad()
            trn_pred = model(trn_X)
            trn_loss = criterion(trn_pred, trn_y)
            trn_loss.backward()
            optimizer.step()

            trn_loss_summary += trn_loss

            if (i + 1) % 15 == 0:
                with torch.no_grad():
                    val_loss_summary = 0.0
                    for j, val in enumerate(val_loader):
                        val_X, val_y = val[0], val[1]
                        if use_cuda:
                            val_X, val_y = val_X.cuda(), val_y.cuda()
                        val_pred = model(val_X)
                        val_loss = criterion(val_pred, val_y)
                        val_loss_summary += val_loss

                print("epoch: {}/{} | step: {}/{} | trn_loss: {:.4f} | val_loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, num_batches, (trn_loss_summary / 15) ** (1 / 2),
                    (val_loss_summary / len(val_loader)) ** (1 / 2)
                ))

                trn_loss_list.append((trn_loss_summary / 15) ** (1 / 2))
                val_loss_list.append((val_loss_summary / len(val_loader)) ** (1 / 2))
                trn_loss_summary = 0.0

    print("finish Training")

    plt.figure(figsize=(16, 9))
    x_range = range(len(trn_loss_list))
    plt.plot(x_range, trn_loss_list, label="trn")
    plt.plot(x_range, val_loss_list, label="val")
    plt.legend()
    plt.xlabel("training steps")
    plt.ylabel("loss")
    plt.show()
