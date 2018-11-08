# coding:utf8
# @Time    : 18-11-8 下午2:40
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com

from torch import nn
from torch import autograd
import torch
from torch import optim

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.nn import MSELoss
import warnings
warnings.filterwarnings("ignore")

def _score2(pm, pp, plant):
    plant_power = {
        1: 10,
        2: 10,
        3: 40,
        4: 50
    }
    threshold = plant_power[plant] * 0.03
    index = pm >= threshold
    return np.abs(pm[index] - pp[index]).sum() / (np.sum(index) * plant_power[plant])

def load_dataset(plant):
    print(f'loading plant {plant} data')
    train = pd.read_csv(f'/home/zhouzr/project/competition/gngf/data/train_{plant}.csv', parse_dates=["时间"]).drop_duplicates().reset_index(drop=True)
    test = pd.read_csv(f'/home/zhouzr/project/competition/gngf/data/test_{plant}.csv', parse_dates=["时间"])
    train.columns = ['time', 'irr', 'ws', 'wd', 'temp', 'pr', 'hm', 'mirr', 'power']
    test.columns = ['id', 'time', 'irr', 'ws', 'wd', 'temp', 'pr', 'hm']
    data = pd.concat([train, test])
    return data

p1 = load_dataset(1)
train = p1.id.isnull()
test = p1.power.isnull()
p1['wd'] = MinMaxScaler((-1, 1)).fit_transform(p1['wd'].values.reshape(-1,1))
power_scaler = StandardScaler(with_mean=0, with_std=1.).fit(p1[train]['power'].values.reshape(-1,1))
p1.loc[train, 'power'] = power_scaler.transform(p1.loc[train, 'power'].values.reshape(-1,1))


class SingleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x, hn = self.lstm(x)
        x = self.fc(x)
        x = self.relu(x[-1, :, :].squeeze())
        return x

def get_train_test(x, seq, features):
    train = x[x.id.isnull()].reset_index(drop=True)
    test = x[x.id.notnull()].reset_index(drop=True)
    train_size, test_size = train.shape[0], test.shape[0]
    train_x, test_x = [], []
    train_y = train['power'].values[seq:]
    for i in range(seq, train_size):
        train_x.append(x.iloc[i-seq: i][features].values)
    for i in range(train_size, test_size + train_size):
        test_x.append(x.iloc[i-seq: i][features].values)
    return np.stack(train_x, axis=1), np.stack(test_x, axis=1), train_y

train_x, test_x, train_y = get_train_test(p1, 100, features=['hm', 'irr', 'pr', 'temp', 'wd', 'ws'])

train_x = torch.tensor(train_x).float()
test_x = torch.tensor(test_x).float()
train_y = torch.tensor(train_y).float()

def generate_batch(x, y, batch_size, shuffle=True):
    n_sample = x.shape[1]
    idx = np.array(range(n_sample))
    if shuffle:
        np.random.shuffle(idx)
    iterations = n_sample // batch_size
    for step in range(iterations):
        yield (x[:, idx[batch_size*step: batch_size*(step+1)], :],
               y[idx[batch_size*step: batch_size*(step+1)]], step)

epochs = 5
SEQLEN = 100
INPUT_SIZE = 10
batch_size = 4

lstm = SingleLSTM(6, 100)
optimizer = optim.Adam(lstm.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

for epoch in range(1, epochs+1):
    for (batch_x, batch_y, step) in generate_batch(train_x, train_y, batch_size):
        optimizer.zero_grad()
        pred_y = lstm(batch_x)
        # pred_y = pred_y.detach()
        loss = loss_func(batch_y, pred_y)
        if step % 4**3:
            print(f'epoch: {epoch}, step: {step}, loss: {loss: .4f}')
        loss.backward()
        optimizer.step()
