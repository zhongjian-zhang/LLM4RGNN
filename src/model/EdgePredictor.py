# !/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   EdgePredictor.py
@Time    :   2024/3/28 11:14
@Author  :   zhongjian zhang
"""
import torch.nn as nn
import torch.nn.functional as F

class EdgePredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_class=1, num_layers=3, dropout=0.5):
        super(EdgePredictor, self).__init__()
        self.mlp = MLP(embedding_dim, hidden_dim, num_class, num_layers, dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.convs.append(nn.Linear(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.activations.append(nn.RReLU())
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.activations.append(nn.RReLU())
        self.convs.append(nn.Linear(hidden_channels, out_channels))
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = self.activations[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)
        return x
