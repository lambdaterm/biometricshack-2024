import torch
import torch.nn as nn


class EmbeddingMapper(nn.Module):
    def __init__(self, input_dim=512, output_dim=512):
        super(EmbeddingMapper, self).__init__()
        self.lin1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024))
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048))
        self.lin3 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024))
        self.lin4 = nn.Linear(1024, output_dim)

    def forward(self, x):
        # print(x.shape)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        return x
