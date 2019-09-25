import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add


embedding = nn.Embedding(1500, 16)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(16, 32)
        self.conv2 = GCNConv(32, 41)

    def forward(self, data):
        x, edge_index = embedding(torch.squeeze(data.x)), data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Graph level read out by scattered sum
        x = scatter_add(x, data.batch, dim=0)

        return x
