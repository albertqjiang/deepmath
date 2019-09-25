import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv
from torch_scatter import scatter_add


class Net(torch.nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        sd = params["state_dimension"]
        # 1340 is the maximum number of vocabs for human proofs
        self.embedding = nn.Embedding(1340, sd)
        self.conv1 = GCNConv(sd, sd * 2)
        self.conv2 = GCNConv(sd * 2, 41)

    def forward(self, data):
        x, edge_index = self.embedding(torch.squeeze(data.x)), data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        # Graph level read out by scattered sum
        x = scatter_add(x, data.batch, dim=0)
        return x


class GNN(torch.nn.Module):
    def __init__(self, params):
        super(GNN, self).__init__()
        sd = params["state_dimension"]
        hl = params["hidden_layers"]
        cuda = params["cuda"]
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # 1340 is the maximum number of vocabs for human proofs
        self.embedding = nn.Embedding(1340, sd)
        self.nn_in = nn.Sequential(
            nn.Linear(sd, sd),
            nn.ReLU(inplace=True),
            nn.Linear(sd, sd),
        )
        self.gin_in = GINConv(self.nn_in)
        self.bn_in = nn.BatchNorm1d(sd)

        self.nn_hidden = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(sd, sd),
                    nn.ReLU(inplace=True),
                    nn.Linear(sd, sd),
                ) for _ in range(hl)
            ]
        )
        self.gin_hidden = nn.ModuleList(
            [GINConv(hidden_nn) for hidden_nn in self.nn_hidden]
        )
        self.bn_hidden = nn.ModuleList(
            [
                nn.BatchNorm1d(sd) for _ in range(len(self.gin_hidden))
            ]
        )

        self.nn_out = nn.Sequential(
            nn.Linear(sd, sd),
            nn.ReLU(inplace=True),
            nn.Linear(sd, sd),
        )
        self.gin_out = GINConv(self.nn_out)

        self.fc1 = nn.Linear(sd, sd)
        self.fc2 = nn.Linear(sd, 41)
        self.to(device)

    def forward(self, data):
        x, edge_index = self.embedding(torch.squeeze(data.x)), data.edge_index
        x = F.relu(self.gin_in(x, edge_index))
        x = self.bn_in(x)
        for i in range(len(self.nn_hidden)):
            x = F.relu(self.gin_hidden[i](x, edge_index))
            x = self.bn_hidden[i](x)
        x = F.relu(self.gin_out(x, edge_index))

        # Graph level read out by scattered sum
        x = scatter_add(x, data.batch, dim=0)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


class GRN(torch.nn.Module):
    def __init__(self, params):
        super(GRN, self).__init__()
        sd = params["state_dimension"]
        self.hl = params["hidden_layers"]
        cuda = params["cuda"]
        device = torch.device("cuda") if cuda else torch.device("cpu")
        # 1340 is the maximum number of vocabs for human proofs
        self.embedding = nn.Embedding(1340, sd)
        self.nn_in = nn.Sequential(
            nn.Linear(sd, sd),
            nn.ReLU(inplace=True),
            nn.Linear(sd, sd),
        )
        self.gin_in = GINConv(self.nn_in)
        self.bn_in = nn.BatchNorm1d(sd)

        self.nn_out = nn.Sequential(
            nn.Linear(sd, sd),
            nn.ReLU(inplace=True),
            nn.Linear(sd, sd),
        )
        self.gin_out = GINConv(self.nn_out)

        self.fc1 = nn.Linear(sd, sd)
        self.fc2 = nn.Linear(sd, 41)
        self.to(device)

    def forward(self, data):
        x, edge_index = self.embedding(torch.squeeze(data.x)), data.edge_index
        x = F.relu(self.gin_in(x, edge_index))
        x = self.bn_in(x)
        for i in range(self.hl):
            x = F.relu(self.gin_in(x, edge_index))
            x = self.bn_in(x)
        x = F.relu(self.gin_out(x, edge_index))

        # Graph level read out by scattered sum
        x = scatter_add(x, data.batch, dim=0)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x
