import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.autograd import Variable

from utils import to_line_graph_features


class NodeConvGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, num_convolutions=2):
        if hidden_dim is None:
            hidden_dim = input_dim

        super(NodeConvGNN, self).__init__()
        conv = tg.nn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv(input_dim, hidden_dim, cached=False))
        for _ in range(1, num_convolutions):
            self.convs.append(conv(hidden_dim, hidden_dim, cached=False))
              
        self.linear = nn.Linear(2*hidden_dim, output_dim)

    def forward(self, x, g_edge_index, lg_edge_index, index01):
        # Do node convolutions
        for conv in self.convs:
            x = F.relu(conv(x, g_edge_index))

        # Convert node features to line graph features
        x = to_line_graph_features(g_edge_index, x)

        # Filter the output of the 0-1 edge
        x = x[index01].unsqueeze(0)

        # Generate network outputs
        # sigmoid so we can use binary cross entropy loss
        x = F.sigmoid(self.linear(x))

        return x
        

class EdgeConvGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=None, output_dim=1, num_convolutions=2):
        if hidden_dim is None:
            hidden_dim = input_dim

        super(EdgeConvGNN, self).__init__()
        conv = tg.nn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv

        self.convs = nn.ModuleList()
        self.convs.append(conv(input_dim*2, hidden_dim, cached=False))
        for _ in range(1, num_convolutions):
            self.convs.append(conv(hidden_dim, hidden_dim, cached=False))
              
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, g_edge_index, lg_edge_index, index01):
        # Convert node features to line graph features
        x = to_line_graph_features(g_edge_index, x)

        # Do edge convolutions
        for conv in self.convs:
            x = F.relu(conv(x, lg_edge_index))

        # Filter the output of the 0-1 edge
        x = x[index01].unsqueeze(0)

        # Generate network outputs
        # sigmoid so we can use binary cross entropy loss
        x = F.sigmoid(self.linear(x))

        return x