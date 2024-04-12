#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 11:31:24 2019

@author: lei.cai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch.autograd import Variable

from utils import to_line_graph_features


class NodeConvGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=52, output_dim=1, num_convolutions=2):
        super(NodeConvGNN, self).__init__()
        conv = tg.nn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv
        self.convs = nn.ModuleList()
        self.convs.append(conv(input_dim, hidden_dim, cached=False))
        for _ in range(1, num_convolutions):
            self.convs.append(conv(hidden_dim, hidden_dim, cached=False))
              
        self.linear1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, g_edge_index, lg_edge_index):
        # Do node convolutions
        for conv in self.convs:
            x = F.relu(conv(x, g_edge_index))

        # Convert node features to line graph features
        x = to_line_graph_features(g_edge_index, x)
        
        # First linear layer to get rich edge features
        x = F.relu(self.linear1(x))

        # Filter the output of the 0-1 edge
        # TODO
        raise Exception()

        # Generate network outputs
        x = self.linear2(x)

        return x
        

class EdgeConvGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=52, output_dim=1, num_convolutions=2):
        super(EdgeConvGNN, self).__init__()
        conv = tg.nn.GCNConv  # SplineConv  NNConv   GraphConv   SAGEConv

        self.linear1 = nn.Linear(2*input_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(0, num_convolutions):
            self.convs.append(conv(hidden_dim, hidden_dim, cached=False))
              
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, g_edge_index, lg_edge_index):
        # Convert node features to line graph features
        x = to_line_graph_features(g_edge_index, x)

        # First linear layer to get edge features
        x = F.relu(self.linear1(x))

        # Do edge convolutions
        for conv in self.convs:
            x = F.relu(conv(x, lg_edge_index))

        # Filter the output of the 0-1 edge
        # TODO
        raise Exception()

        # Generate network outputs
        x = self.linear2(x)

        return x