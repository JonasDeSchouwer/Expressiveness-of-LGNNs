from typing import Generator, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import DataLoader

import pickle
import networkx as nx
from random import random
from tqdm import tqdm
import math

from utils import sort_for_unique_edges, subgraph_extraction, to_line_graph


class Graph:
    """
    targets: 0 if the edge is missing, 1 if the edge is present
    """

    edge_index: torch.Tensor
    node_feat: torch.Tensor
    line_edge_index: torch.Tensor
    targets: torch.Tensor

    def __init__(self, edge_index, node_feat, line_edge_index, targets):
        self.edge_index = edge_index
        self.node_feat = node_feat
        self.line_edge_index = line_edge_index
        self.targets = targets


class Dataset():
    train_graphs: List[Graph]
    test_graphs: List[Graph]
    
    def __init__(self, path=None):
        self.train_graphs = None
        self.test_graphs = None
        if path is not None:
            self.load(path)
    
    def save(self, folder):
        with open(f"{folder}/train_graphs.pckl", 'wb') as f:
            pickle.dump(self.train_graphs, f)
        with open(f"{folder}/test_graphs.pckl", 'wb+') as f:
            pickle.dump(self.test_graphs, f)

    def load(self, folder):
        with open(f"{folder}/train_graphs.pckl", 'rb') as f:
            self.train_graphs = pickle.load(f)
        with open(f"{folder}/test_graphs.pckl", 'rb') as f:
            self.test_graphs = pickle.load(f)
    
    def iter_batches(self, batch_size, train=True) -> Generator[List[Graph], None, None]:
        graphs = self.train_graphs if train else self.test_graphs
        n_graphs = len(graphs)
        batch_idxs = torch.randperm(n_graphs)
        batch_idxs = torch.chunk(batch_idxs, math.ceil(n_graphs/batch_size))
        for batch in batch_idxs:
            yield [graphs[i] for i in batch]


def generate_dataset(dataloader, test_ratio=0.1, num_samples_per_graph=3000, pos_ratio=0.5, h=1):
    num_train = math.ceil((1-test_ratio) * len(dataloader))
    num_test = len(dataloader) - num_train
    print(f"dataset contains {len(dataloader)} graphs: using {num_train} for training and {num_test} for testing")

    print("generating training graphs...")
    train_graphs = []
    for i in tqdm(range(num_train)):
        data = dataloader[i]
        train_graphs += sample_pos(data.edge_index, data.x, int(num_samples_per_graph * pos_ratio), h)
        train_graphs += sample_neg(data.edge_index, data.x, int(num_samples_per_graph * (1-pos_ratio)), h)
    
    print("generating test graphs...")
    test_graphs = []
    for i in tqdm(range(num_train, len(dataloader))):
        data = dataloader[i]
        test_graphs += sample_pos(data.edge_index, data.x, int(num_samples_per_graph * pos_ratio), h)
        test_graphs += sample_neg(data.edge_index, data.x, int(num_samples_per_graph * (1-pos_ratio)), h)
    
    dataset = Dataset()
    dataset.train_graphs = train_graphs
    dataset.test_graphs = test_graphs

    print(f"dataset generated: {len(train_graphs)} training graphs and {len(test_graphs)} testing graphs")

    return dataset


def sample_pos(edge_index, node_feat, num_samples, h):
    """
    sample num_samples positive edges from the graph and return the h-hop enclosing subgraphs
    """
    sort_for_unique_edges(edge_index)   # to avoid duplicates

    pos_edges = edge_index[:, :int(len(edge_index[0]) / 2)]
    pos_edge_idxs = torch.randperm(pos_edges.shape[1])[:num_samples]
    pos_edges = pos_edges[:, pos_edge_idxs]

    pos_graphs = []
    for i in tqdm(range(num_samples)):
        subgraph = subgraph_extraction(list(pos_edges[:, i].flatten()), edge_index, node_feat, h)
        line_subgraph_edge_index = to_line_graph(subgraph.edge_index, node_feat.shape[0])
        pos_graphs.append(Graph(subgraph.edge_index, subgraph.x, line_subgraph_edge_index, torch.tensor([1])))

    return pos_graphs


def sample_neg(edge_index, node_feat, num_samples, h):
    """
    sample num_samples negative edges from the graph and return the h-hop enclosing subgraphs
    """
    sort_for_unique_edges(edge_index)   # to avoid duplicates

    neg_edges = []  # list of negative edges, each in the format of a tensor
    for i in range(num_samples):
        neg_edge = torch.tensor([int(random() * node_feat.shape[0]), int(random() * node_feat.shape[0])]).view(2,1)
        while torch.any(torch.all(edge_index == neg_edge, dim=0)):
            neg_edge = torch.tensor([int(random() * node_feat.shape[0]), int(random() * node_feat.shape[0])])
        neg_edges.append(neg_edge)

    neg_graphs = []
    for i in tqdm(range(num_samples)):
        subgraph = subgraph_extraction(list(neg_edges[i].flatten()), edge_index, node_feat, h)
        line_subgraph_edge_index = to_line_graph(subgraph.edge_index, node_feat.shape[0])
        neg_graphs.append(Graph(subgraph.edge_index, subgraph.x, line_subgraph_edge_index, torch.tensor([0])))

    return neg_graphs


if __name__ == "__main__":
    dataloader = tg.datasets.CoraFull(root='data')
    dataset = generate_dataset(dataloader)
    dataset.save('code/data/CoraDataset')
    print("dataset saved")