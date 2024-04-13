import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.data import DataLoader

import networkx as nx
from random import random
from tqdm import tqdm


def subgraph_extraction(link, edge_index, node_feat, h=1):
    """
    extract the h-hop enclosing subgraph around link `link'
    https://github.com/LeiCaiwsu/LGLP
    """

    # nodes: a list of (unique) nodes in the subgraph
    # nodes_dist: the distance of each node to the target nodes
    # fringe: a set of nodes on the fringe of the subgraph
    # fringe_from_0: the set of nodes that was first reached by expanding from node 0
    # fringe_from_1: the set of nodes that was first reached by expanding from node 1
    dist = 0
    link = link.tolist() if isinstance(link, torch.Tensor) else link
    nodes = [link[0], link[1]]
    fringe_from_0 = set([link[0]])  
    fringe_from_1 = set([link[1]])  
    nodes_dist_to_0 = [0, -1]
    nodes_dist_to_1 = [-1, 0]
    for dist in range(1, h+1):
        fringe_from_0 = neighbors(fringe_from_0, edge_index)
        fringe_from_1 = neighbors(fringe_from_1, edge_index)
        
        fringe_from_0 -= set(nodes)
        fringe_from_1 -= set(nodes)
        
        fringe_from_both = fringe_from_0.intersection(fringe_from_1)
        fringe_from_0 -= fringe_from_both
        fringe_from_1 -= fringe_from_both

        if len(fringe_from_0) == len(fringe_from_1) == len(fringe_from_both) == 0:
            break

        nodes += list(fringe_from_0)
        nodes_dist_to_0 += [dist] * len(fringe_from_0)
        nodes_dist_to_1 += [-1] * len(fringe_from_0)
        nodes += list(fringe_from_1)
        nodes_dist_to_0 += [-1] * len(fringe_from_1)
        nodes_dist_to_1 += [dist] * len(fringe_from_1)
        nodes += list(fringe_from_both)
        nodes_dist_to_0 += [dist] * len(fringe_from_both)
        nodes_dist_to_1 += [dist] * len(fringe_from_both)

    # create and label subgraph
    sub_edge_index, _ = tg.utils.subgraph(nodes, edge_index, relabel_nodes=True, num_nodes=node_feat.shape[0])
    sub_node_feat = torch.cat((
        node_feat[nodes, :],
        torch.tensor(nodes_dist_to_0).view(-1, 1).float(),
        torch.tensor(nodes_dist_to_1).view(-1, 1).float()
    ), dim=1)

    # make sure there is always an edge from node 0 to node 1 in both directions
    edge01 = torch.any(torch.all(sub_edge_index == torch.tensor([[0],[1]]), dim=0))
    edge10 = torch.any(torch.all(sub_edge_index == torch.tensor([[1],[0]]), dim=0))
    if not edge01:
        sub_edge_index = torch.cat((sub_edge_index, torch.tensor([[0],[1]])), dim=1)
    if not edge10:
        sub_edge_index = torch.cat((sub_edge_index, torch.tensor([[1],[0]])), dim=1)

    # make sure the first m edges are each unique
    sub_edge_index = sort_for_unique_edges(sub_edge_index)

    subgraph = tg.data.Data(edge_index=sub_edge_index, x=sub_node_feat)
    return subgraph


def neighbors(fringe, edge_index):
    """
    find all 1-hop neighbors of nodes in fringe from A
    """
    res = set()
    for node in fringe:
        idxs = edge_index[0] == node    # The idxs of the edges for which the source is node
        neighbours = set(torch.unique(edge_index[1][idxs]).tolist())
        res = res.union(neighbours)
    return res


def sort_for_unique_edges(edge_index):
    """
    assuming that the graph has no self-loops and is undirected
    make sure that the first m edges are each unique and the following m edges are the duplicates
    """
    assert edge_index.shape[0] % 2 == 0
    assert not torch.any(edge_index[0] == edge_index[1]), "the graph has self-loops"
    
    first_idxs = edge_index[0] < edge_index[1]
    edge_index_first = edge_index[:, first_idxs]
    edge_index_second = edge_index[:, ~first_idxs]
    edge_index = torch.cat((edge_index_first, edge_index_second), dim=1)

    return edge_index


def to_line_graph(edge_index, num_nodes):
    """
    create the line edge index of the input graph
    """
    unique_edges = edge_index[:, :int(len(edge_index[0]) / 2)]
    assert torch.all(unique_edges[0] < unique_edges[1]), "the graph edges have not been sorted"
    
    num_edges = unique_edges.shape[1]

    # node_occurs_in[i, j] = True if node i occurs in edge j 
    node_occurs_in = torch.zeros((num_nodes, num_edges), dtype=torch.bool)
    node_occurs_in[unique_edges[0], torch.arange(num_edges)] = True
    node_occurs_in[unique_edges[1], torch.arange(num_edges)] = True

    # create the line graph
    line_edges = []
    for node in range(num_nodes):
        edges_with_this_node = torch.where(node_occurs_in[node])[0] #[0] because torch.where returns a tuple
        for i in range(len(edges_with_this_node)):
            for j in range(i+1, len(edges_with_this_node)):
                line_edges.append(torch.tensor([edges_with_this_node[i], edges_with_this_node[j]]))
    if len(line_edges) == 0:
        line_edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        line_edge_index = torch.stack(line_edges, dim=0).t()

    # add the other direction of the edges
    line_edge_index = torch.cat((line_edge_index, line_edge_index[[1,0],:]), dim=1)

    return line_edge_index


def to_line_graph_features(edge_index, node_feat):
    """
    create the node features of the line graph by concatenating the node features of the corresponding endpoints
    """
    unique_edges = edge_index[:, :int(len(edge_index[0]) / 2)]
    assert torch.all(unique_edges[0] < unique_edges[1]), "the graph edges have not been sorted"

    # create the line graph features
    line_node_feat = torch.cat((node_feat[unique_edges[0]], node_feat[unique_edges[1]]), dim=1)

    return line_node_feat


