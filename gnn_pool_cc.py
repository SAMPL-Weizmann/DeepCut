import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNpool(nn.Module):
    def __init__(self, input_dim, conv_hidden, mlp_hidden, num_clusters, device):
        """
        implementation of mincutpool model from: https://arxiv.org/pdf/1907.00481v6.pdf
        @param input_dim: Size of input nodes features
        @param conv_hidden: Size Of conv hidden layers
        @param mlp_hidden: Size of mlp hidden layers
        @param num_clusters: Number of cluster to output
        @param device: Device to run the model on
        """
        super(GNNpool, self).__init__()
        self.device = device
        self.num_clusters = num_clusters
        self.mlp_hidden = mlp_hidden

        # GNN conv
        self.convs = pyg_nn.GCN(input_dim, conv_hidden, 1, act='elu')
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(conv_hidden, mlp_hidden), nn.ELU(), nn.Dropout(0.25),
            nn.Linear(mlp_hidden, self.num_clusters))

    def forward(self, data, A):
        """
        forward pass of the model
        @param data: Graph in Pytorch geometric data format
        @param A: Adjacency matrix of the graph
        @return: Adjacency matrix of the graph and pooled graph (argmax of S)
        """
        x, edge_index, edge_atrr = data.x, data.edge_index, data.edge_attr

        x = self.convs(x, edge_index, edge_atrr)  # applying con5v
        x = F.elu(x)

        # pass feats through mlp
        H = self.mlp(x)

        # cluster assignment for matrix S
        S = F.softmax(H)

        return A, S

    def loss(self, A, S):
        """
        loss calculation, relaxed form of Normalized-cut
        @param A: Adjacency matrix of the graph
        @param S: Polled graph (argmax of S)
        @return: loss value
        """
        # cc loss
        X = torch.matmul(S, S.t())
        cc_loss = -torch.sum(A * X)

        return cc_loss
