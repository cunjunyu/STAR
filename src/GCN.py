import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adjacency_matrix(adjacency_matrix):
    # Add self-loops
    adjacency_matrix = adjacency_matrix + torch.eye(adjacency_matrix.size(0)).to(adjacency_matrix.device)
    # Degree matrix
    degree_matrix = torch.diag(adjacency_matrix.sum(1))
    # Inverse square root of degree matrix
    inv_sqrt_degree = torch.diag(torch.pow(adjacency_matrix.sum(1), -0.5))
    # Symmetric normalized adjacency matrix
    normalized_adjacency = torch.mm(torch.mm(inv_sqrt_degree, adjacency_matrix), inv_sqrt_degree)
    return normalized_adjacency


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0, activation=F.relu):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency_matrix):
        # x: [num_nodes, in_features]
        # adjacency_matrix: [num_nodes, num_nodes]
        support = self.linear(x)
        out = torch.matmul(adjacency_matrix, support)
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        return out

class GCNEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0):
        super(GCNEncoder, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features, dropout, activation=F.relu)
        self.gcn2 = GCNLayer(hidden_features, out_features, dropout, activation=None)

    def forward(self, x, adjacency_matrix):
        adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)
        x = self.gcn1(x, adjacency_matrix)
        x = self.gcn2(x, adjacency_matrix)
        return x  # [num_nodes, out_features]
