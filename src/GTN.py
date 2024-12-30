import torch
import torch.nn as nn
import torch.nn.functional as F

class GTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge=1):
        super(GTLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge = num_edge

        # A learnable weight per adjacency / edge type
        # Shape = (num_edge, in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(num_edge, in_channels, out_channels))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adjacency):
        n_nodes = x.size(0)
        device = x.device
        
        # Start with a zero output feature
        out = torch.zeros(n_nodes, self.out_channels, device=device)
        
        for i in range(self.num_edge):
            # (N, in_channels)
            Ax = torch.matmul(adjacency, x)  
            # (N, out_channels)
            AxW = torch.matmul(Ax, self.weight[i])  
            out += AxW
        
        return out


class GTNEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super(GTNEncoder, self).__init__()
        # First GT layer: (in_features -> hidden_features)
        self.gtlayer1 = GTLayer(in_channels=in_features, 
                                out_channels=hidden_features, 
                                num_edge=1)
        
        # Second GT layer: (hidden_features -> out_features)
        self.gtlayer2 = GTLayer(in_channels=hidden_features, 
                                out_channels=out_features, 
                                num_edge=1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adjacency_matrix):
        # Pass through first GT layer
        x = self.gtlayer1(x, adjacency_matrix)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass through second GT layer
        x = self.gtlayer2(x, adjacency_matrix)
        return x
