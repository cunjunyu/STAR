import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.concat = concat  # Whether to apply ELU after output

        # Learnable weight matrix
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention coefficients
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Activation function
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [num_nodes, in_features]
        # adj: [num_nodes, num_nodes]

        Wh = torch.mm(h, self.W)  # [num_nodes, out_features]
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, Wh)  # [num_nodes, out_features]

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh: [num_nodes, out_features]
        num_nodes = Wh.size(0)

        # Self-attention on the nodes - Shared attention mechanism
        Wh1 = Wh.repeat(1, num_nodes).view(num_nodes * num_nodes, -1)
        Wh2 = Wh.repeat(num_nodes, 1)
        a_input = torch.cat([Wh1, Wh2], dim=1).view(num_nodes, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        return e

class GATEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0, alpha=0.2):
        super(GATEncoder, self).__init__()
        self.gat1 = GATLayer(in_features, hidden_features, dropout=dropout, alpha=alpha, concat=True)
        self.gat2 = GATLayer(hidden_features, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = self.gat2(x, adj)
        return x  # [num_nodes, out_features]
