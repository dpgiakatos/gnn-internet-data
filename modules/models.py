import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
import dgl
from dgl.nn.pytorch import SAGEConv, GraphConv, GATConv, GINConv


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)

    def forward(self, g, in_feat):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, num_heads)
        self.conv2 = GATConv(h_feats, h_feats, num_heads)

    def forward(self, g, in_feat):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, in_feat)
        # h = torch.flatten(h, start_dim=1, end_dim=2) # only for 1 layer
        h = F.relu(h)
        h = self.conv2(g, h)
        h = torch.flatten(h, start_dim=1, end_dim=3)
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GIN, self).__init__()
        self.linear1 = nn.Linear(in_feats, h_feats)
        self.conv1 = GINConv(self.linear1)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.conv2 = GINConv(self.linear2)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class MLP(nn.Module):
    def __init__(self, h_feats, num_classes=None):
        super(MLP, self).__init__()
        if num_classes is None:
            self.W1 = nn.Linear(h_feats * 2, h_feats)
            self.W2 = nn.Linear(h_feats, 1)
        else:
            self.W1 = nn.Linear(h_feats, h_feats * 10)
            self.W2 = nn.Linear(h_feats * 10, num_classes)
        self.num_classes = num_classes

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': torch.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def apply_nodes(self, nodes):
        h = torch.cat([nodes.data['h']], 1)
        return {'score': torch.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, in_feat):
        with g.local_scope():
            g.ndata['h'] = in_feat
            if self.num_classes is None:
                g.apply_edges(self.apply_edges)
                return g.edata['score']
            else:
                g.apply_nodes(self.apply_nodes)
                return g.ndata['score']
