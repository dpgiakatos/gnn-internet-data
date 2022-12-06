import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class MLPPredictor(nn.Module):
    def __init__(self, h_feats, num_classes=None):
        super().__init__()
        if num_classes is None:
            self.W1 = nn.Linear(h_feats * 2, h_feats)
            self.W2 = nn.Linear(h_feats, 1)
        else:
            self.W1 = nn.Linear(h_feats, h_feats * 10)
            self.W2 = nn.Linear(h_feats * 10, num_classes)
        self.num_classes = num_classes

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        src, dst, _ = edges.edges()
        return {
            'score': torch.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1),
            'src': src,
            'dst': dst
        }

    def apply_nodes(self, nodes):
        h = torch.cat([nodes.data['h']], 1)
        return {'score': torch.sigmoid(self.W2(F.relu(self.W1(h)))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            if self.num_classes is None:
                g.apply_edges(self.apply_edges)
                return g.edata['score'], (g.edata['src'], g.edata['dst'])
            else:
                g.apply_nodes(self.apply_nodes)
                return g.ndata['score']
