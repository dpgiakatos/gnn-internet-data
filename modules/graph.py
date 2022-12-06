import networkx as nx
import pandas as pd
from random import sample


class Graph:
    def __init__(self):
        self.graph = None

    def read_from_edgelist(self, filename):
        self.graph = nx.read_edgelist(filename, delimiter=',', nodetype=int, comments='src_id,dst_id')
        print(self.graph)

    def create_subgraph(self, percentage):
        nodes_subset = sample(self.graph.nodes(), round(len(self.graph.nodes()) * percentage))
        self.graph = self.graph.subgraph(nodes_subset)
        print(self.graph)

    def delete_nodes_with_degree(self, degree):
        remove_set = [node for node, deg in dict(self.graph.degree()).items() if deg <= degree]
        self.graph.remove_nodes_from(remove_set)
        print(self.graph)

    def get_connected_components(self):
        self.graph = self.graph.subgraph(max(nx.connected_components(self.graph), key=len))

    def get_graph(self):
        return self.graph

    @staticmethod
    def write_edgelist(graph, filename):
        def line_prepender(file, line):
            with open(file, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write(line.rstrip('\r\n') + '\n' + content)

        nx.write_edgelist(graph, filename, delimiter=',', data=False)
        line_prepender(filename, 'src_id,dst_id')

    @staticmethod
    def write_nodelist(graph, path, features_filename, label=None, to_merge=False, embeddings_filename=None):
        if embeddings_filename is None:
            features = pd.read_csv(features_filename)
        else:
            embeddings = pd.read_csv(embeddings_filename)
            col = ['ASN']
            col.extend(label)
            labels = pd.read_csv(features_filename).loc[:, col]
            features = pd.merge(embeddings, labels, on="ASN")
        if label is not None:
            col = ['ASN']
            col.extend(label)
            labels = features.loc[:, col]
            if to_merge:
                col_to_merge = []
                for col in label:
                    if 'None' not in col and labels[col].value_counts()[1] < 500:
                        col_to_merge.append(col)
                new_col = []
                for index, row in labels.iterrows():
                    merged = False
                    for col in col_to_merge:
                        if row[col] == 1:
                            new_col.append(1.0)
                            merged = True
                    if not merged:
                        new_col.append(0.0)
                labels = labels.drop(col_to_merge, axis=1)
                labels['_'.join(col_to_merge)] = new_col
            features = features.drop(label, axis=1)
        f = open(path + '/node_feature_export.csv', 'w')
        fn = open(path + '/node_featureless_export.csv', 'w')
        if label is not None:
            w = 'node_id,feat,label\n'
        else:
            w = 'node_id,feat\n'
        f.write(w)
        fn.write(w)
        for node in graph.nodes():
            node_features = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            node_features = ', '.join([str(feature) for feature in node_features])
            w = f'{str(node)},"{node_features}"\n'
            wn = f'{str(node)},"1.0,0.0"\n'
            if label is not None:
                node_labels = labels.loc[labels['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
                node_labels = ', '.join([str(n_label) for n_label in node_labels])
                w = f'{str(node)},"{node_features}","{node_labels}"\n'
                wn = f'{str(node)},"1.0,0.0","{node_labels}"\n'
            f.write(w)
            fn.write(wn)

    @staticmethod
    def write_nodelist_with_subset(graph, path, features_filename, label=None, to_merge=False, embeddings_filename=None):
        if embeddings_filename is None:
            features = pd.read_csv(features_filename)
        else:
            embeddings = pd.read_csv(embeddings_filename)
            col = ['ASN']
            col.extend(label)
            labels = pd.read_csv(features_filename).loc[:, col]
            features = pd.merge(embeddings, labels, on="ASN")
        if label is not None:
            col = ['ASN']
            col.extend(label)
            labels = features.loc[:, col]
            if to_merge:
                col_to_merge = []
                for col in label:
                    if 'None' not in col and labels[col].value_counts()[1] < 500:
                        col_to_merge.append(col)
                new_col = []
                for index, row in labels.iterrows():
                    merged = False
                    for col in col_to_merge:
                        if row[col] == 1:
                            new_col.append(1.0)
                            merged = True
                    if not merged:
                        new_col.append(0.0)
                labels = labels.drop(col_to_merge, axis=1)
                labels['_'.join(col_to_merge)] = new_col
            features = features.drop(label, axis=1)
        f = open(path + '/node_feature_export.csv', 'w')
        fn = open(path + '/node_featureless_export.csv', 'w')
        if label is not None:
            w = 'node_id,feat,label\n'
        else:
            w = 'node_id,feat\n'
        f.write(w)
        fn.write(w)
        new_graph = graph.copy()
        nodes_to_remove = []
        for node in graph.nodes():
            try:
                exists = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            except:
                nodes_to_remove.append(node)
        new_graph.remove_nodes_from(nodes_to_remove)
        for node in new_graph.nodes():
            node_features = features.loc[features['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
            node_features = ', '.join([str(feature) for feature in node_features])
            w = f'{str(node)},"{node_features}"\n'
            wn = f'{str(node)},"1.0,0.0"\n'
            if label is not None:
                node_labels = labels.loc[labels['ASN'] == node].fillna(0).to_numpy()[0].tolist()[1:]
                node_labels = ', '.join([str(n_label) for n_label in node_labels])
                w = f'{str(node)},"{node_features}","{node_labels}"\n'
                wn = f'{str(node)},"1.0,0.0","{node_labels}"\n'
            f.write(w)
            fn.write(wn)
        nx.write_edgelist(new_graph, path+'edges_export.csv', data=False, delimiter=',')
