import dgl
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_percentage_error, f1_score, precision_score, recall_score
from sklearnex import patch_sklearn, config_context
from imblearn.under_sampling import RandomUnderSampler


class GNN:
    def __init__(self, task, metric, debug=False):
        self.task = task
        self.metric = metric
        self.debug = debug
        if task == 'link_prediction':
            if metric != 'AC':
                raise Exception('The support metric of link_prediction is `AC`')
        elif task == 'node_classification':
            if metric not in ['RMSE', 'AC']:
                raise Exception('The support metrics of node_classification are `RMSE` and `AC`')
        else:
            raise Exception('The task should be `node_classification` or `link_prediction`')
        self.graph = None
        self.train_mask = None
        self.test_mask = None
        self.train_graph = None
        self.test_graph = None
        self.train_pos_graph = None
        self.train_neg_graph = None
        self.test_pos_graph = None
        self.test_neg_graph = None
        self.train_set = None
        self.test_set = None
        if torch.cuda.is_available():
            patch_sklearn()

    def load_dataset(self, data_path, force_reload=False):
        self.graph = dgl.data.CSVDataset(data_path, force_reload=force_reload)[0]
        if self.debug:
            print(self.graph)

    @staticmethod
    def create_dataset_baseline(graph, features):
        u, v = graph.edges()
        x = []
        for i in range(len(u)):
            x.append(features[u[i].item()].tolist() + features[v[i].item()].tolist())
        return x

    def split_dataset(self, percentage):
        if self.task == 'node_classification':
            node_ids = self.graph.nodes().tolist()
            test_size = int(len(node_ids) * percentage)

            self.train_graph = dgl.remove_nodes(self.graph, node_ids[:test_size])
            self.test_graph = dgl.remove_nodes(self.graph, node_ids[test_size:])

            X_res = []
            if len(self.get_label_shape()) > 1:
                rus = RandomUnderSampler(random_state=42)
                X_res, _ = rus.fit_resample(np.reshape(node_ids[test_size:], (-1, 1)), np.argmax(self.train_graph.ndata['label'].tolist(), axis=1))

            self.train_mask = []
            self.test_mask = []
            for val in self.graph.nodes().tolist():
                if type(self.graph.ndata['label'][val].tolist()) == list and np.argmax(self.graph.ndata['label'][val].tolist()) == 0:
                    self.train_mask.append(False)
                    self.test_mask.append(False)
                elif val in node_ids[:test_size]:
                    self.train_mask.append(False)
                    self.test_mask.append(True)
                elif len(self.get_label_shape()) == 1 and val in node_ids[test_size:]:
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                elif len(self.get_label_shape()) > 1 and val in X_res.flatten():
                    self.train_mask.append(True)
                    self.test_mask.append(False)
                else:
                    self.train_mask.append(False)
                    self.test_mask.append(False)

            self.train_mask = torch.tensor(self.train_mask)
            self.test_mask = torch.tensor(self.test_mask)

            self.train_graph = self.graph

            if len(self.get_label_shape()) > 1:
                self.train_graph.ndata['label'] = np.delete(self.train_graph.ndata['label'], 0, 1)


            if torch.cuda.is_available():
                self.graph = self.graph.to('cuda')
                self.train_mask = self.train_mask.to('cuda')
                self.test_mask = self.test_mask.to('cuda')
                self.train_graph = self.train_graph.to('cuda')
                self.test_graph = self.test_graph.to('cuda')
        elif self.task == 'link_prediction':
            u, v = self.graph.edges()

            edge_ids = np.arange(self.graph.number_of_edges())
            edge_ids = np.random.permutation(edge_ids)
            test_size = int(len(edge_ids) * percentage)
            test_pos_u, test_pos_v = u[edge_ids[:test_size]], v[edge_ids[:test_size]]
            train_pos_u, train_pos_v = u[edge_ids[test_size:]], v[edge_ids[test_size:]]

            number_of_nodes = self.graph.number_of_nodes()
            adjacency = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())), shape=(number_of_nodes, number_of_nodes))
            adjacency_neg = 1 - adjacency.todense() - np.eye(number_of_nodes)
            neg_u, neg_v = np.where(adjacency_neg != 0)

            neg_edge_ids = np.random.choice(len(neg_u), self.graph.number_of_edges())
            test_neg_u, test_neg_v = neg_u[neg_edge_ids[:test_size]], neg_v[neg_edge_ids[:test_size]]
            train_neg_u, train_neg_v = neg_u[neg_edge_ids[test_size:]], neg_v[neg_edge_ids[test_size:]]

            self.train_graph = dgl.remove_edges(self.graph, edge_ids[:test_size])

            self.train_pos_graph = dgl.graph((train_pos_u, train_pos_v), num_nodes=number_of_nodes)
            self.train_neg_graph = dgl.graph((train_neg_u, train_neg_v), num_nodes=number_of_nodes)

            self.test_pos_graph = dgl.graph((test_pos_u, test_pos_v), num_nodes=number_of_nodes)
            self.test_neg_graph = dgl.graph((test_neg_u, test_neg_v), num_nodes=number_of_nodes)

            train_pos = self.create_dataset_baseline(self.train_pos_graph, self.train_graph.ndata['feat'])
            train_neg = self.create_dataset_baseline(self.train_neg_graph, self.train_graph.ndata['feat'])
            train_pos_neg = pd.DataFrame(list(zip(train_pos + train_neg, torch.ones(len(train_pos)).tolist() + torch.zeros(len(train_neg)).tolist())), columns=['feat', 'label'])
            self.train_set = train_pos_neg.sample(frac=1).reset_index(drop=True)

            test_pos = self.create_dataset_baseline(self.test_pos_graph, self.train_graph.ndata['feat'])
            test_neg = self.create_dataset_baseline(self.test_neg_graph, self.train_graph.ndata['feat'])
            test_pos_neg = pd.DataFrame(list(zip(test_pos + test_neg, torch.ones(len(test_pos)).tolist() + torch.zeros(len(test_neg)).tolist())), columns=['feat', 'label'])
            self.test_set = test_pos_neg.sample(frac=1).reset_index(drop=True)


            if torch.cuda.is_available():
                self.train_graph = self.train_graph.to('cuda')

                self.train_pos_graph = self.train_pos_graph.to('cuda')
                self.train_neg_graph = self.train_neg_graph.to('cuda')

                self.test_pos_graph = self.test_pos_graph.to('cuda')
                self.test_neg_graph = self.test_neg_graph.to('cuda')

    def compute_loss(self, scores=None, labels=None, pos_score=None, neg_score=None):
        if self.task == 'link_prediction':
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        if torch.cuda.is_available():
            scores = scores.to('cuda')
            labels = labels.to('cuda')
        if self.task == 'link_prediction':
            return F.binary_cross_entropy_with_logits(scores, labels)
        else:
            return F.cross_entropy(scores, labels)

    def compute_metric(self, scores=None, labels=None, pos_score=None, neg_score=None):
        if self.task == 'link_prediction' and pos_score is not None and neg_score is not None:
            scores = torch.cat([pos_score, neg_score])
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
        if torch.cuda.is_available() and str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
            scores = scores.to('cpu')
            labels = labels.to('cpu')
        if self.metric == 'AC':
            if self.task == 'link_prediction':
                if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                    return (roc_auc_score(labels.numpy(), scores.numpy()), precision_score(labels.numpy(), np.rint(scores.numpy())), recall_score(labels.numpy(), np.rint(scores.numpy()))), (labels.numpy(), np.rint(scores.numpy()))
                else:
                    return (roc_auc_score(labels, scores), precision_score(labels, np.rint(scores)), recall_score(labels, np.rint(scores))), (labels, np.rint(scores))
            else:
                if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                    return (accuracy_score(np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1)), f1_score(np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1), average='macro')), (np.argmax(labels.numpy(), axis=1), np.argmax(scores.numpy(), axis=1))
                else:
                    return (accuracy_score(np.argmax(labels, axis=1), np.argmax(scores, axis=1)), f1_score(np.argmax(labels, axis=1), np.argmax(scores, axis=1), average='macro')), (np.argmax(labels, axis=1), np.argmax(scores, axis=1))
        elif self.metric == 'RMSE':
            if str(type(scores)) != "<class 'numpy.ndarray'>" and str(type(labels)) != "<class 'numpy.ndarray'>":
                return (mean_squared_error(labels.numpy(), scores.numpy()), mean_absolute_percentage_error(labels.numpy(), scores.numpy())), (labels.numpy(), scores.numpy())
            else:
                return (mean_squared_error(labels, scores), mean_absolute_percentage_error(labels, scores)), (labels, scores)

    def train(self, model, predictor, optimizer, epochs=100):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        for e in range(1, epochs+1):
            # forward
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            if self.task == 'node_classification':
                score = predictor(self.train_graph, h)
                loss = self.compute_loss(scores=score[self.train_mask], labels=self.train_graph.ndata['label'][self.train_mask])
            elif self.task == 'link_prediction':
                pos_score, _ = predictor(self.train_pos_graph, h)
                neg_score, _ = predictor(self.train_neg_graph, h)
                loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 5 == 0 and self.debug:
                print(f'In epoch {e}, loss: {loss}')

    def train_with_embeddings(self, model, optimizer, epochs=100):
        if torch.cuda.is_available():
            model = model.to('cuda')
        for e in range(1, epochs+1):
            if self.task == 'node_classification':
                score = model(self.train_graph, self.train_graph.ndata['feat'])
                loss = self.compute_loss(scores=score[self.train_mask], labels=self.train_graph.ndata['label'][self.train_mask])
            elif self.task == 'link_prediction':
                pos_score = model(self.train_pos_graph, self.train_graph.ndata['feat'])
                neg_score = model(self.train_neg_graph, self.train_graph.ndata['feat'])
                loss = self.compute_loss(pos_score=pos_score, neg_score=neg_score)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 5 == 0 and self.debug:
                print(f'In epoch {e}, loss: {loss}')

    def train_with_baseline(self, model):
        with config_context(target_offload='auto'):
            if self.task == 'link_prediction':
                model = model.fit(self.train_set['feat'].tolist(), self.train_set['label'].tolist())
            elif self.task == 'node_classification':
                model = model.fit(self.train_graph.ndata['feat'][self.train_mask].tolist(), self.train_graph.ndata['label'][self.train_mask].tolist())
        return model

    def score(self, model, predictor):
        if torch.cuda.is_available():
            model = model.to('cuda')
            predictor = predictor.to('cuda')
        with torch.no_grad():
            h = model(self.train_graph, self.train_graph.ndata['feat'])
            if self.task == 'node_classification':
                score = predictor(self.train_graph, h)
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                return computed_metric, res
            elif self.task == 'link_prediction':
                pos_score, (pos_src, pos_dst) = predictor(self.test_pos_graph, h)
                neg_score, (neg_src, neg_dst) = predictor(self.test_neg_graph, h)
                nodes_src = torch.cat([pos_src, neg_src]).tolist()
                nodes_dst = torch.cat([pos_dst, neg_dst]).tolist()
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(pos_score=pos_score, neg_score=neg_score)
                return computed_metric, res, (nodes_src, nodes_dst)

    def score_with_embeddings(self, model):
        if torch.cuda.is_available():
            model = model.to('cuda')
        with torch.no_grad():
            if self.task == 'node_classification':
                score = model(self.train_graph, self.train_graph.ndata['feat'])
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=score[self.test_mask], labels=self.train_graph.ndata['label'][self.test_mask])
                return computed_metric, res
            elif self.task == 'link_prediction':
                pos_score = model(self.test_pos_graph, self.train_graph.ndata['feat'])
                neg_score = model(self.test_neg_graph, self.train_graph.ndata['feat'])
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(pos_score=pos_score, neg_score=neg_score)
            return computed_metric, res

    def score_with_baseline(self, model):
        with config_context(target_offload='auto'):
            if self.task == 'link_prediction':
                pred = model.predict(self.test_set['feat'].tolist())
                if self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=pred, labels=self.test_set['label'].tolist())
            elif self.task == 'node_classification':
                pred = model.predict(self.train_graph.ndata['feat'][self.test_mask].tolist())
                if self.metric == 'RMSE':
                    computed_metric, res = self.compute_metric(scores=pred, labels=self.train_graph.ndata['label'][self.test_mask].tolist())
                elif self.metric == 'AC':
                    computed_metric, res = self.compute_metric(scores=pred, labels=self.train_graph.ndata['label'][self.test_mask].tolist())
        return computed_metric, res

    def get_train_shape(self):
        return self.train_graph.ndata['feat'].shape

    def get_label_shape(self):
        return self.train_graph.ndata['label'].shape
