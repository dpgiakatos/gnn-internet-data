import pickle
import itertools
from tqdm import tqdm
from modules.gnn import GNN
from modules.models import GraphSAGE, GCN, GAT, GIN, MLP
from modules.predictors import MLPPredictor
from torch.optim import Adam
from tabulate import tabulate
from sklearnex.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


res = dict()
data = []

# Set dataset directory
path = ''

experiments = {
    'gnn': ['with-features', 'without-features'],
    'node2vec': ['embeddings16-10-100', 'embeddings16-50-200', 'embeddings16-wl40-nm150', 'embeddings16-wl4-nm20'],
    'bgp2vec': ['embeddings'],
    'baseline': ['node-features']
}

for index, experiment in enumerate(experiments):
    res[experiment] = dict()
    for variation in experiments[experiment]:
        print(f'Experiment: {index+1} of {len(experiments)}, Variation: {experiments[experiment].index(variation)+1} of {len(experiments[experiment])}')
        res[experiment][variation] = dict()

        gnn = GNN('link_prediction', 'AC')
        gnn.load_dataset(path + f'dataset/link-prediction/{experiment}/{variation}/', force_reload=True)
        gnn.split_dataset(0.1)

        for nn_model in tqdm([GraphSAGE, GCN, GAT, GIN, MLP, RandomForestClassifier]):
            auc = []
            pr = []
            re = []
            res[experiment][variation][str(nn_model)] = {
                'iterations': dict()
            }
            for i in range(10):
                if experiment == 'gnn' and nn_model in [GraphSAGE, GCN, GAT, GIN]:
                    model = nn_model(gnn.get_train_shape()[1], 16)
                    predictor = MLPPredictor(16)
                    optimizer = Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=0.01)

                    gnn.train(model, predictor, optimizer, epochs=100)
                    metric, (label, score), (nodes_src, nodes_dst) = gnn.score(model, predictor)
                elif experiment in ['node2vec', 'bgp2vec'] and nn_model == MLP:
                    model = nn_model(gnn.get_train_shape()[1])
                    optimizer = Adam(itertools.chain(model.parameters()), lr=0.01)

                    gnn.train_with_embeddings(model, optimizer, epochs=100)
                    metric, (label, score) = gnn.score_with_embeddings(model)
                    nodes_src, nodes_dst = [], []
                elif experiment == 'baseline' and nn_model in [RandomForestClassifier]:
                    model = nn_model()

                    model = gnn.train_with_baseline(model)
                    metric, (label, score) = gnn.score_with_baseline(model)
                    nodes_src, nodes_dst = [], []
                else:
                    break

                res[experiment][variation][str(nn_model)]['iterations'][i+1] = {'label': label, 'predict': score, 'src': nodes_src, 'dst': nodes_dst}
                auc.append(metric[0])
                pr.append(metric[1])
                re.append(metric[2])
            if len(auc):
                data.append([str(nn_model), experiment, variation, max(auc), sum(auc) / 10, max(pr), sum(pr) / 10, max(re), sum(re) / 10])


with open(path + 'results/link-prediction.pickle', 'wb') as f:
    pickle.dump(res, f)

table = tabulate(data, headers=['Model', 'Experiment', 'Variation', 'Max AUC', 'Avg AUC', 'Max Precision', 'Avg Precision', 'Max Recall', 'Avg Recall'])

with open(path + 'results/link-prediction.txt', 'w') as f:
    f.write(table)

print(table)
