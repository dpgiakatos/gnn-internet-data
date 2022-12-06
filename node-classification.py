import pickle
import itertools
from tqdm import tqdm
from modules.gnn import GNN
from modules.models import GraphSAGE, GCN, GAT, GIN, MLP
from modules.predictors import MLPPredictor
from torch.optim import Adam
from tabulate import tabulate
from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")


res = dict()
data = []

# Set dataset directory
path = ''

experiments = {
    'AS_hegemony': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline'],
    'AS_rank_continent': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline'],
    'peeringDB_info_ratio': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline'],
    'peeringDB_info_scope': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline'],
    'peeringDB_info_type': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline'],
    'peeringDB_policy_general': ['with-features', 'without-features', 'node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings', 'baseline']
}

for index, experiment in enumerate(experiments):
    res[experiment] = dict()
    for variation in experiments[experiment]:
        print(f'Experiment: {index+1} of {len(experiments)}, Variation: {experiments[experiment].index(variation)+1} of {len(experiments[experiment])}')
        res[experiment][variation] = dict()

        if experiment in ['AS_hegemony']:
            gnn = GNN('node_classification', 'RMSE')
        else:
            gnn = GNN('node_classification', 'AC')
        gnn.load_dataset(path + f'dataset/node-classification/{experiment}/{variation}/', force_reload=True)
        gnn.split_dataset(0.1)

        for nn_model in tqdm([GraphSAGE, GCN, GAT, GIN, MLP, RandomForestClassifier, RandomForestRegressor]):
            err_rmse = []
            err_mape = []
            met_acc = []
            met_f1 = []
            res[experiment][variation][str(nn_model)] = {
                'iterations': dict()
            }
            for i in range(10):
                if variation == 'baseline' and nn_model in [RandomForestClassifier, RandomForestRegressor]:
                    if experiment not in ['AS_hegemony'] and nn_model == RandomForestRegressor:
                        break
                    model = nn_model()
                    try:
                        model = gnn.train_with_baseline(model)
                    except:
                        break

                    metric, (label, score) = gnn.score_with_baseline(model)
                elif variation in ['with-features', 'without-features'] and nn_model in [GraphSAGE, GCN, GAT, GIN]:
                    model = nn_model(gnn.get_train_shape()[1], 32)
                    try:
                        predictor = MLPPredictor(32, gnn.get_label_shape()[1])
                    except:
                        predictor = MLPPredictor(32, 1)
                    optimizer = Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=0.01)

                    gnn.train(model, predictor, optimizer, epochs=300)
                    metric, (label, score) = gnn.score(model, predictor)
                elif variation in ['node2vec-embeddings16-10-100', 'node2vec-embeddings16-50-200', 'node2vec-embeddings16-wl4-nm20', 'node2vec-embeddings16-wl40-nm150', 'bgp2vec-embeddings'] and nn_model == MLP:
                    try:
                        model = nn_model(gnn.get_train_shape()[1], gnn.get_label_shape()[1])
                    except:
                        model = nn_model(gnn.get_train_shape()[1], 1)
                    optimizer = Adam(itertools.chain(model.parameters()), lr=0.01)

                    gnn.train_with_embeddings(model, optimizer, epochs=300)
                    metric, (label, score) = gnn.score_with_embeddings(model)
                    nodes_src, nodes_dst = [], []
                else:
                    break

                res[experiment][variation][str(nn_model)]['iterations'][i+1] = {'label': label, 'predict': score}
                if experiment in ['AS_hegemony']:
                    err_rmse.append(metric[0])
                    err_mape.append(metric[1])
                else:
                    met_acc.append(metric[0])
                    met_f1.append(metric[1])
            if len(met_acc) or len(err_rmse):
                if experiment in ['AS_hegemony']:
                    d = [None, None, None, None, min(err_rmse), sum(err_rmse) / 10, min(err_mape), sum(err_mape) / 10]
                else:
                    d = [max(met_acc), sum(met_acc) / 10, max(met_f1), sum(met_f1) / 10, None, None, None, None]
                data.append([str(nn_model), experiment, variation] + d)


with open(path + 'results/node-classification.pickle', 'wb') as f:
    pickle.dump(res, f)

table = tabulate(data, headers=['Model', 'Experiment', 'Variation', 'Max ACC', 'Avg ACC', 'Max macro-F1', 'Avg macro-F1', 'Min RMSE', 'Avg RMSE', 'Min MAPE', 'Avg MAPE'])

with open(path + 'results/node-classification.txt', 'w') as f:
    f.write(table)

print(table)
