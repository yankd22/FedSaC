import copy
import math
import random
import time
from FedSaC.test import compute_acc, compute_local_test_accuracy

import numpy as np
import torch
import torch.optim as optim

from FedSaC.config import get_args
from FedSaC.utils import aggregation_by_graph, update_graph_matrix_neighbor
from FedSaC.model import simplecnn, textcnn
from FedSaC.prepare_data import get_dataloader
from FedSaC.attack import *
from sklearn.decomposition import PCA
import glog as log
import sys
import os
import json



def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, pca):
    principal_list = []
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            log.info('>> Client %d test1 | (Pre) Personalized Test Acc: %.5f | Generalized Test Acc: %.5f', net_id, personalized_test_acc, generalized_test_acc)

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if net_id in cluster_models:
            cluster_model = cluster_models[net_id].cuda()
        
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
        

            if net_id in cluster_models:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)
            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            log.info('>> Client %d test2 | (Pre) Personalized Test Acc: %.5f | Generalized Test Acc: %.5f', net_id, personalized_test_acc, generalized_test_acc)
        log.info('>> concatenate')
        if net_id in benign_client_list:
            net.eval()
            net.cuda()
            feature_list = []
            with torch.no_grad():
                for x, _ in train_local_dl:
                    x = x.cuda()
                    feature = net.extract_feature(x)
                    feature_list.append(feature.cpu().numpy()) 
            feature_array = np.concatenate(feature_list, axis=0)
            pca.fit_transform(feature_array)
            orthogonal_basis = pca.components_
            principal_list.append(orthogonal_basis)
        log.info('>> concatenate end')
        net.to('cpu')
    return principal_list, np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


args, cfg = get_args()
if args.partition == "noniid_1":
    args.partition = "noniid"
    args.beta = 0.1
elif args.partition == "noniid_2":
    args.partition = "noniid"
    args.beta = 0.5



seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

k_principal = int(args.k_principal)
pca = PCA(n_components=k_principal)


n_party_per_round = 10
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
log.info('>> -------- Benign clients: %d --------', benign_client_list)

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn

hidden_dim = args.hidden_dim
global_model = model(hidden_dim, cfg['classes_size'])
global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(cfg['client_num']):
    local_models.append(model(hidden_dim, cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

for net in local_models:
    net.load_state_dict(global_parameters)

    
cluster_model_vectors = {}
total_round = cfg["comm_round"]
for round in range(total_round):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        log.info('>> Clients in this round : %d', party_list_this_round)
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}
    principal_list, mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, train_local_dls, val_local_dls, 
                                                                    test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, pca)
    
    total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
    fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
    if round/total_round > args.alpha_bound:
        matrix_alpha = 0
    else:
        matrix_alpha = args.matrix_alpha
    graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, principal_list, dw, fed_avg_freqs, matrix_alpha, 
                                                    args.matrix_beta, args.complementary_metric, args.difference_measure)   # Graph Matrix is not normalized yet
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)                                                    # Aggregation weight is normalized here

    log.info('>> (Current) Round %d | Local Per: %.5f', round, mean_personalized_acc)
    # print('-'*80)
open(os.path.join(args.exp_dir, 'done'), 'a').close()
 