import subprocess
import os
import random
import pickle
import torch

for split in ['train', 'val', 'test']:
    path = '/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/p_sub/{}'.format(split)
    files = os.listdir(path)
    ids = [x.split('_')[0] for x in files]
    ids = list(set(ids))
    name_dict = {}
    for i in ids:
        name_dict[i] = random.random()

    for name in files:
        name_id = name.split('_')[0]
        [mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k] = pickle.load(open(path+'/{}'.format(name), 'rb'))

        if name_dict[name_id] <= 0.8:
            destination_file = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/p_sub/train_split/{}".format(name)
        elif name_dict[name_id] <= 0.9:
            destination_file = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/p_sub/val_split/{}".format(name)
        else:
            destination_file = "/cluster/home/wenkai/dgl_graphormer_local/geom_drugs/data/p_sub/test_split/{}".format(name)
        with open(destination_file, "wb") as f:
            pickle.dump((mol, g, G, pos, dist, angle, edge_index, idx_i, idx_j, idx_k), f)



